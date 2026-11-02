"""
Agente LiveKit del gemelo digital de Héctor Padilla
===================================================

    Micrófono del usuario ──LiveKit──> Deepgram (voz a texto)
                                            │
                          Base44 (conocimiento, timeout corto)
                                            │
                                      Claude (cerebro)
                                            │
                    ElevenLabs (TTS con la VOZ CLONADA de Héctor)
                                            │
                      LiveKit data stream ──> Anam (cara + lip sync)
                                            │
                 video + audio ──LiveKit──> navegador del usuario

Lo que se conserva del orquestador actual: Claude, el system prompt, el
conocimiento de Base44 y la voz de Héctor. Lo que cambia: LiveKit transporta
el audio y Anam solo pone la cara (ya no hay viaje por el navegador).

Correr localmente (desde esta carpeta):
    python agent.py dev

Variables de entorno: ver .env.example
"""

import logging
import os
import time
from typing import Optional

import httpx
from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    ChatContext,
    ChatMessage,
    JobContext,
    TurnHandlingOptions,
    cli,
    inference,
)
from livekit.plugins import anam, anthropic, deepgram, elevenlabs

from system_prompt import SYSTEM_PROMPT

load_dotenv(".env.local")
load_dotenv()

logger = logging.getLogger("gemelo_hp_agent")
logger.setLevel(logging.INFO)


# ---------------------------------------------------------------------------
# Configuración (todo por variables de entorno)
# ---------------------------------------------------------------------------

def _require(nombre: str) -> str:
    valor = os.environ.get(nombre, "").strip()
    if not valor:
        raise RuntimeError(
            f"Falta la variable de entorno {nombre}. Revisa agent/.env.example"
        )
    return valor


def _env_float(nombre: str) -> Optional[float]:
    valor = os.environ.get(nombre, "").strip()
    return float(valor) if valor else None


def _env_bool(nombre: str, default: bool) -> bool:
    valor = os.environ.get(nombre, "").strip().lower()
    if not valor:
        return default
    return valor in ("1", "true", "yes", "si", "sí")


# Anam (cara). Mismo avatar que ya usas hoy.
ANAM_AVATAR_ID = _require("ANAM_AVATAR_ID")
# Si tu avatar no soporta este modelo, se reintenta solo, sin avatarModel.
ANAM_AVATAR_MODEL = os.environ.get("ANAM_AVATAR_MODEL", "cara-4.1-express").strip()

# Voz de Héctor. OJO: ANAM_VOICE_ID NO sirve aquí; en la integración con
# LiveKit la voz sale del TTS del agente, así que hace falta el ID de la voz
# clonada EN ElevenLabs.
ELEVENLABS_VOICE_ID = _require("ELEVENLABS_VOICE_ID")
ELEVENLABS_MODEL = os.environ.get("ELEVENLABS_MODEL", "eleven_turbo_v2_5")

CLAUDE_MODEL = os.environ.get("CLAUDE_MODEL", "claude-sonnet-5")
CLAUDE_MAX_TOKENS = int(os.environ.get("CLAUDE_MAX_TOKENS", "300"))
CLAUDE_TEMPERATURE = _env_float("CLAUDE_TEMPERATURE")  # vacío = default de Anthropic

# "multi" entiende español e inglés en la misma sesión. Pon "es" si solo
# hablas español (suele transcribir un poco mejor).
STT_LANGUAGE = os.environ.get("STT_LANGUAGE", "multi")
# Palabras que Deepgram suele fallar, separadas por coma. Opcional.
STT_KEYTERMS = [t.strip() for t in os.environ.get("STT_KEYTERMS", "").split(",") if t.strip()]

# Base44 (conocimiento). Opcional: sin URL, el gemelo responde solo con su prompt.
BASE44_APP_URL = os.environ.get("BASE44_APP_URL", "").rstrip("/")
BASE44_SERVICE_TOKEN = os.environ.get("BASE44_SERVICE_TOKEN", "")
# Se suma directo a la latencia: el LLM no arranca hasta que Base44 responde.
BASE44_TIMEOUT_SECONDS = float(os.environ.get("BASE44_TIMEOUT_SECONDS", "2.0"))
KNOWLEDGE_MIN_WORDS = int(os.environ.get("KNOWLEDGE_MIN_WORDS", "3"))

# Fin de turno. Vacío = usa los valores por defecto de LiveKit.
MIN_ENDPOINTING_DELAY = _env_float("MIN_ENDPOINTING_DELAY")
MAX_ENDPOINTING_DELAY = _env_float("MAX_ENDPOINTING_DELAY")
# Generación anticipada: LiveKit empieza a generar con la transcripción
# parcial. Si Base44 agrega contexto, esa generación se descarta y se repite,
# así que aquí arranca apagada. Pruébala encendida si Base44 casi no se usa.
PREEMPTIVE_GENERATION = _env_bool("PREEMPTIVE_GENERATION", False)


# ---------------------------------------------------------------------------
# Conocimiento de Base44
# ---------------------------------------------------------------------------

async def obtener_conocimiento_base44(http: httpx.AsyncClient, pregunta: str) -> str:
    """
    Pide a Base44 (getKnowledgeForQuery) el conocimiento relevante.
    Nunca lanza excepciones: si falla o tarda, regresa "" y el gemelo
    responde sin ese contexto.
    """
    if not BASE44_APP_URL or len(pregunta.split()) < KNOWLEDGE_MIN_WORDS:
        return ""

    headers = {"Content-Type": "application/json"}
    if BASE44_SERVICE_TOKEN:
        headers["Authorization"] = f"Bearer {BASE44_SERVICE_TOKEN}"

    inicio = time.perf_counter()
    try:
        response = await http.post(
            f"{BASE44_APP_URL}/api/functions/getKnowledgeForQuery",
            headers=headers,
            json={"query": pregunta},
            timeout=BASE44_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        conocimiento = response.json().get("knowledge", "") or ""
        logger.info("Base44 respondió en %.0f ms", (time.perf_counter() - inicio) * 1000)
        return conocimiento
    except Exception as exc:  # noqa: BLE001 - nada de Base44 debe tumbar un turno
        logger.warning(
            "Base44 falló o tardó (%.0f ms): %s", (time.perf_counter() - inicio) * 1000, exc
        )
        return ""


# ---------------------------------------------------------------------------
# El agente
# ---------------------------------------------------------------------------

class GemeloHP(Agent):
    def __init__(self, http: httpx.AsyncClient) -> None:
        super().__init__(instructions=SYSTEM_PROMPT)
        self._http = http

    async def on_user_turn_completed(
        self, turn_ctx: ChatContext, new_message: ChatMessage
    ) -> None:
        """
        Corre cuando el usuario termina de hablar y ANTES de que Claude genere.
        Si este método lanza una excepción, LiveKit descarta el turno y el
        gemelo no contesta, por eso todo va dentro de try/except.
        """
        try:
            pregunta = new_message.text_content or ""
            conocimiento = await obtener_conocimiento_base44(self._http, pregunta)
            if conocimiento:
                # Mismo patrón que la documentación de LiveKit para RAG.
                # Este mensaje queda ANTES del mensaje del usuario.
                turn_ctx.add_message(
                    role="assistant",
                    content=(
                        "Conocimiento relevante de la base de Héctor para la siguiente "
                        f"pregunta (úsalo solo si aplica):\n{conocimiento}"
                    ),
                )
        except Exception:  # noqa: BLE001
            logger.exception("Falló la inyección de conocimiento; se responde sin él")


# ---------------------------------------------------------------------------
# Construcción de la sesión y del avatar
# ---------------------------------------------------------------------------

def construir_sesion() -> AgentSession:
    turn_handling = TurnHandlingOptions(
        preemptive_generation={"enabled": PREEMPTIVE_GENERATION},
    )
    endpointing = {}
    if MIN_ENDPOINTING_DELAY is not None:
        endpointing["min_delay"] = MIN_ENDPOINTING_DELAY
    if MAX_ENDPOINTING_DELAY is not None:
        endpointing["max_delay"] = MAX_ENDPOINTING_DELAY
    if endpointing:
        turn_handling["endpointing"] = endpointing

    stt_kwargs = {"keyterm": STT_KEYTERMS} if STT_KEYTERMS else {}
    llm_kwargs = {"temperature": CLAUDE_TEMPERATURE} if CLAUDE_TEMPERATURE is not None else {}

    return AgentSession(
        stt=deepgram.STT(model="nova-3", language=STT_LANGUAGE, **stt_kwargs),
        llm=anthropic.LLM(model=CLAUDE_MODEL, max_tokens=CLAUDE_MAX_TOKENS, **llm_kwargs),
        tts=elevenlabs.TTS(voice_id=ELEVENLABS_VOICE_ID, model=ELEVENLABS_MODEL),
        vad=inference.VAD(model="silero"),
        turn_handling=turn_handling,
    )


def construir_avatar(modelo: Optional[str]) -> anam.AvatarSession:
    return anam.AvatarSession(
        persona_config=anam.PersonaConfig(
            name="Gemelo Digital de Héctor Padilla",
            avatarId=ANAM_AVATAR_ID,
            avatarModel=modelo or None,
        ),
    )


async def iniciar_avatar(session: AgentSession, room) -> None:
    """
    Arranca el avatar de Anam. Si Anam rechaza el modelo pedido (por ejemplo un
    avatar viejo que no soporta cara-4.1-express), reintenta una vez con el
    modelo por defecto del avatar en vez de dejar la sesión sin cara.
    """
    modelos = [ANAM_AVATAR_MODEL, None] if ANAM_AVATAR_MODEL else [None]
    ultimo_error: Optional[Exception] = None
    for modelo in modelos:
        try:
            await construir_avatar(modelo).start(session, room=room)
            logger.info("Avatar de Anam listo (avatarModel=%s)", modelo or "default del avatar")
            return
        except Exception as exc:  # noqa: BLE001
            ultimo_error = exc
            logger.warning("Anam no pudo iniciar con avatarModel=%s: %s", modelo, exc)
    assert ultimo_error is not None
    raise ultimo_error


def _ms(segundos: Optional[float]) -> str:
    return "n/d" if segundos is None else f"{segundos * 1000:.0f} ms"


def registrar_latencias(session: AgentSession) -> None:
    """Imprime, por cada respuesta del gemelo, dónde se fue el tiempo."""

    @session.on("conversation_item_added")
    def _al_agregar_mensaje(ev) -> None:
        item = ev.item
        metricas = getattr(item, "metrics", None)
        if getattr(item, "role", None) != "assistant" or not metricas:
            return
        logger.info(
            "LATENCIA | total=%s | fin de turno=%s | Base44+hook=%s | Claude 1er token=%s | "
            "TTS 1er audio=%s | reproducción=%s",
            _ms(metricas.get("e2e_latency")),
            _ms(metricas.get("end_of_turn_delay")),
            _ms(metricas.get("on_user_turn_completed_delay")),
            _ms(metricas.get("llm_node_ttft")),
            _ms(metricas.get("tts_node_ttfb")),
            _ms(metricas.get("playback_latency")),
        )


# ---------------------------------------------------------------------------
# Punto de entrada
# ---------------------------------------------------------------------------

server = AgentServer()


@server.rtc_session()  # sin agent_name: se asigna solo a cada sala nueva
async def entrypoint(ctx: JobContext) -> None:
    ctx.log_context_fields = {"room": ctx.room.name}

    http = httpx.AsyncClient(timeout=BASE44_TIMEOUT_SECONDS)
    ctx.add_shutdown_callback(http.aclose)

    session = construir_sesion()
    registrar_latencias(session)

    # El avatar se inicia ANTES que la sesión: así el audio del TTS va a Anam
    # y no directo al navegador.
    await iniciar_avatar(session, ctx.room)
    await session.start(agent=GemeloHP(http), room=ctx.room)


if __name__ == "__main__":
    cli.run_app(server)
