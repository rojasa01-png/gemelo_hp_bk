"""
Piezas compartidas por los dos agentes:

  - agent.py           -> Deepgram + Claude + ElevenLabs + Anam
  - agent_realtime.py  -> OpenAI Realtime (texto) + ElevenLabs + Anam

Aquí vive lo que es igual en ambos: configuración por variables de entorno,
la voz de Héctor (ElevenLabs), el conocimiento de Base44, el arranque del
avatar de Anam y el registro de latencias.
"""

import logging
import os
import time
from typing import Optional

import httpx
from dotenv import load_dotenv
from livekit.agents import AgentSession
from livekit.plugins import anam, elevenlabs

load_dotenv(".env.local")
load_dotenv()

logger = logging.getLogger("gemelo_hp_agent")
logger.setLevel(logging.INFO)


# ---------------------------------------------------------------------------
# Utilidades de configuración
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


# ---------------------------------------------------------------------------
# Configuración compartida
# ---------------------------------------------------------------------------

# Anam (cara). Mismo avatar que ya usas hoy.
_require("ANAM_API_KEY")  # el plugin la lee solo; aquí se exige para fallar temprano
ANAM_AVATAR_ID = _require("ANAM_AVATAR_ID")
# Si tu avatar no soporta este modelo, se reintenta solo, sin avatarModel.
ANAM_AVATAR_MODEL = os.environ.get("ANAM_AVATAR_MODEL", "cara-4.1-express").strip()

# Voz de Héctor. OJO: ANAM_VOICE_ID NO sirve aquí; en la integración con
# LiveKit la voz sale del TTS del agente, así que hace falta el ID de la voz
# clonada EN ElevenLabs (y ELEVEN_API_KEY, que el plugin lee solo).
ELEVENLABS_VOICE_ID = _require("ELEVENLABS_VOICE_ID")
ELEVENLABS_MODEL = os.environ.get("ELEVENLABS_MODEL", "eleven_turbo_v2_5")

# Base44 (conocimiento). Opcional: sin URL, el gemelo responde solo con su prompt.
BASE44_APP_URL = os.environ.get("BASE44_APP_URL", "").rstrip("/")
BASE44_SERVICE_TOKEN = os.environ.get("BASE44_SERVICE_TOKEN", "")
# Se suma directo a la latencia de los turnos que consultan Base44.
BASE44_TIMEOUT_SECONDS = float(os.environ.get("BASE44_TIMEOUT_SECONDS", "2.0"))
KNOWLEDGE_MIN_WORDS = int(os.environ.get("KNOWLEDGE_MIN_WORDS", "3"))


# ---------------------------------------------------------------------------
# Voz de Héctor
# ---------------------------------------------------------------------------

def construir_tts() -> elevenlabs.TTS:
    return elevenlabs.TTS(voice_id=ELEVENLABS_VOICE_ID, model=ELEVENLABS_MODEL)


# ---------------------------------------------------------------------------
# Conocimiento de Base44
# ---------------------------------------------------------------------------

async def obtener_conocimiento_base44(
    http: httpx.AsyncClient, pregunta: str, min_words: int = KNOWLEDGE_MIN_WORDS
) -> str:
    """
    Pide a Base44 (getKnowledgeForQuery) el conocimiento relevante.
    Nunca lanza excepciones: si falla o tarda, regresa "" y el gemelo
    responde sin ese contexto.
    """
    if not BASE44_APP_URL or len(pregunta.split()) < min_words:
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
# Avatar de Anam
# ---------------------------------------------------------------------------

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

    Debe llamarse ANTES de session.start(): así el audio va a Anam y no
    directo al navegador.
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


# ---------------------------------------------------------------------------
# Latencias
# ---------------------------------------------------------------------------

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
            "LATENCIA | total=%s | fin de turno=%s | herramientas/Base44=%s | LLM 1er token=%s | "
            "TTS 1er audio=%s | reproducción=%s",
            _ms(metricas.get("e2e_latency")),
            _ms(metricas.get("end_of_turn_delay")),
            _ms(metricas.get("on_user_turn_completed_delay")),
            _ms(metricas.get("llm_node_ttft")),
            _ms(metricas.get("tts_node_ttfb")),
            _ms(metricas.get("playback_latency")),
        )
