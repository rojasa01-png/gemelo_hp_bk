"""
Agente LiveKit A: Deepgram + Claude + voz de Héctor + Anam
==========================================================

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

import os

import httpx
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
from livekit.plugins import anthropic, deepgram

from common import (
    _env_bool,
    _env_float,
    construir_tts,
    iniciar_avatar,
    logger,
    obtener_conocimiento_base44,
    registrar_latencias,
    BASE44_TIMEOUT_SECONDS,
)
from system_prompt import SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# Configuración propia de este agente
# ---------------------------------------------------------------------------

CLAUDE_MODEL = os.environ.get("CLAUDE_MODEL", "claude-sonnet-5")
CLAUDE_MAX_TOKENS = int(os.environ.get("CLAUDE_MAX_TOKENS", "300"))
CLAUDE_TEMPERATURE = _env_float("CLAUDE_TEMPERATURE")  # vacío = default de Anthropic

# "multi" entiende español e inglés en la misma sesión. Pon "es" si solo
# hablas español (suele transcribir un poco mejor).
STT_LANGUAGE = os.environ.get("STT_LANGUAGE", "multi")
# Palabras que Deepgram suele fallar, separadas por coma. Opcional.
STT_KEYTERMS = [t.strip() for t in os.environ.get("STT_KEYTERMS", "").split(",") if t.strip()]

# Fin de turno. Vacío = usa los valores por defecto de LiveKit.
MIN_ENDPOINTING_DELAY = _env_float("MIN_ENDPOINTING_DELAY")
MAX_ENDPOINTING_DELAY = _env_float("MAX_ENDPOINTING_DELAY")
# Generación anticipada: LiveKit empieza a generar con la transcripción
# parcial. Si Base44 agrega contexto, esa generación se descarta y se repite,
# así que aquí arranca apagada. Pruébala encendida si Base44 casi no se usa.
PREEMPTIVE_GENERATION = _env_bool("PREEMPTIVE_GENERATION", False)


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
# Construcción de la sesión
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
        tts=construir_tts(),
        vad=inference.VAD(model="silero"),
        turn_handling=turn_handling,
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
