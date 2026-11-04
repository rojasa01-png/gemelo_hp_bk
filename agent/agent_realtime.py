"""
Agente LiveKit B: OpenAI Realtime + voz de Héctor + Anam
========================================================

    Micrófono del usuario ──LiveKit──> OpenAI Realtime
                                       (escucha el AUDIO directo, razona,
                                        responde SOLO TEXTO)
                                            │   ▲
                                            │   └── herramienta buscar_conocimiento
                                            │       (Base44, solo si hace falta)
                                            ▼
                    ElevenLabs (TTS con la VOZ CLONADA de Héctor)
                                            │
                      LiveKit data stream ──> Anam (cara + lip sync)
                                            │
                 video + audio ──LiveKit──> navegador del usuario

Es la "arquitectura híbrida" que documenta LiveKit: el modelo Realtime en modo
texto (modalities=["text"]) más un TTS aparte. No hay STT separado: Realtime
entiende el audio por sí mismo. La voz de OpenAI no se usa.

Diferencias con agent.py (Claude):
  - El cerebro es OpenAI Realtime, no Claude.
  - Base44 es una HERRAMIENTA que el modelo decide llamar. El hook
    on_user_turn_completed de agent.py no funciona con modelos Realtime.

Correr localmente (desde esta carpeta):
    python agent_realtime.py dev

Variables de entorno: ver .env.example
"""

import os

import httpx
from livekit.agents import Agent, AgentServer, AgentSession, JobContext, cli, function_tool
from livekit.plugins import openai
from openai.types.realtime import RealtimeReasoning
from openai.types.realtime.realtime_audio_input_turn_detection import SemanticVad

from common import (
    BASE44_APP_URL,
    BASE44_TIMEOUT_SECONDS,
    construir_tts,
    iniciar_avatar,
    logger,
    obtener_conocimiento_base44,
    registrar_latencias,
)
from system_prompt import SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# Configuración propia de este agente
# ---------------------------------------------------------------------------

# Si tu cuenta tiene un modelo más nuevo (por ejemplo gpt-realtime-2), ponlo aquí.
OPENAI_REALTIME_MODEL = os.environ.get("OPENAI_REALTIME_MODEL", "gpt-realtime")

# Qué tan rápido decide OpenAI que terminaste de hablar:
# low = espera más (no te corta), high = responde más rápido. Vacío = default.
OPENAI_TURN_EAGERNESS = os.environ.get("OPENAI_TURN_EAGERNESS", "").strip().lower()
_EAGERNESS_VALIDAS = ("auto", "low", "medium", "high")

# Solo para modelos Realtime con razonamiento (ej. gpt-realtime-2):
# "low" o "minimal" responden más rápido. Vacío = default del modelo.
OPENAI_REASONING_EFFORT = os.environ.get("OPENAI_REASONING_EFFORT", "").strip().lower()

REGLAS_HERRAMIENTA = """
Herramienta de conocimiento: antes de responder cualquier pregunta sobre
Héctor, Tadah, Grupo Scanda, MAIA, sus clientes, proyectos, servicios o su
forma de trabajar, llama a buscar_conocimiento con la pregunta del usuario.
No la uses para saludos ni charla general. Si no regresa nada útil, dilo
honestamente en vez de inventar.
""".strip()


# ---------------------------------------------------------------------------
# El agente
# ---------------------------------------------------------------------------

class GemeloHPRealtime(Agent):
    def __init__(self, http: httpx.AsyncClient) -> None:
        instrucciones = SYSTEM_PROMPT
        tools = []

        # La herramienta solo existe si Base44 está configurado; si no, el
        # modelo la llamaría para nada.
        if BASE44_APP_URL:
            instrucciones = f"{SYSTEM_PROMPT}\n\n{REGLAS_HERRAMIENTA}"

            @function_tool
            async def buscar_conocimiento(pregunta: str) -> str:
                """Busca en la base de conocimiento de Héctor información sobre
                Héctor, sus empresas, clientes, proyectos y su forma de pensar.

                Args:
                    pregunta: La pregunta del usuario, con el contexto necesario
                        para que la búsqueda tenga sentido por sí sola.
                """
                # Nunca lanza excepciones: si Base44 falla, regresa el mensaje
                # de abajo y el gemelo sigue conversando.
                conocimiento = await obtener_conocimiento_base44(http, pregunta, min_words=1)
                return conocimiento or "No encontré información relevante en la base de conocimiento."

            tools.append(buscar_conocimiento)

        super().__init__(instructions=instrucciones, tools=tools)


# ---------------------------------------------------------------------------
# Construcción de la sesión
# ---------------------------------------------------------------------------

def construir_sesion() -> AgentSession:
    kwargs = {}

    if OPENAI_TURN_EAGERNESS:
        if OPENAI_TURN_EAGERNESS not in _EAGERNESS_VALIDAS:
            raise RuntimeError(
                f"OPENAI_TURN_EAGERNESS debe ser una de {_EAGERNESS_VALIDAS}, "
                f"no '{OPENAI_TURN_EAGERNESS}'"
            )
        kwargs["turn_detection"] = SemanticVad(
            type="semantic_vad",
            create_response=True,
            eagerness=OPENAI_TURN_EAGERNESS,
            interrupt_response=True,
        )

    if OPENAI_REASONING_EFFORT:
        kwargs["reasoning"] = RealtimeReasoning(effort=OPENAI_REASONING_EFFORT)

    return AgentSession(
        # modalities=["text"]: OpenAI escucha audio pero responde solo texto;
        # la voz la pone el TTS de abajo (el clon de Héctor).
        llm=openai.realtime.RealtimeModel(
            model=OPENAI_REALTIME_MODEL, modalities=["text"], **kwargs
        ),
        tts=construir_tts(),
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
    await session.start(agent=GemeloHPRealtime(http), room=ctx.room)


if __name__ == "__main__":
    cli.run_app(server)
