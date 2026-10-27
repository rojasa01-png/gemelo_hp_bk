"""
Orquestador: Base44 -> Claude -> Anam
=====================================

Este servidor es el "cerebro intermedio" del gemelo digital en vivo.

Dos formas de conectarlo con Anam (las dos funcionan al mismo tiempo):

  A) Cliente (la actual, llmId "CUSTOMER_CLIENT_V1"):
     el navegador escucha MESSAGE_HISTORY_UPDATED, llama a POST /chat-stream
     y le pasa el texto a Anam con talk(). Hay un viaje extra por el navegador.

  B) Servidor (nueva, recomendada para bajar latencia):
     Anam llama directo a POST /v1/chat/completions (formato OpenAI, SSE).
     Se activa poniendo ANAM_LLM_ID en el entorno (ver README al final).

Flujo en ambos casos:
  1. Se recibe el historial de la conversación.
  2. Se consulta a Base44 qué conocimiento es relevante (con timeout corto).
  3. Se arma el prompt final (system prompt + conocimiento + historial).
  4. Se llama a Claude en streaming.
  5. La respuesta regresa token por token para que Anam la convierta
     en voz + animación facial.

Requisitos (requirements.txt):
    fastapi
    uvicorn[standard]
    httpx
    anthropic
    python-dotenv

Correr localmente:
    uvicorn orquestador_gemelo_hp:app --reload --port 8000

Variables de entorno (archivo .env, NUNCA subir a git):
    ANTHROPIC_API_KEY=sk-ant-...
    BASE44_APP_URL=https://tu-app.base44.app
    BASE44_SERVICE_TOKEN=...        # si Base44 lo requiere
    ANAM_API_KEY=...
    ANAM_AVATAR_ID=...
    ANAM_VOICE_ID=...

  Opcionales:
    ANAM_AVATAR_MODEL=cara-4.1-express   # "" para no mandarlo
    ANAM_LLM_ID=...                      # activa el modo servidor (B)
    ANAM_LLM_SECRET=...                  # secreto que Anam manda como Bearer
    CLAUDE_MODEL=claude-sonnet-5
    CLAUDE_MAX_TOKENS=300
    BASE44_TIMEOUT_SECONDS=2.0
    KNOWLEDGE_MIN_WORDS=3
    END_OF_SPEECH_SENSITIVITY=0.3
    SILENCE_BEFORE_AUTO_END_TURN_SECONDS=1.5
    FRONTEND_ORIGIN=https://tu-frontend.vercel.app
"""

import os
import json
import time
import logging
from contextlib import asynccontextmanager
from typing import List, Dict, Any, AsyncIterator

import httpx
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from dotenv import load_dotenv
import anthropic

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------

load_dotenv()

ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]
BASE44_APP_URL = os.environ["BASE44_APP_URL"].rstrip("/")
BASE44_SERVICE_TOKEN = os.environ.get("BASE44_SERVICE_TOKEN", "")
ANAM_API_KEY = os.environ["ANAM_API_KEY"]
ANAM_AVATAR_ID = os.environ["ANAM_AVATAR_ID"]  # ID del avatar de Héctor
ANAM_VOICE_ID = os.environ["ANAM_VOICE_ID"]    # ID de la voz clonada de Héctor

# Modelo de avatar de Anam. cara-4.1-express (20-oct-2026): animación más
# expresiva y mejor lip sync. Si tu avatar actual no lo soporta, el session
# token se reintenta automáticamente sin este campo (ver /session-token).
ANAM_AVATAR_MODEL = os.environ.get("ANAM_AVATAR_MODEL", "cara-4.1-express")

# Si defines ANAM_LLM_ID (el id que regresa POST https://api.anam.ai/v1/llms),
# Anam llama directo a /v1/chat/completions y se elimina el viaje por el navegador.
ANAM_LLM_ID = os.environ.get("ANAM_LLM_ID", "")
ANAM_LLM_SECRET = os.environ.get("ANAM_LLM_SECRET", "")

CLAUDE_MODEL = os.environ.get("CLAUDE_MODEL", "claude-sonnet-5")
# Respuestas habladas: más cortas = el avatar termina antes y se siente más ágil.
CLAUDE_MAX_TOKENS = int(os.environ.get("CLAUDE_MAX_TOKENS", "300"))

# Base44 es lo único que corre ANTES del primer token de Claude, así que su
# timeout se suma directo a la latencia. Antes: 8 s. Ahora 2 s por defecto.
BASE44_TIMEOUT_SECONDS = float(os.environ.get("BASE44_TIMEOUT_SECONDS", "2.0"))
# Mensajes muy cortos ("sí", "ok", "gracias") no necesitan buscar conocimiento.
KNOWLEDGE_MIN_WORDS = int(os.environ.get("KNOWLEDGE_MIN_WORDS", "3"))

END_OF_SPEECH_SENSITIVITY = float(os.environ.get("END_OF_SPEECH_SENSITIVITY", "0.3"))
SILENCE_BEFORE_AUTO_END_TURN_SECONDS = float(
    os.environ.get("SILENCE_BEFORE_AUTO_END_TURN_SECONDS", "1.5")
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("orquestador_gemelo_hp")

# Clientes asíncronos y compartidos: no bloquean el event loop de FastAPI y
# reutilizan conexiones (se evita el handshake TLS en cada turno).
claude_client = anthropic.AsyncAnthropic(api_key=ANTHROPIC_API_KEY)
http_client = httpx.AsyncClient(timeout=10.0)


@asynccontextmanager
async def lifespan(_: FastAPI):
    yield
    await http_client.aclose()


app = FastAPI(title="Orquestador Gemelo Digital HP", lifespan=lifespan)

# CORS: el frontend (la página HTML/JS con el SDK de Anam) va a vivir en
# otro dominio (ej. Vercel/Netlify), así que el navegador necesita permiso
# explícito para llamar a este backend. Reemplaza con el dominio real
# una vez que esté desplegado; usar "*" solo mientras pruebas localmente.
FRONTEND_ORIGIN = os.environ.get("FRONTEND_ORIGIN", "*")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_ORIGIN],
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# System prompt (la versión que ya probamos en Anam) + reglas de voz.
# En producción, esto podría vivir en un archivo aparte (system_prompt.txt)
# en vez de estar embebido aquí.
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_BASE = """
Eres el gemelo digital de Héctor Padilla Maya — estratega de negocios,
Board Advisor y fundador de Grupo Scanda / MAIA y Tadah Digital Marketing.
Piensas, hablas y decides como Héctor: directo, estratégico, retador,
analítico, con criterio propio.

Resultados primero. Tecnología después.

Nunca opinas sobre: política partidista, controversias de redes sin
sustancia, competidores nombrados directamente, información financiera
confidencial de clientes.

Responde de forma directa, sin preámbulos, en español de México
(o en inglés si te hablan en inglés). Nunca digas "excelente pregunta"
ni frases de coach motivacional.

Tienes acceso a fragmentos de conocimiento recuperados de la base de
Héctor (ver más abajo, si los hay). Si el conocimiento no cubre la
pregunta, dilo honestamente en vez de inventar.
""".strip()

# Todo lo que escribe Claude se convierte en voz. Markdown, listas o emojis
# se leerían raro o se leerían literalmente, y las respuestas largas se
# sienten lentas aunque el primer token llegue rápido.
REGLAS_DE_VOZ = """
Formato de voz (obligatorio): tu respuesta se va a decir en voz alta.
Habla en 1 a 4 oraciones cortas, como en una conversación real. Sin
markdown, sin listas, sin viñetas, sin emojis, sin encabezados. Si el
tema es largo, da la idea principal y pregunta si quieren que profundices.
""".strip()

SYSTEM_PROMPT = f"{SYSTEM_PROMPT_BASE}\n\n{REGLAS_DE_VOZ}"


# ---------------------------------------------------------------------------
# Paso 1: consultar Base44 por conocimiento relevante
# ---------------------------------------------------------------------------

async def obtener_conocimiento_base44(pregunta: str) -> str:
    """
    Llama a la función de backend de Base44 (getKnowledgeForQuery)
    y regresa el fragmento de conocimiento relevante como texto plano.

    Si Base44 no responde a tiempo o falla, regresa cadena vacía en vez de
    tronar todo el flujo (el gemelo puede seguir respondiendo sin
    ese contexto adicional).
    """
    if len(pregunta.split()) < KNOWLEDGE_MIN_WORDS:
        return ""

    url = f"{BASE44_APP_URL}/api/functions/getKnowledgeForQuery"
    headers = {"Content-Type": "application/json"}
    if BASE44_SERVICE_TOKEN:
        headers["Authorization"] = f"Bearer {BASE44_SERVICE_TOKEN}"

    inicio = time.perf_counter()
    try:
        response = await http_client.post(
            url,
            headers=headers,
            json={"query": pregunta},
            timeout=BASE44_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        data = response.json()
        logger.info("Base44 respondió en %.0f ms", (time.perf_counter() - inicio) * 1000)
        return data.get("knowledge", "")
    except (httpx.HTTPError, ValueError) as exc:
        logger.warning(
            "Base44 no respondió a tiempo o falló (%.0f ms): %s",
            (time.perf_counter() - inicio) * 1000,
            exc,
        )
        return ""


# ---------------------------------------------------------------------------
# Paso 2: armar el prompt final para Claude
# ---------------------------------------------------------------------------

def _texto_de_contenido(content: Any) -> str:
    """El formato OpenAI permite content como string o como lista de partes."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            parte.get("text", "") for parte in content if isinstance(parte, dict)
        )
    return ""


def construir_mensajes_claude(
    historial: List[Dict[str, Any]], conocimiento: str
) -> List[Dict[str, str]]:
    """
    Convierte el historial (lista de {role, content}) al formato que espera
    la API de Claude, e inyecta el conocimiento recuperado de Base44 como
    contexto adicional antes de la última pregunta.
    """
    mensajes: List[Dict[str, str]] = []

    for turno in historial:
        rol_original = turno.get("role")
        if rol_original == "system":
            continue  # el system prompt va aparte, en el parámetro `system`
        rol = "user" if rol_original == "user" else "assistant"
        texto = _texto_de_contenido(turno.get("content", ""))
        if not texto:
            continue
        # Claude exige roles alternados; si llegan dos seguidos, se unen.
        if mensajes and mensajes[-1]["role"] == rol:
            mensajes[-1]["content"] += f"\n{texto}"
        else:
            mensajes.append({"role": rol, "content": texto})

    # Claude exige que la conversación termine en un mensaje "user".
    # A veces el historial llega terminando en "assistant" (ej. justo después
    # de que el avatar termina de hablar) — recortamos esos mensajes finales
    # para evitar el error 400 "must end with a user message".
    while mensajes and mensajes[-1]["role"] != "user":
        mensajes.pop()

    if conocimiento and mensajes:
        mensajes[-1]["content"] = (
            f"[Conocimiento relevante de Base44]\n{conocimiento}\n\n"
            f"[Pregunta del usuario]\n{mensajes[-1]['content']}"
        )

    # Claude también exige que el primer mensaje sea "user".
    while mensajes and mensajes[0]["role"] != "user":
        mensajes.pop(0)

    return mensajes


# ---------------------------------------------------------------------------
# Paso 3: llamar a Claude en streaming
# ---------------------------------------------------------------------------

async def generar_respuesta_stream(mensajes: List[Dict[str, str]]) -> AsyncIterator[str]:
    """
    Generador asíncrono que hace streaming de la respuesta de Claude, token
    por token, para que Anam pueda empezar a generar voz/animación antes de
    que termine la respuesta completa (baja la latencia percibida).
    """
    inicio = time.perf_counter()
    primer_token = True
    try:
        async with claude_client.messages.stream(
            model=CLAUDE_MODEL,
            max_tokens=CLAUDE_MAX_TOKENS,
            system=SYSTEM_PROMPT,
            messages=mensajes,
        ) as stream:
            async for texto in stream.text_stream:
                if primer_token:
                    logger.info(
                        "Claude primer token en %.0f ms",
                        (time.perf_counter() - inicio) * 1000,
                    )
                    primer_token = False
                yield texto
    except anthropic.APIError as exc:
        logger.error("Error de la API de Claude: %s", exc)
        if primer_token:
            yield "Perdón, tuve un problema para responder. ¿Me lo repites?"


# ---------------------------------------------------------------------------
# Modo A (actual): el navegador llama a /chat-stream
# ---------------------------------------------------------------------------

@app.post("/session-token")
async def session_token():
    """
    El frontend llama a este endpoint primero, antes de iniciar cualquier
    sesión con Anam. Esto evita exponer la API key de Anam en el navegador:
    la key vive únicamente aquí, en el servidor.
    """
    persona_config: Dict[str, Any] = {
        "name": "Gemelo Digital de Héctor Padilla",
        "avatarId": ANAM_AVATAR_ID,
        "voiceId": ANAM_VOICE_ID,
        # CUSTOMER_CLIENT_V1 = "yo controlo las respuestas desde el cliente".
        # Con ANAM_LLM_ID, Anam llama directo a /v1/chat/completions.
        "llmId": ANAM_LLM_ID or "CUSTOMER_CLIENT_V1",
        "systemPrompt": SYSTEM_PROMPT,
        "languageCode": "es",  # evita que el STT transcriba/traduzca a inglés
        "voiceDetectionOptions": {
            # Ver notas al final: confirmar en la documentación de Anam en qué
            # dirección afecta este valor antes de moverlo.
            "endOfSpeechSensitivity": END_OF_SPEECH_SENSITIVITY,
            # Cuánto tolera una pausa natural a media oración antes de asumir
            # que ya terminaste de hablar.
            "silenceBeforeAutoEndTurnSeconds": SILENCE_BEFORE_AUTO_END_TURN_SECONDS,
        },
    }
    if ANAM_AVATAR_MODEL:
        persona_config["avatarModel"] = ANAM_AVATAR_MODEL

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {ANAM_API_KEY}",
    }
    url = "https://api.anam.ai/v1/auth/session-token"

    try:
        response = await http_client.post(
            url, headers=headers, json={"personaConfig": persona_config}
        )

        # Si el avatar actual no soporta el modelo pedido, no rompemos la
        # sesión: reintentamos una vez con el modelo que Anam elija.
        if response.status_code in (400, 422) and "avatarModel" in persona_config:
            logger.warning(
                "Anam rechazó avatarModel=%s (%s): %s. Reintentando sin avatarModel.",
                ANAM_AVATAR_MODEL,
                response.status_code,
                response.text[:300],
            )
            persona_config.pop("avatarModel")
            response = await http_client.post(
                url, headers=headers, json={"personaConfig": persona_config}
            )

        response.raise_for_status()
        return response.json()  # { "sessionToken": "..." }
    except httpx.HTTPError as exc:
        logger.error("No se pudo generar el session token de Anam: %s", exc)
        return {"error": "No se pudo iniciar la sesión con Anam"}


@app.post("/chat-stream")
async def chat_stream(request: Request):
    """
    El navegador manda algo como:
    {
        "messages": [
            {"role": "user", "content": "Hola, ¿qué hace Tadah?"},
            ...
        ]
    }
    """
    body: Dict[str, Any] = await request.json()
    historial = body.get("messages", [])

    if not historial:
        return {"error": "No se recibió historial de conversación"}

    ultima_pregunta = _texto_de_contenido(historial[-1].get("content", ""))

    conocimiento = await obtener_conocimiento_base44(ultima_pregunta)
    mensajes = construir_mensajes_claude(historial, conocimiento)

    if not mensajes:
        logger.info("No hay mensaje de usuario válido para responder; se ignora.")
        return StreamingResponse(iter([]), media_type="text/plain")

    return StreamingResponse(
        generar_respuesta_stream(mensajes),
        media_type="text/plain",
    )


# ---------------------------------------------------------------------------
# Modo B (nuevo): Anam llama directo, formato OpenAI chat.completions + SSE
# ---------------------------------------------------------------------------

def _chunk_openai(completion_id: str, created: int, delta: Dict[str, Any], finish: Any) -> str:
    chunk = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": CLAUDE_MODEL,
        "choices": [
            {"index": 0, "delta": delta, "logprobs": None, "finish_reason": finish}
        ],
    }
    return f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"


async def _stream_openai_sse(mensajes: List[Dict[str, str]]) -> AsyncIterator[str]:
    completion_id = f"chatcmpl-gemelo-{int(time.time() * 1000)}"
    created = int(time.time())

    yield _chunk_openai(completion_id, created, {"role": "assistant"}, None)
    async for texto in generar_respuesta_stream(mensajes):
        yield _chunk_openai(completion_id, created, {"content": texto}, None)
    yield _chunk_openai(completion_id, created, {}, "stop")
    yield "data: [DONE]\n\n"


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """
    Endpoint compatible con OpenAI que Anam llama desde sus servidores cuando
    la persona usa un LLM registrado (POST https://api.anam.ai/v1/llms con
    llmFormat "openai"). Anam exige streaming SSE.
    """
    if ANAM_LLM_SECRET:
        if request.headers.get("authorization", "") != f"Bearer {ANAM_LLM_SECRET}":
            raise HTTPException(status_code=401, detail="No autorizado")

    body: Dict[str, Any] = await request.json()
    historial = body.get("messages", [])

    ultima_pregunta = ""
    for turno in reversed(historial):
        if turno.get("role") == "user":
            ultima_pregunta = _texto_de_contenido(turno.get("content", ""))
            break

    conocimiento = await obtener_conocimiento_base44(ultima_pregunta)
    mensajes = construir_mensajes_claude(historial, conocimiento)

    if not mensajes:
        # Sondeos de Anam (prompts pequeños de prueba) o historial sin turno
        # de usuario: igual hay que responder con un stream válido.
        logger.info("Sin mensaje de usuario válido; se responde con stream vacío.")
        mensajes = [{"role": "user", "content": "Hola"}]

    return StreamingResponse(
        _stream_openai_sse(mensajes),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/health")
async def health():
    """Endpoint simple para confirmar que el servidor está vivo."""
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Cómo activar el modo B (servidor):
#
# 1. Despliega este archivo y confirma que https://TU-APP/health responde.
# 2. Registra el LLM en Anam (una sola vez):
#
#    curl -X POST "https://api.anam.ai/v1/llms" \
#      -H "Authorization: Bearer $ANAM_API_KEY" \
#      -H "Content-Type: application/json" \
#      -d '{
#        "displayName": "Gemelo HP (Claude + Base44)",
#        "urls": [{"url": "https://TU-APP/v1/chat/completions"}],
#        "llmFormat": "openai",
#        "modelName": "claude-sonnet-5",
#        "secret": "EL_MISMO_VALOR_DE_ANAM_LLM_SECRET"
#      }'
#
# 3. Copia el `id` de la respuesta a ANAM_LLM_ID y redespliega.
# 4. En el frontend ya no hace falta escuchar MESSAGE_HISTORY_UPDATED ni
#    llamar a /chat-stream: Anam habla solo. Mantén /chat-stream mientras
#    migras para poder volver atrás quitando ANAM_LLM_ID.
# 5. Corre la prueba de validación del LLM en Anam Lab antes de producción.
#
# Notas pendientes:
# - La función getKnowledgeForQuery debe existir en Base44 y definir qué
#   regresa exactamente (hoy se espera {"knowledge": "texto"}).
# - endOfSpeechSensitivity: el comentario original decía "0 = espera hasta
#   estar seguro, 1 = responde muy rápido", pero la guía de rendimiento de
#   Anam dice que un valor alto puede retrasar la decisión de fin de turno.
#   Confirma la dirección real en la documentación antes de cambiarlo.
# ---------------------------------------------------------------------------
