"""
Orquestador: Base44 -> Claude -> Anam
=====================================

Este servidor es el "cerebro intermedio" del gemelo digital en vivo.

Flujo:
  1. Anam manda el historial de la conversación a este servidor
     (modo Custom LLM, llmId: "CUSTOMER_CLIENT_V1").
  2. Este servidor le pregunta a Base44 qué conocimiento es relevante
     para la última pregunta del usuario.
  3. Arma el prompt final (system prompt de gemelo_hp + conocimiento + historial).
  4. Llama a Claude con ese prompt.
  5. Regresa la respuesta a Anam en streaming, para que la convierta
     en voz + animación facial.

Requisitos (requirements.txt):
    fastapi
    uvicorn
    requests
    anthropic
    python-dotenv

Correr localmente:
    uvicorn orquestador_gemelo_hp:app --reload --port 8000

Variables de entorno necesarias (archivo .env, NUNCA subir a git):
    ANTHROPIC_API_KEY=sk-ant-...
    BASE44_APP_URL=https://tu-app.base44.app
    BASE44_SERVICE_TOKEN=...      # si Base44 requiere un token para asServiceRole
"""

import os
import json
import logging
from typing import List, Dict, Any

import requests
from fastapi import FastAPI, Request
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

CLAUDE_MODEL = "claude-sonnet-5"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("orquestador_gemelo_hp")

app = FastAPI(title="Orquestador Gemelo Digital HP")
claude_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

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
# System prompt recortado (la versión que ya probamos en Anam).
# En producción, esto podría vivir en un archivo aparte (system_prompt.txt)
# en vez de estar embebido aquí.
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """
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


# ---------------------------------------------------------------------------
# Paso 1: consultar Base44 por conocimiento relevante
# ---------------------------------------------------------------------------

def obtener_conocimiento_base44(pregunta: str) -> str:
    """
    Llama a la función de backend de Base44 (getKnowledgeForQuery)
    y regresa el fragmento de conocimiento relevante como texto plano.

    Si Base44 no responde o falla, regresa cadena vacía en vez de
    tronar todo el flujo (el gemelo puede seguir respondiendo sin
    ese contexto adicional).
    """
    url = f"{BASE44_APP_URL}/api/functions/getKnowledgeForQuery"
    headers = {"Content-Type": "application/json"}
    if BASE44_SERVICE_TOKEN:
        headers["Authorization"] = f"Bearer {BASE44_SERVICE_TOKEN}"

    try:
        response = requests.post(
            url,
            headers=headers,
            json={"query": pregunta},
            timeout=8,  # Base44 a veces tarda más de 5s; 8s da más margen sin arruinar la latencia
        )
        response.raise_for_status()
        data = response.json()
        return data.get("knowledge", "")
    except requests.RequestException as exc:
        logger.warning("Base44 no respondió a tiempo o falló: %s", exc)
        return ""


# ---------------------------------------------------------------------------
# Paso 2: armar el prompt final para Claude
# ---------------------------------------------------------------------------

def construir_mensajes_claude(
    historial: List[Dict[str, str]], conocimiento: str
) -> List[Dict[str, str]]:
    """
    Convierte el historial de Anam (lista de {role, content}) al formato
    que espera la API de Claude, e inyecta el conocimiento recuperado
    de Base44 como contexto adicional antes de la última pregunta.
    """
    mensajes = []

    for turno in historial:
        rol = "user" if turno.get("role") == "user" else "assistant"
        mensajes.append({"role": rol, "content": turno.get("content", "")})

    if conocimiento and mensajes:
        ultimo = mensajes[-1]
        if ultimo["role"] == "user":
            ultimo["content"] = (
                f"[Conocimiento relevante de Base44]\n{conocimiento}\n\n"
                f"[Pregunta del usuario]\n{ultimo['content']}"
            )

    # Claude exige que la conversación termine en un mensaje "user".
    # A veces Anam dispara MESSAGE_HISTORY_UPDATED con el historial
    # terminando en "assistant" (ej. justo después de que el avatar
    # termina de hablar) — recortamos esos mensajes finales para
    # evitar el error 400 "must end with a user message".
    while mensajes and mensajes[-1]["role"] != "user":
        mensajes.pop()

    return mensajes


# ---------------------------------------------------------------------------
# Paso 3: llamar a Claude en streaming
# ---------------------------------------------------------------------------

def generar_respuesta_stream(mensajes: List[Dict[str, str]]):
    """
    Generador que hace streaming de la respuesta de Claude, token por token,
    para que Anam pueda empezar a generar voz/animación antes de que
    termine la respuesta completa (baja la latencia percibida).
    """
    with claude_client.messages.stream(
        model=CLAUDE_MODEL,
        max_tokens=500,
        system=SYSTEM_PROMPT,
        messages=mensajes,
    ) as stream:
        for texto in stream.text_stream:
            yield texto


# ---------------------------------------------------------------------------
# Endpoint que Anam llama (modo Custom LLM, llmId: "CUSTOMER_CLIENT_V1")
# ---------------------------------------------------------------------------

@app.post("/session-token")
async def session_token():
    """
    El frontend llama a este endpoint primero, antes de iniciar cualquier
    sesión con Anam. Esto evita exponer la API key de Anam en el navegador:
    la key vive únicamente aquí, en el servidor.
    """
    try:
        response = requests.post(
            "https://api.anam.ai/v1/auth/session-token",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {ANAM_API_KEY}",
            },
            json={
                "personaConfig": {
                    "name": "Gemelo Digital de Héctor Padilla",
                    "avatarId": ANAM_AVATAR_ID,
                    "voiceId": ANAM_VOICE_ID,
                    "llmId": "CUSTOMER_CLIENT_V1",  # fuerza el modo "yo controlo las respuestas"
                    "systemPrompt": SYSTEM_PROMPT,
                    "languageCode": "es",  # evita que el STT transcriba/traduzca a inglés
                    "voiceDetectionOptions": {
                        # 0 = espera hasta estar seguro de que terminaste; 1 = responde muy rápido.
                        # Lo bajamos para que no corte a media frase.
                        "endOfSpeechSensitivity": 0.3,
                        # Cuánto tolera una pausa natural a media oración antes de asumir
                        # que ya terminaste de hablar.
                        "silenceBeforeAutoEndTurnSeconds": 1.5,
                    },
                }
            },
            timeout=10,
        )
        response.raise_for_status()
        return response.json()  # { "sessionToken": "..." }
    except requests.RequestException as exc:
        logger.error("No se pudo generar el session token de Anam: %s", exc)
        return {"error": "No se pudo iniciar la sesión con Anam"}


@app.post("/chat-stream")
async def chat_stream(request: Request):
    """
    Anam manda algo como:
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

    ultima_pregunta = historial[-1].get("content", "")

    # Paso 1: conocimiento de Base44
    conocimiento = obtener_conocimiento_base44(ultima_pregunta)

    # Paso 2: armar mensajes para Claude
    mensajes = construir_mensajes_claude(historial, conocimiento)

    if not mensajes:
        logger.info("No hay mensaje de usuario válido para responder; se ignora.")
        return StreamingResponse(iter([]), media_type="text/plain")

    # Paso 3: responder en streaming
    return StreamingResponse(
        generar_respuesta_stream(mensajes),
        media_type="text/plain",
    )


@app.get("/health")
async def health():
    """Endpoint simple para confirmar que el servidor está vivo."""
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Notas para quien vaya a completar esto:
#
# 1. El nombre y el formato exacto del endpoint que espera Anam
#    ("/chat-stream", el shape del JSON de entrada/salida) hay que
#    confirmarlo contra la documentación vigente de Anam antes de
#    conectar esto de verdad — este esqueleto sigue el patrón general
#    de sus ejemplos, pero Anam podría pedir un formato de streaming
#    específico (ej. Server-Sent Events con un formato particular).
#
# 2. La función getKnowledgeForQuery todavía no existe en Base44 —
#    hay que crearla ahí primero, y definir qué regresa exactamente
#    (texto plano, o un JSON con varios fragmentos).
#
# 3. Falta manejar autenticación real hacia Base44 (asServiceRole)
#    según lo que pida su documentación de backend functions.
# ---------------------------------------------------------------------------