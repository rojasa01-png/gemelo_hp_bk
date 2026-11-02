"""
Mini servidor que le da un token de LiveKit al navegador.

Cada visita crea una sala nueva; el agente (agent.py) se asigna solo a esa
sala. Las llaves de LiveKit viven aquí, nunca en el navegador.

    uvicorn token_server:app --port 8001

Antes de abrirlo al público: limita cuántos tokens se pueden pedir (rate
limit), porque cada sala con agente cuesta dinero.
"""

import os
import uuid
from datetime import timedelta

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from livekit import api

load_dotenv(".env.local")
load_dotenv()

LIVEKIT_URL = os.environ["LIVEKIT_URL"]
LIVEKIT_API_KEY = os.environ["LIVEKIT_API_KEY"]
LIVEKIT_API_SECRET = os.environ["LIVEKIT_API_SECRET"]

app = FastAPI(title="Token LiveKit - Gemelo HP")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[os.environ.get("FRONTEND_ORIGIN", "*")],
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["*"],
)


@app.post("/livekit-token")
async def livekit_token():
    sala = f"gemelo-{uuid.uuid4().hex[:8]}"
    identidad = f"visitante-{uuid.uuid4().hex[:6]}"
    token = (
        api.AccessToken(LIVEKIT_API_KEY, LIVEKIT_API_SECRET)
        .with_identity(identidad)
        .with_name("Visitante")
        .with_grants(api.VideoGrants(room_join=True, room=sala))
        .with_ttl(timedelta(hours=1))
        .to_jwt()
    )
    return {"url": LIVEKIT_URL, "token": token, "room": sala}


@app.get("/health")
async def health():
    return {"status": "ok"}
