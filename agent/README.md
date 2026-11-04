# Gemelo HP con LiveKit (voz de Héctor + Anam)

Alternativa al orquestador actual. **No reemplaza nada**: el `orquestador_gemelo_hp.py` de la raíz sigue igual.

Hay dos agentes. Los dos usan la voz clonada de Héctor (ElevenLabs), el conocimiento de Base44 y la cara de Anam. Cambia el cerebro:

| | Agente A: `agent.py` | Agente B: `agent_realtime.py` |
|---|---|---|
| Cerebro | Claude | OpenAI Realtime (modo texto) |
| Oído | Deepgram (voz a texto) | OpenAI entiende el audio directo |
| Voz | ElevenLabs (clon de Héctor) | ElevenLabs (clon de Héctor) |
| Base44 | Se consulta en cada turno | Herramienta que el modelo decide llamar |

```
Usuario ──LiveKit──> [Deepgram + Claude]  o  [OpenAI Realtime] ──> ElevenLabs (voz de Héctor)
                                                                         │
Usuario <──LiveKit── video + audio <── Anam (cara + lip sync) <──────────┘
```

## Llaves que necesitas

| Llave | Agente A (Claude) | Agente B (Realtime) | Dónde se saca |
|---|:-:|:-:|---|
| `LIVEKIT_URL`, `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET` | sí | sí | cloud.livekit.io → tu proyecto → Settings → Keys |
| `ANAM_API_KEY` | sí | sí | Anam Lab → API keys |
| `ANAM_AVATAR_ID` | sí | sí | El mismo avatar que ya usas |
| `ELEVEN_API_KEY` | sí | sí | ElevenLabs → Profile → API keys |
| `ELEVENLABS_VOICE_ID` | sí | sí | ElevenLabs → Voices → la voz de Héctor → copiar ID |
| `OPENAI_API_KEY` | no | **sí** | platform.openai.com → API keys |
| `ANTHROPIC_API_KEY` | **sí** | no | console.anthropic.com |
| `DEEPGRAM_API_KEY` | **sí** | no | console.deepgram.com |
| `BASE44_APP_URL`, `BASE44_SERVICE_TOKEN` | opcional | opcional | Tu app de Base44 |

Para el agente B (el de OpenAI Realtime) son **7 llaves**: 3 de LiveKit, Anam, OpenAI y 2 de ElevenLabs (llave e ID de la voz).

> `ANAM_VOICE_ID` ya no se usa: en la integración con LiveKit, Anam solo pone la cara y la voz sale de ElevenLabs.

## Probar localmente

```bash
cd agent
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env.local        # y llena las llaves
python agent_realtime.py dev      # terminal 1: agente B  (o: python agent.py dev para el A)
uvicorn token_server:app --port 8001   # terminal 2: tokens
python -m http.server 8080 --directory web   # terminal 3: página de prueba
```

Abre `http://localhost:8080`, pulsa **Hablar con el gemelo** y permite el micrófono.
Corre **un agente a la vez** en la misma cuenta de LiveKit: los dos se asignan solos a cada sala nueva.

## Cómo medir si mejora

Cada respuesta imprime una línea así en la terminal del agente:

```
LATENCIA | total=... | fin de turno=... | herramientas/Base44=... | LLM 1er token=... | TTS 1er audio=... | reproducción=...
```

Haz unas 10 preguntas iguales con el agente A, el B y el modo actual, y compara el tiempo desde que dejas de hablar hasta que el avatar empieza a hablar. `total` es el número principal.

## Límites y cosas que debes saber

- **No lo he podido correr contra servicios reales.** Verifiqué que el código compila, que los parámetros existen en `livekit-agents 1.8` y probé la lógica con simulaciones. Falta tu prueba con llaves reales.
- **Agente B:** LiveKit documenta el modo texto + TTS aparte, pero no cubre cómo se comportan las interrupciones ni las herramientas en ese modo. Pruébalo hablándole encima mientras responde.
- **Agente B y Base44:** el modelo decide cuándo llamar a la herramienta. Si ves que contesta sobre Héctor sin consultarla, refuerza `REGLAS_HERRAMIENTA` en `agent_realtime.py`.
- **Anam espera 480 ms de audio** antes de generar los primeros cuadros en la ruta de LiveKit, y no se puede cambiar.
- **Plan gratis de LiveKit (Build):** 1,000 minutos de agente al mes, 5 sesiones a la vez y 1 agente desplegado. El agente puede tardar 10–20 s en unirse tras un rato sin uso.
- **Fidelidad de la voz:** `eleven_turbo_v2_5` es el modelo por defecto del plugin. `eleven_flash_v2_5` es más rápido pero puede sonar menos fiel al clon.
- **Despliegue:** el agente es un proceso de larga duración. No va en Vercel; usa LiveKit Cloud (`lk agent deploy`) o cualquier host de contenedores. La página de `web/` sí puede ir en Vercel.
- **Antes de abrirlo al público:** limita la creación de tokens (cada sala con agente cuesta) y pon `FRONTEND_ORIGIN` con tu dominio real.
- `system_prompt.py` es una copia del prompt del orquestador. Si cambias uno, cambia el otro.
- Si el avatar no soporta `cara-4.1-express`, el agente reintenta solo con el modelo por defecto del avatar y lo deja en el log.
