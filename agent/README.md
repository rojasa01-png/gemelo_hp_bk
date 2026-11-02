# Gemelo HP con LiveKit (Claude + voz de Héctor + Anam)

Alternativa al orquestador actual. **No reemplaza nada**: el `orquestador_gemelo_hp.py` de la raíz sigue igual.

```
Usuario ──LiveKit──> Deepgram ──> Base44 ──> Claude ──> ElevenLabs (voz de Héctor)
                                                              │
Usuario <──LiveKit── video + audio <── Anam (cara + lip sync) <┘
```

- **Cerebro:** Claude, con el mismo system prompt y el conocimiento de Base44.
- **Voz:** la clonada de Héctor, generada por ElevenLabs. Anam solo pone la cara.
- **Transporte:** LiveKit lleva el audio al avatar y el video al navegador. Se elimina el viaje por el navegador del modo actual.

## Antes de empezar: la voz de Héctor

En esta integración **Anam no usa tu `ANAM_VOICE_ID`**. La voz sale del TTS del agente, así que necesitas el ID de la voz clonada **en ElevenLabs** (`ELEVENLABS_VOICE_ID`).

- Si la voz ya está en tu cuenta de ElevenLabs, copia su ID desde *Voices*.
- Si la clonaste dentro de Anam Lab, probablemente tengas que clonarla de nuevo en ElevenLabs con las mismas muestras de audio. No pude verificar desde dónde se creó.

## Probar localmente

```bash
cd agent
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env.local        # y llena las llaves
python agent.py dev               # terminal 1: el agente
uvicorn token_server:app --port 8001   # terminal 2: tokens
# terminal 3: sirve la página de prueba
python -m http.server 8080 --directory web
```

Abre `http://localhost:8080`, pulsa **Hablar con el gemelo** y permite el micrófono.

## Cómo medir si mejora

Cada respuesta imprime una línea así en la terminal del agente:

```
LATENCIA | total=... | fin de turno=... | Base44+hook=... | Claude 1er token=... | TTS 1er audio=... | reproducción=...
```

Haz unas 10 preguntas iguales aquí y en el modo actual, y compara el tiempo desde que dejas de hablar hasta que el avatar empieza a hablar. `total` es el número principal. El resto dice dónde se va el tiempo.

## Límites y cosas que debes saber

- **No lo he podido correr contra servicios reales.** Verifiqué que el código compila, que los parámetros existen en `livekit-agents 1.8` y probé la lógica con simulaciones. Falta tu prueba con llaves reales.
- **Anam espera 480 ms de audio** antes de generar los primeros cuadros en la ruta de LiveKit, y no se puede cambiar. Por eso esta ruta no está garantizada como más rápida.
- **Base44 retrasa cada turno** hasta su timeout (2 s por defecto), porque Claude no arranca hasta tenerlo. Mira `Base44+hook` en el log.
- **Fidelidad de la voz:** `eleven_turbo_v2_5` es el modelo por defecto del plugin. `eleven_flash_v2_5` es más rápido pero puede sonar menos fiel al clon.
- **Despliegue:** el agente es un proceso de larga duración. No va en Vercel; usa LiveKit Cloud (`lk agent deploy`) o cualquier host de contenedores. La página de `web/` sí puede ir en Vercel, y `token_server.py` va en el mismo host que el orquestador actual.
- **Antes de abrirlo al público:** limita la creación de tokens (cada sala con agente cuesta) y pon `FRONTEND_ORIGIN` con tu dominio real.
- `system_prompt.py` es una copia del prompt del orquestador. Si cambias uno, cambia el otro.
- Si el avatar no soporta `cara-4.1-express`, el agente reintenta solo con el modelo por defecto del avatar y lo deja en el log.
