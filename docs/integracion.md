# Integración

Hay tres formas de usar MRCD desde otro proyecto, de la más simple a la más completa.

## 1. Python, en el mismo proceso

Para analizar audio dentro de tu propio código. Carga el modelo **una vez** y reutilízalo; `analyze` no es seguro entre hilos (usa una instancia por proceso).

```python
import soundfile as sf
from core.engine import MRCDEngine

engine = MRCDEngine.load("models/ckpt_audio_text.pt")  # caro: hazlo al arrancar

audio, sr = sf.read("grabacion.wav", dtype="float32")   # mono, 16 kHz
assert sr == 16000, "convierte antes: ffmpeg -i in.m4a -ac 1 -ar 16000 out.wav"

result = engine.analyze(audio)
for ev in result["events"]:
    print(ev["label"], ev["decision"], ev["start_ms"], ev["end_ms"], ev["text"])

print(result["timings"], result["rtf"], result["warning"])
```

`engine.analyze` devuelve:

| Clave                              | Contenido                                                                                                                       |
| ---------------------------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| `events`                           | Lista de eventos: `label`, `decision` (`event` o `uncertain`), `start_ms`, `end_ms`, `score`, `score_kind`, `n_windows`, `text` |
| `words`                            | Palabras del ASR con sus tiempos                                                                                                |
| `duration_ms`                      | Duración analizada                                                                                                              |
| `timings`, `rtf`                   | Tiempo por etapa y relación con la duración                                                                                     |
| `config`, `provenance`, `hardware` | Configuración, procedencia del checkpoint y equipo                                                                              |
| `warning`                          | Aviso de procedencia (por ejemplo, etiquetas heurísticas). Muéstralo a tus usuarios                                             |

Opciones al cargar: `whisper_size`, `device`, `asr_chunk_s`, `thresholds` (por clase), `refine_boundaries` y `temperature`.

## 2. API REST

Para integrar desde cualquier lenguaje con la aplicación levantada. Todas las rutas van bajo `/api` y usan `Authorization: Bearer <token>`.

```bash
# 1. Subir el audio (cuerpo binario)
curl -X POST -H "Authorization: Bearer $TOKEN" \
     --data-binary @grabacion.m4a \
     "http://127.0.0.1:8000/api/recordings?filename=grabacion.m4a"
# → {"id": "rec_…", "status": "queued"}

# 2. Pedir el análisis (responde de inmediato con HTTP 202)
curl -X POST -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
     -d '{"recording_id": "rec_…", "systems": ["mrcd"]}' \
     http://127.0.0.1:8000/api/analysis-runs
# → {"comparison_id": "cmp_…", "runs": [{"id": "run_…", "status": "queued"}]}

# 3. Seguir el progreso (eventos SSE) o consultar el estado
curl -N "http://127.0.0.1:8000/api/analysis-runs/run_…/stream?token=$TOKEN"
curl -H "Authorization: Bearer $TOKEN" http://127.0.0.1:8000/api/analysis-runs/run_…

# 4. Descargar el resultado: json | csv | md | eaf
curl -H "Authorization: Bearer $TOKEN" -o run.csv \
     "http://127.0.0.1:8000/api/exports/run_…?format=csv"
```

Estados de una ejecución: `queued`, `running`, `succeeded`, `partial`, `failed`, `cancelled`. Una ejecución con los mismos datos y configuración no se duplica: la API devuelve la existente.

## 3. Archivos de intercambio

| Formato  | Uso                                                                                                               |
| -------- | ----------------------------------------------------------------------------------------------------------------- |
| JSON     | Resultado completo con versión de exportación `mrcd-export-v1`                                                    |
| CSV      | Un evento por fila, para hojas de cálculo o R/Python                                                              |
| Markdown | Reporte legible de una ejecución                                                                                  |
| EAF      | Abrir o seguir anotando en ELAN; la capa `human_disfluency` se vuelve a leer con `core.dataset.read_human_events` |

## Convenciones

- **Tiempo:** milisegundos enteros desde el inicio del audio, intervalos `[inicio, fin)`.
- **Pausas:** `rhetorical_pause` y `neutral_pause` no son disfluencias. Sepáralas antes de contar.
- **Puntuaciones:** `score_kind: softmax_uncalibrated` indica que no son probabilidades de acierto.
