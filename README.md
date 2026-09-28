# MRCD — Dataset, extracción y benchmark de ablación

Motor de Reconocimiento Contextual mediante Fusión Multimodal para la Detección de
Disfluencias del Habla en Español: pipeline experimental de datos, extracción,
anotación y ablación. Existe una demo FastAPI/WebSocket con UI para **micrófono
local y modo `audio_text`**; no hay captura de cámara en vivo ni Docker.

```
config/sources.yaml            consultas de descubrimiento + criterios de inclusión
core/constants.py              taxonomía (7 clases), vector 12-D canónico, umbrales heurísticos
core/extractors/acoustic.py    Silero VAD + pYIN (F0) + RMS a 100 fps
core/extractors/linguistic.py  Faster-Whisper verbatim + rasgos por palabra (+ RoBERTa-BNE opcional)
core/extractors/kinesic.py     MediaPipe Face Landmarker -> vector 12-D + filtro de rostro continuo
core/sync/buffer.py            buffer por timestamp de captura, jitter, offset A/V, interpolación lineal
core/annotation/auto_labeler.py  cascada heurística -> candidatos, ventanas, export ELAN (.eaf)
core/dataset.py                tensores por ventana; lectura de etiquetas humanas desde .eaf
core/contracts.py              contrato versionado de rasgos, taxonomía y evidencia
core/normalization.py          normalización train-only respetando máscaras de ausencia
core/models/fusion.py          nueva implementación cross-attention (Q=acústica, K/V=[texto; cinésica])
core/inference.py              carga validada y predicción común para lote y demo
scripts/data_mining.py         discover / download / extract (yt-dlp + ffmpeg + filtro MediaPipe)
scripts/extract_and_label.py   3 señales + pre-etiquetado + dataset_manifest.json + .eaf
scripts/build_windows.py       ventanas 3 s / 500 ms -> windows_{auto|gold|human}.npz
scripts/llm_judge.py           juez multimodal ciego -> etiquetas gold independientes
scripts/judge_agreement.py     fiabilidad del juez (vs heurísticas, vs otro modelo, vs humano)
scripts/relabel.py             re-etiqueta sin repetir ASR/MediaPipe
scripts/run_ablation.py        audio_only vs audio_text vs trimodal, F1 por clase, IC 95 %
scripts/live_demo.py            API/UI local de micrófono; no captura vídeo
tests/                         pruebas deterministas + corpus sintético end-to-end
```

El módulo de fusión es una **reconstrucción nueva** del API esperado por tests y
scripts: el código original no estaba en este repositorio. No se garantiza que
checkpoints históricos externos sean compatibles. Vuelve a entrenar con
`scripts/train_checkpoints.py`; los nuevos checkpoints guardan normalización,
procedencia y versión de contrato. `train_checkpoints.py` rechaza ventanas sin
`feature_schema`: no puede certificar como actual una extracción antigua aunque
sus dimensiones coincidan. La ablación exploratoria aún puede leer NPZ legados;
eso no los convierte en fuente válida para nuevos checkpoints. El contrato de ventanas `1.2.0` exige
`has_video` para inferencia trimodal: `kmask` puede contener interpolación y no
demuestra que haya 24 fotogramas reales. Ventanas antiguas sin esa bandera deben
reconstruirse desde extracción actual; la inferencia de modos sin vídeo puede leer
NPZ legados si cumplen sus dimensiones. El informe de caso `infer_video.py`
rechaza checkpoints sin contrato o guardados bajo el nombre de otro modo. Los NPZ **crudos** ahora llevan versión, contrato y
configuración del extractor. `build_windows.py`, inferencia y reanudación rechazan
crudos sin procedencia o de otra versión: vuelve a extraer con
`scripts/extract_and_label.py --force` (o `scripts/ingest_single_video.py --force`
para el caso de estudio) antes de construir ventanas nuevas.

La ausencia visual se conserva mediante `kmask`/`has_video`, no se interpreta como
una cara de rasgos cero; las métricas visuales del JSON de inferencia son `null`
cuando la ventana no supera el umbral de cobertura real. Las etiquetas `auto` siguen siendo supervisión débil y
la ablación mantiene el rechazo de train/test con el mismo origen de etiquetas.

## Aplicación de revisión y prueba

`app/` implementa el plan técnico: revisión humana ciega, asistida y adjudicada; snapshots
inmutables; evaluación por eventos (`core/evaluation.py`); playground de grabación/carga con
análisis en un worker separado; comparación de versiones. El motor prepara cada grabación una
sola vez (`core/engine.py`). Puesta en marcha, roles, protocolo y estado:
[`docs/protocolo-mrcd.md`](docs/protocolo-mrcd.md). Guía para anotadores:
[`docs/guia-anotacion-v1.md`](docs/guia-anotacion-v1.md).

```powershell
uvicorn app.api:app --port 8000   # web en http://127.0.0.1:8000
python -m app.worker              # otra terminal
```

## Demo local limitada

`python scripts/live_demo.py --ckpt models/ckpt_audio_text.pt` inicia una UI en
`http://127.0.0.1:8001` y lee el micrófono **del servidor**. Comparte el motor
de inferencia con `scripts/infer_video.py`, pero no implementa vídeo, múltiples
clientes ni un servicio desplegable. `core/sync/buffer.py` prepara ventanas A/V
por timestamp para un futuro adaptador de captura, no está conectado a la demo.

Para el juez general, `--provider auto|anthropic|openai|gemini|groq|ollama`
elige explícitamente el transporte. Los juicios sin fotogramas se etiquetan
como tales y no equivalen a evidencia visual.

## Ejecución completa (Windows, un comando)

```powershell
cd C:\zArnes-tesis\mrcd
powershell -ExecutionPolicy Bypass -File .\setup_and_run.ps1
```

Instala ffmpeg (si falta), crea `.venv`, instala dependencias y corre `scripts/run_all.py`:
tests → modelos → minería (12 semillas + 15–20 por búsqueda) → extracción de 3 señales + pre-anotación
(`--recall-mode`) → ventanas → ablación (prueba de humo) → `results/class_support.md` →
`data/annotation_batch_01/` → tests finales. Todo queda en `RUN_REPORT.md` y `run_log.txt`.
Es reanudable: si se corta, vuelve a lanzar el mismo comando.

Etapas sueltas: `python scripts/data_mining.py {seeds|discover-auto|extract|all|status}`,
`python scripts/extract_and_label.py --recall-mode`, `python scripts/build_windows.py --labels auto`,
`python scripts/run_ablation.py --data data/windows_auto.npz --allow-auto-labels --bootstrap 300`,
`python scripts/report_support.py`, `python scripts/sample_for_annotation.py`.

## Ciclo agéntico: juez multimodal como test independiente

El entrenamiento usa **supervisión débil** (etiquetas heurísticas, "silver"). El test lo etiqueta un
**modelo multimodal a ciegas** (VLM-as-a-Judge), que no ve las reglas ni sus decisiones: recibe frames,
transcripción de contexto y mediciones objetivas. Así el benchmark deja de ser circular.

```powershell
$env:OPENAI_API_KEY="..."
powershell -ExecutionPolicy Bypass -File .\run_agentic.ps1 -Provider openai
```

Sin `-Provider` ni `-Model`, el lanzador elige una clave disponible en este orden:
Anthropic (`claude-sonnet-4-5`), OpenAI (`gpt-4o`), Gemini (`gemini-2.5-flash`).
También acepta claves en `.env`. Para elegir explícitamente:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_agentic.ps1 -Provider gemini
powershell -ExecutionPolicy Bypass -File .\run_agentic.ps1 -Provider groq -Model <modelo-visual-de-Groq>
powershell -ExecutionPolicy Bypass -File .\run_agentic.ps1 -Provider ollama -Model <modelo-visual-local>
```

Gemini requiere `GEMINI_API_KEY`; Groq, `GROQ_API_KEY`; Ollama usa el servidor local
`http://localhost:11434/v1` y no necesita clave. Groq y Ollama requieren `-Model` porque
el lanzador no puede asumir cuál modelo visual está disponible. La caché del juez se separa
por proveedor, modelo y huella de las entradas actuales (rúbrica, contexto acústico/textual,
fotogramas y medio), y por los extremos exactos en milisegundos. Los veredictos
anteriores sin huella o sin identidad de intervalo `ms-v1` no se reutilizan:
vuelve a ejecutar `llm_judge.py` para reconstruir GOLD antes de evaluar. El GOLD
conserva el número de fotogramas usados y la nota de fallback para distinguir juicios
multimodales de juicios sin evidencia visual. `windows_gold.npz` conserva esa
procedencia por ventana; `ablation_results.json` y `ablation_table.md` reportan
la tasa de fallback. Si todos los juicios carecen de fotogramas, el informe los
describe como test de juez **audio/texto**, no como validación multimodal.

`judge_agreement.py` identifica un solo experimento. Si el JSONL compartido contiene
varios proveedores/modelos, pasa ambos argumentos, por ejemplo
`python scripts/judge_agreement.py --provider openai --model gpt-4o` (también al
generar `--make-template 60`). Además coteja cada veredicto con el archivo
`.gold.json` vigente del `data/dataset_manifest.json`: juicios históricos de
otra huella, reintentos fallidos y muestras retiradas no entran en la fiabilidad.
`run_agentic.ps1` pasa la identidad y el manifest automáticamente. Las filas
antiguas sin proveedor, huella o identidad `ms-v1` se omiten; vuelve a ejecutar el juez antes de
calcular fiabilidad.

Etapas: `build_windows --labels auto` (train) → `llm_judge.py` (juicios ciegos, caché reanudable en
`data/judge/judgments.jsonl`) → `build_windows --labels gold` (test) → `judge_agreement.py` (fiabilidad) →
`run_ablation.py --data data/windows_auto.npz --test-data data/windows_gold.npz`.

Detalles que sostienen la validez:

- **Muestreo negativo**: además de los candidatos, el juez recibe ventanas de 3 s SIN candidato. Sin esto,
  el test heredaría la cobertura de las heurísticas y no mediría lo que estas no ven.
- **Estratificación**: hasta `--per-class` ítems por clase, repartidos entre oradores.
  El muestreo reserva ventanas de evaluación únicas (incluidos los negativos) y prioriza
  clases con menos ventanas disponibles; se valida de nuevo antes de cualquier llamada al proveedor.
- **Partición**: se entrena con los hablantes de train (etiquetas silver) y se evalúa con los hablantes de
  test (etiquetas del juez). Ningún orador cruza la frontera.
- **Validación humana (obligatoria para la tesis)**: `judge_agreement.py --make-template 60` genera
  `data/judge/human_check.csv` solo si no existe: las ejecuciones posteriores jamás sobrescriben
  anotaciones humanas. Cada fila queda ligada al intervalo exacto (inicio y fin en milisegundos),
  proveedor, modelo y huella de entrada del GOLD vigente. Los CSV anteriores sin esa identidad o
  con filas obsoletas se rechazan: archívalos, regenera la plantilla y migra cada etiqueta tras
  verificar el video; no se emparejan por tiempos redondeados. Un humano etiqueta la muestra y el
  script calcula el kappa juez–humano. Sin ese número, la tabla de ablación queda sin respaldo.
- **Colisiones de ventana**: si dos juicios GOLD se asignan a la misma ventana de 0,5 s,
  la exportación, construcción de ventanas y fiabilidad se detienen con un diagnóstico.
  Adjudica o retira uno de los intervalos antes de continuar; nunca se toma silenciosamente el último.
- **Sin fotogramas**: incluso un veredicto válido en el primer intento se marca
  `sin_fotogramas` cuando no se aportaron imágenes; no cuenta como evidencia multimodal.

### Cómo redactarlo en la tesis

> Para mitigar la circularidad propia de evaluar sobre heurísticas deterministas, se implementó un protocolo
> de anotación desacoplada: el conjunto de entrenamiento se ensambló por supervisión débil a partir de
> señales de Faster-Whisper, Silero VAD y MediaPipe, mientras que el conjunto de prueba se etiquetó mediante
> inferencia ciega de un modelo multimodal (VLM) sobre fotogramas, transcripción de contexto y mediciones
> acústicas objetivas, incluyendo muestreo negativo para estimar la cobertura de las heurísticas. La calidad
> del etiquetado automático se cuantificó contra una submuestra anotada por un evaluador humano
> (κ = [DATO POR VERIFICAR]), y el benchmark se evaluó con partición estratificada por hablante (GroupKFold)
> e intervalos de confianza por bootstrap de clúster.

## Flujo de anotación humana (alternativa o complemento)

1. Abrir cada `.eaf` en ELAN. El tier `auto_candidates` trae las propuestas y `auto_evidence` explica cada una.
2. El anotador escribe su decisión en el tier `human_disfluency`, que usa el vocabulario controlado de 7 clases.
3. `python scripts/build_windows.py --labels human`
4. `python scripts/run_ablation.py --data data/windows_human.npz` → `results/ablation_table.md`

`run_ablation.py` se niega a correr con etiquetas `auto` (sería circular: las reglas usan
rasgos cinésicos y favorecerían al modo trimodal). `--allow-auto-labels` solo sirve para prueba de humo.

## Qué está verificado y qué no

- Verificado (en entorno sin GPU):
  - 10 pruebas unitarias: sync, acústica, cascada heurística, round-trip EAF, orden 12-D/Euler, ffmpeg, segmentos faciales.
  - Corpus sintético end-to-end (`tests/smoke_synthetic_corpus.py`): recorre todo el pipeline, y el benchmark detecta
    una ganancia trimodal plantada por construcción (`results_smoke_synthetic/`). Esto valida el código, no el modelo.
- **No verificado aquí**: descarga real de YouTube, Whisper y MediaPipe sobre video real. El entorno donde se
  construyó bloquea YouTube, HuggingFace y los modelos de Google. La primera corrida real es la sección "Prueba con 1 video".

## Consideraciones legales y éticas (leer antes de minar)

- **Términos de YouTube**: prohíben descargar salvo por medios que el servicio habilite. La licencia de cada
  video NO filtra, pero se registra en `data/raw/sources_log.csv` y en el manifest; cita la fuente y no
  redistribuyas audio ni video.
- **Datos personales**: voz y rostro de oradores reales que no dieron consentimiento para tu investigación.
  Consulta con tu asesor o comité de ética si este uso es aceptable bajo la Ley 29733 y el reglamento vigente.
  Recomendado: no redistribuir audio ni video, publicar solo rasgos derivados y metadatos.
- **Sesgo del corpus**: los oradores TEDx son fluidos y ensayados. Esperar pocos `block` y `prolongation`.
  Complementar con el corpus propio grabado con consentimiento.

## Umbrales

Todos los valores de `core/constants.py::H` son «umbrales heurísticos de diseño preliminares propuestos
por el autor, sujetos a calibración experimental». El desfase A/V (`--av-offset-ms`) por defecto es 0:
calíbralo con la prueba del aplauso.
