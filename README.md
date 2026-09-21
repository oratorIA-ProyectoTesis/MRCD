# MRCD — Dataset, extracción y benchmark de ablación

Motor de Reconocimiento Contextual mediante Fusión Multimodal para la Detección de
Disfluencias del Habla en Español: módulos de datos, extracción de señales y spike
experimental. Sin API web, UI ni Docker.

```
config/sources.yaml            consultas de descubrimiento + criterios de inclusión
core/constants.py              taxonomía (7 clases), vector 12-D canónico, umbrales heurísticos
core/extractors/acoustic.py    Silero VAD + pYIN (F0) + RMS a 100 fps
core/extractors/linguistic.py  Faster-Whisper verbatim + rasgos por palabra (+ RoBERTa-BNE opcional)
core/extractors/kinesic.py     MediaPipe Face Landmarker -> vector 12-D + filtro de rostro continuo
core/sync/buffer.py            buffer por timestamp de captura, jitter, offset A/V, interpolación lineal
core/annotation/auto_labeler.py  cascada heurística -> candidatos, ventanas, export ELAN (.eaf)
core/dataset.py                tensores por ventana; lectura de etiquetas humanas desde .eaf
core/models/fusion.py          cross-attention (Q=acústica, K/V=[texto; cinésica]) + clasificador
scripts/data_mining.py         discover / download / extract (yt-dlp + ffmpeg + filtro MediaPipe)
scripts/extract_and_label.py   3 señales + pre-etiquetado + dataset_manifest.json + .eaf
scripts/build_windows.py       ventanas 3 s / 500 ms -> windows_{auto|gold|human}.npz
scripts/llm_judge.py           juez multimodal ciego -> etiquetas gold independientes
scripts/judge_agreement.py     fiabilidad del juez (vs heurísticas, vs otro modelo, vs humano)
scripts/relabel.py             re-etiqueta sin repetir ASR/MediaPipe
scripts/run_ablation.py        audio_only vs audio_text vs trimodal, F1 por clase, IC 95 %
tests/                         27 pruebas + corpus sintético end-to-end
```

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
$env:ANTHROPIC_API_KEY="sk-ant-..."      # o  $env:OPENAI_API_KEY="sk-..."
powershell -ExecutionPolicy Bypass -File .\run_agentic.ps1
```

Etapas: `build_windows --labels auto` (train) → `llm_judge.py` (juicios ciegos, caché reanudable en
`data/judge/judgments.jsonl`) → `build_windows --labels gold` (test) → `judge_agreement.py` (fiabilidad) →
`run_ablation.py --data data/windows_auto.npz --test-data data/windows_gold.npz`.

Detalles que sostienen la validez:

- **Muestreo negativo**: además de los candidatos, el juez recibe ventanas de 3 s SIN candidato. Sin esto,
  el test heredaría la cobertura de las heurísticas y no mediría lo que estas no ven.
- **Estratificación**: hasta `--per-class` ítems por clase, repartidos entre oradores.
- **Partición**: se entrena con los hablantes de train (etiquetas silver) y se evalúa con los hablantes de
  test (etiquetas del juez). Ningún orador cruza la frontera.
- **Validación humana (obligatoria para la tesis)**: `judge_agreement.py --make-template 60` genera
  `data/judge/human_check.csv`. Un humano etiqueta esos 60 intervalos (aprox. 1 hora) y el script calcula
  el kappa juez–humano. Sin ese número, nadie puede saber si el juez acierta, y la tabla de ablación queda
  sin respaldo.

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
