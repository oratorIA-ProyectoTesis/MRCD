# MRCD — Dataset, extracción y benchmark de ablación

Motor de Reconocimiento Contextual mediante Fusión Multimodal para la Detección de
Disfluencias del Habla en Español. El motor es un paquete de Python (`mrcd/`) que toma audio
(y, opcionalmente, video), extrae señales acústicas, léxicas y visuales, las fusiona por ventanas
y emite eventos de disfluencia con intervalos y puntajes. Incluye además las herramientas de
datos, evaluación y reproducibilidad usadas para estudiarlo. Sin API web de producción ni Docker.

> **Estado: resultados exploratorios.** Ninguna clase de disfluencia está confirmada como detectable
> con evidencia suficiente, los puntajes del modelo **no están calibrados** (no son probabilidades) y
> los umbrales son heurísticos preliminares. Las cifras de `results/` documentan corridas concretas y
> sus límites; no deben citarse como desempeño validado.

> **Licencia:** este repositorio todavía **no incluye un archivo `LICENSE`**; la decisión está pendiente El metadato de `pyproject.toml` tampoco declara licencia.
> de los autores. Hasta entonces se aplican los derechos reservados por defecto. Los corpus de terceros
> **no se redistribuyen**: ver [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) y [`docs/corpora/`](docs/corpora/README.md).

```
mrcd/                        paquete del motor (antes `core/`)
  constants.py               taxonomía, vector 12-D canónico, umbrales heurísticos
  extractors/                acústica (Silero VAD + pYIN + RMS), lingüística (Faster-Whisper), cinésica, prosodia
  sync/ · live/              buffer por timestamp; sesión en vivo, ASR en streaming, pYIN reanudable
  annotation/auto_labeler.py cascada heurística -> candidatos, ventanas, export ELAN (.eaf)
  models/fusion.py           cross-attention (Q=acústica, K/V=[texto; cinésica]) + clasificador
  engine.py · decision.py    inferencia por ventanas y decisión por eventos (3 estados)
  asr_profiles.py · asr_recovery/   perfiles de ASR (incl. opt-in sin bucles/CUDA) y experimento de recuperación
  calibration.py · fusion_text.py · av/   calibración y abstención experimentales, fusión de texto, compuerta de evidencia facial (exploratorio)
  context_eval/ · corpora/   evaluación contextual e importadores de corpus (solo código)
  service/                   servicio local (contrato mrcd.analysis/v1), CLI y servidor HTTP
scripts/                     minería, extracción, ventanas, ablación, línea base, paridad, auditorías
tests/                       ~800 pruebas (varias requieren datos locales y se omiten sin ellos)
docs/contracts/              contratos de eventos, servicio, ASR y cabezal, con ejemplos validados
docs/architecture/           modelo C4 (Mermaid/PNG/SVG) y verificador
docs/corpora/                manifiestos, hashes SHA-256 y cómo obtener cada corpus
docs/oe4/                    informe OE4 reducido (hojas de firma del asesor NO incluidas)
docs/sus-kit/                borrador de kit SUS (no aplicado; sin datos de personas)
results/                     informes y métricas agregadas (sin texto, etiquetas ni audio de terceros)
```

Documentación: [contratos](docs/contracts/README.md) · [arquitectura](docs/architecture/README.md) ·
[corpus](docs/corpora/README.md) · [informe OE4](docs/oe4/INFORME_OE4_REDUCIDO.md).

Los ejemplos y *fixtures* de contratos tienen los campos de texto reemplazados por `[text withheld]`
(el texto original procedía de un video de terceros). Las pruebas que necesitan datos locales (corpus,
clips, cachés de ASR) se omiten con un mensaje explícito si esos datos no existen.

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
  - la suite de pruebas por defecto (ver abajo): sync, acústica, cascada heurística, round-trip EAF, orden 12-D/Euler, ffmpeg, segmentos faciales.
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

## El ASR borra disfluencias: transcribir por trozos

Whisper se entrena para producir texto legible. Cuanto más contexto ve de una vez, más
«corrige» hacia prosa fluida — y borra exactamente lo que este sistema debe detectar.

Demostración causal sobre el mismo audio, mismo checkpoint, misma ventana (53.7 s):

| ASR            | texto             | predicción              |
| -------------- | ----------------- | ----------------------- |
| archivo entero | «La primera»      | `prolongation` 44.0 %   |
| trozos de 8 s  | «La… la… primera» | **`repetition` 94.8 %** |

El clasificador no fallaba: nunca recibió la evidencia. Marcadores que sobreviven según
el tamaño de trozo (`results/asr_chunk_study.md`, sesión de 90 s): 5 s → 5, **8 s → 6**,
13 s → 6, 30 s → 4, entero → 3. Las 2 repeticiones que el hablante dijo realmente
sobreviven con ≤ 13 s y se destruyen con ≥ 30 s.

◆ **Afecta al corpus de entrenamiento**: la extracción transcribía segmentos enteros y
**64 de 87 grabaciones superan los 30 s** (mediana 51 s). `detect_repetitions` opera
sobre las palabras del ASR, así que una repetición que Whisper no transcribe no puede
etiquetarse. Por defecto ahora `--asr-chunk-s 8`; `0` reproduce el comportamiento anterior.

```powershell
python scripts/asr_chunk_study.py --wav sesion.wav              # marcadores por tamaño de trozo
python scripts/diagnose_session.py --wav sesion.wav --counterfactual   # capa ASR vs clasificador
python scripts/compare_extractions.py --b data/v4/dataset_manifest.json
```

`diagnose_session.py --counterfactual` reevalúa cada ventana con la acústica intacta y el
léxico sustituido por palabras neutras, conservando los tiempos. Si la clase se mantiene,
la decisión venía de la señal acústica; si colapsa, el modelo leía palabras. Medido sobre
la sesión de prueba: **27/30 sobreviven** — el modelo no reaprendió la cascada heurística.

## Guardias de validez (el sistema no puede inflar una cifra en silencio)

Medido con `scripts/identity_probe.py` sobre este corpus: una sonda **lineal** identifica cuál de
17 charlas es una ventana de 3 s con **75 % de exactitud balanceada usando solo la cara** (12.8× el
azar de 0.0588). La rama acústica llega a 10.9×; la prosódica, que ya se centraba por hablante, a 3.0×.
Cualquier protocolo que deje material de la misma charla a ambos lados del corte le entrega esa señal
al clasificador, y el F1 resultante mide reconocimiento de charla, no detección de disfluencias.

De ahí estas guardias:

| requisito                  | dónde                                       | comportamiento                                                                                                                             |
| -------------------------- | ------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| Partición sin fuga         | `run_ablation.py --partition`               | `speaker` por defecto. `recording` y `window` **abortan** sin `--leaky-ok`, y con él escriben `ablation_table.LEAKY.md`, nunca la oficial. |
| Normalización por hablante | `build_windows.py --kinesic-norm`           | `speaker` por defecto: z-score cinésico por canal. Baja la fuga de 12.8× a **4.0×**.                                                       |
| Soporte mínimo             | `mrcd/constants.py::MIN_SUPPORT_FOR_METRIC` | Una clase con soporte < 10 sale como «sin datos suficientes» y queda fuera del macro-F1. `block` (n=2) ya no recibe un número.             |
| Versión de esquema         | `mrcd/constants.py::FEATURE_SCHEMA_VERSION` | Se sella en cada `.npz` y `.pt`. `load_ckpt` **falla** ante versión ausente o distinta en vez de adaptarse.                                |
| Modo solo-video            | `MODES` incluye `kinesic_only`              | Sin rama acústica ni prosódica, aunque se le pase `pros_dim`. Distingue «el video no tiene información» de «la fusión la ignora».          |

Límites declarados, no resueltos:

- La normalización por hablante **no elimina** la fuga. Con rasgos invariantes a transformación afín
  (autocorrelación y correlación cruzada) la identificabilidad es **9.2× antes y después**: la dinámica
  facial también identifica al hablante.
- Tras el remedio, la rama **acústica** pasa a ser la peor fuga (10.9×). Solo se le centra el F0.
- **No se aplica a la rama lingüística**: sus rasgos binarios dispersos empeoran (3.6× → 11.3×).
- La calibración usa todo el material del hablante: es normalización **transductiva**. No toca
  etiquetas y en despliegue ese material existe, pero no es un protocolo inductivo puro.

[`PREREGISTRO.md`](PREREGISTRO.md) **no está congelado ni vigente**: conserva campos `[COMPLETAR]` y cubre una pregunta anterior (fuga intra-charla audiovisual). Es un documento histórico, pendiente de actualizar y ratificar con un commit real antes de citarlo como protocolo.

Pruebas (todas con `OMP_NUM_THREADS=4`; detalle de quién midió qué en el addendum de la ronda 7 de `results/ejecucion_material4_20261030/INFORME_FINAL.md`):

- Autor, árbol local, tras la ronda 7: 741 aprobadas y 9 omitidas por defecto; 750 aprobadas y 0 omitidas con `MRCD_SLOW=1` (ejecutado solo, sin otra carga). Autor, ronda 6 (histórico): 698 + 9 y 707 + 0; el revisor independiente reprodujo esas dos cifras.
- Revisor independiente, árbol limpio en 4443d71 (sin datos ignorados por git), por defecto: 681 aprobadas y 26 omitidas. El autor no volvió a medir un árbol limpio tras la ronda 7. Las cifras antiguas de árbol limpio (639 + 17 / 641 + 15) son históricas.
- Rama publicable (sin ningún dato local ni corpus de terceros), suite por defecto con `OMP_NUM_THREADS=4`, clon limpio de la rama tras la actualización del 2026-11-01: **895 aprobadas, 27 omitidas, 0 fallidas** (141 s). Medición previa: 797 + 26. Las omisiones declaran su motivo (`pytest -rs`): datos de corpus ausentes (`material/`, DIHANA, AMERESCO, cachés de ASR) o suites lentas (`MRCD_SLOW=1`).
- La diferencia entre árbol local y limpio son pruebas que requieren datos locales (DIHANA `turns.csv`, clips/caché de AMERESCO, clips ligados a checkpoints).
- **"Bit-idéntico" solo se verifica en una máquina descargada con `OMP_NUM_THREADS=4` fijado antes de importar numpy.** Las pruebas lentas de decodificación comparan flotantes a 1e-6 y son sensibles a la carga: con dos suites concurrentes fallaron 6 en cada una. Ahora se omiten (no fallan) si la CPU está ocupada en más de 50 % o si `OMP_NUM_THREADS` no es 4; `MRCD_SKIP_LOAD_CHECK=1` fuerza la ejecución. No ejecutes la suite lenta junto con otro proceso pesado.

```powershell
python scripts/identity_probe.py                                     # R1: fuga por rama
python scripts/identity_probe.py --features invariant                # R4: identidad residual
python scripts/build_windows.py --labels auto --kinesic-norm speaker --out data/v3/windows_auto.npz
python scripts/run_ablation.py --data data/v3/windows_auto.npz --test-data data/v3/windows_gold.npz
```

## Umbrales

Todos los valores de `mrcd/constants.py::H` son «umbrales heurísticos de diseño preliminares propuestos
por el autor, sujetos a calibración experimental». El desfase A/V (`--av-offset-ms`) por defecto es 0:
calíbralo con la prueba del aplauso.
