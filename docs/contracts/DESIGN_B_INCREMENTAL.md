# Diseño: variante "B incremental" del análisis MRCD de la sesión

| Campo                             | Valor                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| --------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Estado | **Implementado el 2026-11-01 (autorrevisión; sin revisión independiente), APAGADO por defecto.** La implementación sigue este diseño; las diferencias y la evidencia están en `results/mejora_objetivos_20261101/S8_incremental/README_S8.md`. Diferencias deliberadas: la reconexión del navegador no usa `Last-Event-ID` (el servidor de la app siempre emite primero el estado actual, que basta porque el avance son contadores monótonos); el evento SSE del navegador solo anuncia `completed`/`unavailable` y la pantalla lee el documento por la ruta autorizada de la sesión; en MRCD las cabeceras `X-Offset-Samples` y `X-Content-SHA256` acompañan al cuerpo PCM. |
| Fecha                             | 2026-11-01                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| Repos afectados                   | `<repo>` (motor y `mrcd.service`, contrato `mrcd.analysis/v1` rev 1.2) y `ORATORIA-PROYECTO` rama local `feat/mrcd-integration` (backend + frontend), trabajados por separado                                                                                                                                                                                                                                                                                                                                  |
| Variante de referencia            | **B original**: el backend acumula el PCM de `/ws/session` en memoria y lo envía una sola vez a MRCD al cerrar (`docs/contracts/PRIVACY_DECISION.md` §6-§7, `results/mejora_objetivos_20261101/S7_app/README_S7.md` §9)                                                                                                                                                                                                                                                                                                      |
| Variante nueva                    | **B incremental**: el mismo audio, solo en memoria, llega a MRCD por partes durante la sesión; MRCD decodifica cada fragmento ASR de 8 s en cuanto está completo y, al cerrar, ejecuta lo que depende del archivo completo                                                                                                                                                                                                                                                                                                   |
| Condiciones aprobadas (literales) | No cambiar Whisper, sus parámetros, el preprocesamiento ni los algoritmos posteriores; equivalencia EXACTA contra el modo offline (transcripción, timestamps, eventos, clasificación, puntuaciones y reporte final) con pruebas automatizadas sobre las mismas grabaciones; tolerancia a fallos sin perder ni duplicar fragmentos; el offline queda como referencia y fallback; documentar como variante 'B incremental' frente a B original; benchmark de ambos modos. Perfiles GPU (`*_cuda`) fuera de los predeterminados |
| Activación                        | Opt-in con bandera en ambos repos; **offline por defecto** hasta demostrar la equivalencia                                                                                                                                                                                                                                                                                                                                                                                                                                   |

Convención de citas: `ruta:línea` relativo a la raíz de cada repo. `app:` = `ORATORIA-PROYECTO`. `fw:` = `.venv/Lib/site-packages/faster_whisper/transcribe.py` (faster-whisper 1.2.1, ctranslate2 4.8.2 en el venv del motor).

---

## 1. Hallazgos (A-D) con evidencia

### A. Ruta offline exacta del ASR

**Cadena de llamadas.** `MrcdAnalyzer.analyze_bytes` (`mrcd/service/analyzer.py:682-691`) calcula el sha256 de los BYTES del WAV recibido (`analyzer.py:687`, `mrcd/service/audio_io.py:37-38`), decodifica el WAV a `int16 -> float32 / 32768` (`audio_io.py:107`, `audio_io.py:122`) y, bajo un candado, llama a `DisfluencyEngine.analyze` (`analyzer.py:762-766`). `analyze` valida mono, finitud y 16 kHz y **rechaza** remuestrear (`mrcd/engine.py:314-324`), exige `MIN_AUDIO_S = CONTEXT_S + WINDOW_S = 13 s` (`engine.py:167`, `engine.py:325-331`) y transcribe con `self._asr.transcribe_chunked(audio, chunk_s=8.0)` (`engine.py:333-339`).

**Preprocesado global antes del ASR: ninguno.** No hay normalización de amplitud, remuestreo, recorte ni VAD previos al ASR en esta ruta: el único cambio es la conversión elemento a elemento `int16/32768` (`audio_io.py:122`), que no depende del archivo completo. El VAD interno de faster-whisper está apagado (`vad_filter=False`, `mrcd/asr_profiles.py:54`, pasado en `asr_profiles.py:82`).

**Cada fragmento se decodifica de forma independiente.**

- Parámetros por perfil (`asr_profiles.py:47-58`, `asr_profiles.py:75-86`): `temperature=0.0` (un solo valor), `beam_size=5`, `condition_on_previous_text=False`, `initial_prompt` fijo por perfil (p0 = `WHISPER_INITIAL_PROMPT`, `mrcd/constants.py:150`; p1 = `P1_PROMPT`, `asr_profiles.py:29-33`), `word_timestamps=True`, `suppress_tokens=[]`.
- Con un `temperature` escalar faster-whisper construye `temperatures=[0.0]` (`fw:982-983`) y solo muestrea si `temperature > 0` (`fw:1432-1438`): con 0.0 es búsqueda por haz determinista, sin semilla ni fallback de temperatura.
- `VerbatimASR.transcribe` (`mrcd/extractors/linguistic.py:198-205`) llama una vez a `WhisperModel.transcribe(audio, **kwargs)` (`linguistic.py:51-67`); no guarda texto ni contexto de llamadas anteriores. `condition_on_previous_text=False` solo afecta a las ventanas de 30 s DENTRO de una llamada; un fragmento de 8 s es una sola ventana.
- Estado entre fragmentos: solo `self._chunk_offset_s` y `self.redecode_events` (`linguistic.py:270-271`, `linguistic.py:283`), usados únicamente por los perfiles `*_noloop` para anotar metadatos.

**Límites de fragmento** (`linguistic.py:249-300`):

1. Si `len(audio) <= 8 s` se hace UNA pasada sobre todo el audio y se devuelve sin filtrar (`linguistic.py:272-273`). En el servicio no se alcanza (mínimo 13 s), pero el diseño lo respeta.
2. `step = int(6.0 * 16000) = 96000`, `size = int(8.0 * 16000) = 128000` muestras (`linguistic.py:275-276`); inicios `0, 96000, 192000, ...` mientras `start < len(audio)` (`linguistic.py:278`); el segmento es `audio[start:start+size]` y el bucle **se corta** en el primer segmento de menos de 1 s (`linguistic.py:279-281`).
3. Un fragmento con `start + size <= len(audio)` es **completo**: su contenido no depende de la duración final y puede decodificarse en cuanto llega su última muestra. Solo los fragmentos **parciales** (`len(audio) - start` en `[1 s, 8 s)`) dependen de la duración final; con paso 6 s y tamaño 8 s hay como máximo **dos** (sus inicios caen en un intervalo de 7 s).

**Regla exacta de fusión en el solape** (`linguistic.py:284-299`): para el fragmento `k>0` se descartan las palabras cuyo punto medio relativo `(start+end)/2 < 2.0 s`; las demás se desplazan `+ start/16000` y se agregan en orden de fragmento; al final, `out.sort(key=lambda w: w.start)` (orden estable). Es deduplicación por región, no por texto (`linguistic.py:265-268`).

**Re-decodificación `*_noloop`** (perfiles no predeterminados, `asr_profiles.py:166-181`): dentro de `transcribe`, si el texto del fragmento parece un bucle se re-decodifica con la escalera fija (`asr_profiles.py:397-401`; `linguistic.py:207-247`). Sigue siendo **por fragmento** (no rompe la independencia), pero: (a) agrega eventos a `redecode_events` con `chunk_offset_s` (`linguistic.py:225-244`), que el documento final publica (`engine.py:469-476`); (b) el peldaño 2 muestrea a `T=0.4` con `best_of=5` y el propio código lo declara **no reproducible bit a bit** pese a la semilla fija (`linguistic.py:232-233`, `asr_profiles.py:171`). Consecuencia: para `*_noloop` la equivalencia exacta solo puede exigirse cuando `n_non_reproducible == 0`; si un fragmento usó el peldaño muestreado, ni el offline es reproducible contra sí mismo.

**Verificación empírica** (sección 1.A.1, más abajo): script de solo lectura en el scratchpad que decodifica cada fragmento por separado, en otro proceso, a partir de bytes recibidos por partes, intercalando grabaciones, y fusiona con la misma regla.

#### 1.A.1 Resultado literal de la verificación

Script: `verify_incremental.py` en el scratchpad de la sesión (fuera de ambos repos; no se versiona). Procedimiento, en **dos procesos distintos**, ambos con `OMP_NUM_THREADS=4` y `HF_HUB_OFFLINE=1` en el entorno antes de Python, `MrcdAnalyzer(device="cpu")`:

1. **Proceso offline**: para cada entrada, WAV canónico (misma cabecera de 44 bytes que `app:.../audio_buffer.py:147-164`) -> `analyze_bytes(..., include_text=true, decision=true)`, capturando la salida de `transcribe_chunked` sin alterarla.
2. **Proceso incremental**: las 11 entradas se "reciben" a la vez, intercaladas por turnos (bloques de 160 ms en la mitad de ellas; tamaños aleatorios de 1 a 40 000 muestras con semilla `20261101` en la otra mitad). Cada fragmento de 8 s se decodifica con `VerbatimASR.transcribe` en cuanto su última muestra llega, convirtiendo SOLO los bytes acumulados hasta ese momento; los fragmentos de distintas grabaciones y perfiles quedan intercalados. Al "cerrar" se decodifican los parciales y se fusiona con la regla de `linguistic.py:284-299`; después `analyze_bytes` con `transcribe_chunked` sustituido por las palabras en caché.
3. **Comparación**: palabras como `(text, norm, repr(start), repr(end), repr(prob), punct_after)` y documento completo con `==`, excluyendo solo `generated_at`.

Entradas: sintética del E2E de S7 (45,466 s), `c60s` (60,0 s, múltiplo de 6), cortes exactos de `c60s` a 32,0 s (6k+8), 31,0 s, 37,0 s (último parcial = 1,0 s), 36,999 s (último parcial < 1 s, se descarta), 48,0 s (múltiplo de 6 y de 8), AMERESCO `a24` (13,6 s), `w22s` (22,0 s); p0 en las 9 y `p1_disfluent_prompt` en 2. No se imprime ni se reproduce texto.

Salida literal (`compare.log`):

```
e2e_45.466s|p0: words=79 words_equal=True doc_equal=True doc_sha=1c59a48768b62a12/1c59a48768b62a12 events=11 sizes=160ms calls=8 chunks=8 offline_wall=26.86s close_wall=8.31s diff=[]
c60_60.0s_mult6|p0: words=183 words_equal=True doc_equal=True doc_sha=2f4fabc35af0d867/2f4fabc35af0d867 events=18 sizes=random calls=10 chunks=10 offline_wall=33.77s close_wall=6.61s diff=[]
c60_cut32.0s_6k+8|p0: words=98 words_equal=True doc_equal=True doc_sha=b44b31952c70fa87/b44b31952c70fa87 events=9 sizes=160ms calls=6 chunks=6 offline_wall=12.17s close_wall=4.19s diff=[]
c60_cut31.0s|p0: words=94 words_equal=True doc_equal=True doc_sha=b474cad52b3a3c39/b474cad52b3a3c39 events=9 sizes=random calls=6 chunks=6 offline_wall=18.48s close_wall=7.45s diff=[]
c60_cut37.0s_last1s|p0: words=115 words_equal=True doc_equal=True doc_sha=f0d7475294e287bd/f0d7475294e287bd events=13 sizes=160ms calls=7 chunks=7 offline_wall=23.83s close_wall=7.98s diff=[]
c60_cut36.999s_lastlt1s|p0: words=113 words_equal=True doc_equal=True doc_sha=13b627b11bd9b836/13b627b11bd9b836 events=14 sizes=random calls=6 chunks=6 offline_wall=21.99s close_wall=4.85s diff=[]
c60_cut48.0s_mult6and8|p0: words=156 words_equal=True doc_equal=True doc_sha=d97859e9c76ac1d7/d97859e9c76ac1d7 events=16 sizes=160ms calls=8 chunks=8 offline_wall=25.68s close_wall=5.39s diff=[]
a24_13.6s|p0: words=38 words_equal=True doc_equal=True doc_sha=082566a6b988dce9/082566a6b988dce9 events=1 sizes=random calls=3 chunks=3 offline_wall=8.83s close_wall=6.18s diff=[]
w22_22.0s|p0: words=50 words_equal=True doc_equal=True doc_sha=a4bf9dcc54b46a60/a4bf9dcc54b46a60 events=5 sizes=160ms calls=4 chunks=4 offline_wall=13.12s close_wall=3.81s diff=[]
e2e_45.466s|p1_disfluent_prompt: words=79 words_equal=True doc_equal=False doc_sha=a9644e7a7e549ebd/1d15baef17f1ab52 events=8 sizes=random calls=8 chunks=8 offline_wall=23.08s close_wall=4.33s diff=['provenance']
a24_13.6s|p1_disfluent_prompt: words=36 words_equal=True doc_equal=False doc_sha=1d8e2a5730adb323/9758ca511ef522c4 events=3 sizes=160ms calls=3 chunks=3 offline_wall=6.47s close_wall=6.17s diff=['provenance']
ALL_EQUAL False
```

Diferencia única en los dos casos p1 (diff recursivo de `provenance`):

```
.environment.cpu_threads.torch | 4 | 1
```

**Lectura:**

- **Palabras: 11/11 idénticas bit a bit** (texto, tiempos y probabilidades) en p0 y p1, con fragmentos decodificados en otro proceso, en otro orden, intercalados con otras grabaciones y a partir de bytes recibidos por partes. Cada fragmento se decodificó exactamente una vez (`calls == chunks`). Esto confirma empíricamente la independencia por fragmento descrita arriba.
- **Documento: 9/9 idénticos en p0.** En p1, eventos, clases, puntuaciones, decisiones, resumen y transcrito son idénticos; difiere solo el campo de procedencia `provenance.environment.cpu_threads.torch` (4 frente a 1).
- **Causa (verificada aparte):** `silero_vad/model.py:3` (venv) ejecuta `torch.set_num_threads(1)` al importarse, lo que ocurre en el primer análisis del proceso (`acoustic.py:42`); y crear un `WhisperModel` vuelve a fijar el pool de OpenMP compartido en 4. Traza literal en un proceso limpio: `start 4`, `after import silero_vad 1`, `after import faster_whisper 1`, `after WhisperModel() 4`, `after transcribe 4`, `after load_ckpt 4`. Por eso el valor depende de la **historia del proceso**: en el proceso offline el motor p1 se cargó después del primer análisis (vuelve a 4); en el incremental se cargó antes (queda en 1). **No es un efecto del modo incremental: el propio offline reporta un valor distinto según el orden en que se cargaron los motores.** Aquí los valores numéricos de V0 coincidieron con 4 y con 1 hilos, pero eso no está garantizado en general (las reducciones en coma flotante pueden depender del número de hilos).
- **Consecuencia para el diseño** (se incorpora en D y en 2.6): la equivalencia exacta exige fijar el estado del proceso ANTES de atender peticiones: precargar todos los perfiles habilitados y luego ejecutar el calentamiento (que importa Silero), de modo que todo análisis, offline o incremental, corra con el mismo número de hilos de torch. La referencia de las pruebas es siempre el offline **en el mismo proceso ya calentado**.
- **Otra observación de la ejecución**: un primer intento con `MrcdAnalyzer()` (dispositivo `auto`) imprimió `[asr] GPU no utilizable (Library cublas64_12.dll is not found ...); usando CPU int8` al cargar p0 y luego **se detuvo al cargar el segundo motor (p1)**: el tiempo de CPU del proceso quedó fijo en 0:08:36 durante más de 20 min. No se diagnosticó. Con `device="cpu"` (el mismo CPU int8 al que caía el modo `auto`) no ocurrió.
- Los tiempos `offline_wall`/`close_wall` son solo indicativos (la máquina no estaba aislada y varían hasta ×1,6 entre corridas del mismo archivo); no son un benchmark. En archivos cortos (13,6 s) casi todo el audio está en fragmentos parciales y el cierre cuesta casi lo mismo que el offline.

**Conclusión de A**: la equivalencia exacta es posible para p0 y p1 en CPU con la caché por fragmento, **siempre que el estado de hilos del proceso esté fijado** (precarga + calentamiento) y el dispositivo esté fijado en CPU. Para `*_noloop` vale solo sin peldaño muestreado.

### B. Partes que dependen del archivo completo (se ejecutan al cierre)

`DisfluencyEngine._evaluate` (`engine.py:529-605`) extrae TODO sobre el audio completo:

| Paso                                                                                         | Dónde                                                 | Por qué es global                                                                                                                                                                                                                        |
| -------------------------------------------------------------------------------------------- | ----------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Rejilla de ventanas `0, 0.5, ... <= dur-3`                                                   | `engine.py:545-548`                                   | depende de la duración final                                                                                                                                                                                                             |
| Rasgos acústicos: RMS, pYIN, STFT, VAD                                                       | `engine.py:560`, `mrcd/extractors/acoustic.py:90-126` | pYIN decodifica con Viterbi sobre toda la señal (`acoustic.py:101-102`); Silero VAD corre sobre el audio entero (`acoustic.py:113-115`, `acoustic.py:42-47`); hay una comprobación global `abs(audio).max() > 1.5` (`acoustic.py:96-97`) |
| Centro F0 del hablante                                                                       | `engine.py:561`, `mrcd/extractors/prosody.py:141`     | estadístico sobre todos los cuadros                                                                                                                                                                                                      |
| Ventanas lingüísticas                                                                        | `engine.py:562-565`                                   | usan todas las palabras fusionadas                                                                                                                                                                                                       |
| Inferencia V0 (`bs=1`) y cabeza Vc opcional                                                  | `engine.py:577-580`, `engine.py:607-639`              | por ventana, pero necesita los tensores anteriores                                                                                                                                                                                       |
| Fusión en eventos                                                                            | `engine.py:641-695`                                   | recorre todas las ventanas                                                                                                                                                                                                               |
| Metadatos ASR (huella con sha256 del audio completo, anomalías sobre el transcrito completo) | `engine.py:349`, `engine.py:458-467`, `engine.py:498` | globales                                                                                                                                                                                                                                 |
| Documento v1 (eventos, resumen, decisión, avisos)                                            | `analyzer.py:286-535`                                 | función pura del resultado                                                                                                                                                                                                               |

**Costo medido** (`results/cierre_oe1_oe3_20261031/3b/cost/COST_REPORT.md:13-17`; CPU, OMP=4, i7-13650HX): ASR p0 RTF mediana 0.44 (p1 0.53), rasgos acústicos/prosódicos (Silero + pYIN) 0.039, V0 0.0015, Vc 0.0014, ventanas lingüísticas, decisión y reporte < 0.001 cada uno. **El ASR es ~90 % del tiempo**; lo que obligatoriamente queda para el cierre es ~9 % del costo. Para 600 s de audio: offline ≈ 0.48 × 600 ≈ 290 s después de cerrar; incremental ≈ 0.04 × 600 (≈ 24 s) + ASR de los ≤ 2 fragmentos parciales (≤ 15 s de audio, ≈ 7 s) + fragmentos completos aún en cola. Estimación, no medición: la mide el benchmark (sección 4). Hay un caso atípico de costo ASR no proporcional a la duración (`COST_REPORT.md:15`).

Ni pYIN ni Silero pueden adelantarse sin cambiar el algoritmo (Viterbi global, estado del VAD), así que **no se adelantan**.

### C. `mrcd/live/streaming_asr.py::step_chunk_aligned` y lo medido en 3b

`step_chunk_aligned` (`mrcd/live/streaming_asr.py:73-124`) usa la misma rejilla 8 s / 2 s, congela fragmentos completos y re-decodifica en cada tick hasta dos parciales. En 3b se midió desacuerdo 0 con el offline y 0 % de cambios de clase por ventana (`results/cierre_oe1_oe3_20261031/3b/asr/ASR_STREAMING_VS_OFFLINE.md:50`, `:53`), y las pruebas comparan `(norm, round(start, 3))` con un ASR falso (`tests/test_asr_chunking.py:123-132`).

**Decisión: NO reutilizarlo para B incremental; usar una caché de resultados ASR por fragmento consultada por el MISMO `transcribe_chunked` del offline.** Motivos:

1. **Reimplementa la planificación y la fusión** en lugar de ejecutar la del offline: calcula índices con flotantes y `int(round(...))` (`streaming_asr.py:95-96`) frente a enteros en el offline (`linguistic.py:275-279`); no replica el atajo `len <= 8 s` (`linguistic.py:272-273`): con audio en `(7 s, 8 s]` decodifica un fragmento 1 provisional que el offline nunca decodifica (`streaming_asr.py:118`), y solo el filtro de punto medio suele ocultarlo.
2. **No prueba igualdad bit a bit**: las pruebas redondean tiempos a 3 decimales y no comparan probabilidades (`tests/test_asr_chunking.py:131`).
3. **Re-decodifica los parciales en cada tick** (costo sin utilidad para el reporte final) y con `*_noloop` acumularía `redecode_events` duplicados de los ticks provisionales sin fijar `_chunk_offset_s`.
4. Exige un anillo de 13 s y pierde fragmentos si el consumidor se atrasa (`streaming_asr.py:84`, `:110-111`); en el servicio no hay anillo: el audio completo está en memoria.

La caché es más simple y **exacta por construcción**: al cierre se ejecuta `engine.analyze` sin cambios, con un envoltorio de ASR cuyo `transcribe(seg)` devuelve el resultado guardado si `sha256(seg)` coincide y, si no, decodifica de verdad. Si la planificación incremental se equivocara, el efecto sería un fallo de caché (más latencia), nunca un resultado distinto.

### D. Determinismo

| Factor                               | Evidencia                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            | Qué fijar                                                                                                                                                                                                                       |
| ------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `OMP_NUM_THREADS`                    | El texto de ctranslate2 int8 depende del número de hilos; solo `4` reproduce lo fijado, y debe estar en el entorno ANTES del primer `import numpy` (`mrcd/reproducibility.py:3-18`, `:141-201`; aviso en `engine.py:217-233`)                                                                                                                                                                                                                                                                        | Mismo proceso del servicio para fragmentos y cierre, lanzado con `OMP_NUM_THREADS=4` (comando de `COMANDOS.MD`). Al reabrir tras reinicio, el proceso nuevo debe tener el mismo valor (se registra en `provenance.environment`) |
| Dispositivo                          | `device="auto"` prueba CUDA y cae a CPU int8 (`linguistic.py:109-143`); en esta máquina imprime "GPU no utilizable" y usa CPU, y en 1.A.1 el proceso se detuvo al cargar un segundo motor con `auto`. Un GPU usable cambiaría el texto (`engine.py:501-508`)                                                                                                                                                                                                                                         | Servicio arrancado con `--device cpu` en modo incremental. El modo incremental acepta solo perfiles CPU (`p0`, `p1_disfluent_prompt`; `*_noloop` con la salvedad de A); la caché incluye `device` y `compute_type` en la clave  |
| Hilos de ctranslate2                 | `WhisperModel` se construye sin `cpu_threads` (`linguistic.py:118-120`; `reproducibility.py:216-217`)                                                                                                                                                                                                                                                                                                                                                                                                | No cambiar; misma instancia de modelo por perfil                                                                                                                                                                                |
| Hilos de torch (V0, Vc, Silero ONNX) | `engine_environment` registra `torch.get_num_threads()` (`reproducibility.py:208-215`). **Medido en 1.A.1**: importar `silero_vad` lo fija en 1 y cada `WhisperModel(...)` nuevo lo devuelve a 4 (pool OpenMP compartido); el valor depende del orden de carga de motores frente al primer análisis. Fijarlo con: `--device cpu`, precarga de TODOS los perfiles habilitados y DESPUÉS calentamiento (importa Silero); sin carga perezosa en modo incremental; `/v1/health` expone el valor efectivo | El cierre corre en el mismo hilo trabajador y proceso que el offline (`server.py:40-51`)                                                                                                                                        |
| Orden de ejecución                   | `transcribe` no tiene estado entre llamadas (A); la verificación 1.A.1 intercala fragmentos de varias grabaciones y perfiles                                                                                                                                                                                                                                                                                                                                                                         | Orden libre para fragmentos; el **ensamblado** lo hace siempre `transcribe_chunked` en orden de fragmento                                                                                                                       |
| Procesos vs hilos                    | El motor no es seguro entre hilos (`server.py:41`, `analyzer.py:592`, `analyzer.py:762`)                                                                                                                                                                                                                                                                                                                                                                                                             | Toda llamada al motor (fragmento o cierre) pasa por el ÚNICO hilo trabajador; nunca dos decodificaciones simultáneas sobre el mismo modelo                                                                                      |
| Lote de inferencia                   | `predict(..., bs=1)` "evita cambios numéricos por lote" (`engine.py:576-577`)                                                                                                                                                                                                                                                                                                                                                                                                                        | No cambia: el cierre llama a `_evaluate` sin modificar                                                                                                                                                                          |
| Bytes de entrada                     | El sha256 del documento es el de los bytes del WAV (`analyzer.py:687`); el WAV de la app tiene una cabecera fija de 44 bytes (`app:backend/src/oratoria/platform/mrcd/audio_buffer.py:147-164`)                                                                                                                                                                                                                                                                                                      | Al cierre, el servicio reconstruye exactamente ese WAV canónico a partir del PCM recibido y lo pasa por `analyze_bytes`                                                                                                         |

---

## 2. Arquitectura propuesta

### 2.1 Principio rector

El cierre incremental **ejecuta el mismo código que el offline** (`MrcdAnalyzer.analyze_bytes` sobre el WAV canónico) con una sola diferencia: el ASR consulta una caché de fragmentos ya decodificados. Nada de la planificación, fusión, rasgos, clasificación ni reporte se reimplementa. Por eso la equivalencia se reduce a demostrar que `transcribe(seg)` devuelve lo mismo antes o después (hecho A) y que la clave de caché identifica unívocamente el segmento.

### 2.2 Componentes nuevos en `mrcd` (servicio; el motor no cambia)

| Componente                                | Archivo propuesto                        | Responsabilidad                                                                                                                                                                                                                                                                                                                                                                     |
| ----------------------------------------- | ---------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `ChunkPlan`                               | `mrcd/service/incremental/plan.py`       | Funciones puras con enteros: `complete_chunks(n_samples) -> [(k, start)]` (todos los `k` con `start + 128000 <= n`) y `final_plan(n)` documentado como espejo de `linguistic.py:272-281`. Solo decide QUÉ decodificar temprano; el ensamblado final no la usa                                                                                                                       |
| `ChunkCache`                              | `mrcd/service/incremental/cache.py`      | Diccionario en memoria por sesión: clave `(profile_fingerprint_base, k, start, len_samples, sha256(float32 del segmento))`, valor `list[Word]` + `redecode_events` del fragmento. `profile_fingerprint_base` = `asr_cache_key(...)` (`asr_profiles.py:218-253`) con `audio_sha256` reemplazado por el sha del segmento                                                              |
| `CachedASR`                               | `mrcd/service/incremental/cached_asr.py` | Subclase de `VerbatimASR` creada sin `__init__`, que comparte el modelo, el perfil y los atributos del ASR real y SOLO sobrescribe `transcribe(seg)`: si la clave está en caché devuelve una copia (y reinyecta sus `redecode_events` con el `_chunk_offset_s` vigente); si no, llama a `VerbatimASR.transcribe(self, seg)`. Registra `hits/misses` (no van al documento)           |
| `IncrementalSession`                      | `mrcd/service/incremental/session.py`    | `bytearray` PCM16 en memoria, ensamblado por desplazamiento de muestras, tabla de lotes recibidos `(seq -> offset, n_bytes, sha256)`, prefijo contiguo, índice del siguiente fragmento a decodificar, estado (`open`, `closing`, `completed`, `failed`, `cancelled`, `expired`), suscriptores de progreso. `discard()` sobrescribe con ceros como `audio_buffer.py:166-177`         |
| `SessionRegistry`                         | mismo módulo                             | Límite de sesiones simultáneas, de segundos por sesión (600 s, igual que `DEFAULT_MAX_DURATION_S`, `analyzer.py:48`) y de bytes totales; expiración por inactividad                                                                                                                                                                                                                 |
| `EngineWorker` (evolución de `JobRunner`) | `mrcd/service/server.py`                 | Un solo hilo y una cola con prioridad: (0) cierres, (1) fragmentos de sesiones cuyo cierre ya llegó, (2) fragmentos normales, (3) trabajos `POST /v1/jobs`. FIFO dentro de cada prioridad. Sigue habiendo UN hilo: el motor no es seguro entre hilos                                                                                                                                |
| Cierre                                    | `IncrementalSession.finalize()`          | (1) verifica contigüidad, total de muestras y sha256 del WAV canónico que declara la app; (2) construye el WAV canónico; (3) obtiene el motor del perfil, hace `copy.copy(engine)` y asigna `_asr = CachedASR(engine._asr, cache)` (la instancia compartida no se toca); (4) llama `analyzer._run(decode_wav(wav), options)` con ese motor. Resultado: documento `mrcd.analysis/v1` |
| Progreso                                  | `GET /v1/sessions/{sid}/events` (SSE)    | `StreamingResponse` `text/event-stream` (FastAPI ya instalado); sin dependencias nuevas                                                                                                                                                                                                                                                                                             |
| Precarga y calentamiento                  | `mrcd/service/cli.py` (`serve`)          | Ver 2.6                                                                                                                                                                                                                                                                                                                                                                             |

### 2.3 Cambios en el backend de la app (`app:backend/src/oratoria/platform/mrcd/`)

- **El `SessionAudioBuffer` se mantiene igual** (`app:.../audio_buffer.py:113-145`): sigue siendo la fuente de verdad, el respaldo para el fallback offline y la base del sha256 de enlace (`app:.../analysis.py:413-441`).
- **Nuevo `IncrementalUploader`** (`incremental.py`): se crea junto al búfer cuando `ORATORIA_MRCD_MODE=incremental` y hay consentimiento (`app:.../analysis.py:180-193`). Tras cada `append` exitoso (`app:backend/src/oratoria/platform/ws/connection_handler.py:384`) toma los bytes nuevos por desplazamiento (no copia el búfer completo), los agrupa en lotes de `ORATORIA_MRCD_INCREMENTAL_BATCH_MS` (2000 ms por defecto = 12,5 tramas de 160 ms del worklet, `app:frontend/src/contexts/speech-analysis/audio/pcm-worklet.ts:19`) y los envía en segundo plano con `seq` propio, `offset_samples` y `sha256` del lote. Cola acotada; nunca bloquea el bucle del WebSocket.
- **Reglas de corte idénticas a B original**: si el búfer marca hueco (`audio_buffer.py:135-139`), límite de duración o de memoria (`audio_buffer.py:126-131`), el uploader **cancela** la sesión MRCD (`DELETE`) y deja de enviar; al cerrar, las precondiciones existentes (`app:.../analysis.py:355-382`) producen el mismo `unavailable` que hoy. No hay relleno ni división en tramos.
- **Cierre** (`app:.../analysis.py:207-254`, `:412-455`): si el uploader está sano, envía los lotes pendientes y `POST /close` con `total_samples`, `wav_sha256` (de `to_wav_bytes()`) y `last_seq`; espera el resultado por SSE. Si el uploader está degradado, la sesión MRCD no existe, el cierre responde `incremental_*` o el SSE se corta sin resultado: **fallback offline** = el camino actual `POST /v1/jobs` con el WAV completo. El sha256 de enlace se verifica igual en ambos caminos (`analysis.py:432-441`).
- **Modo de verificación en campo** `ORATORIA_MRCD_MODE=incremental_verify`: además del incremental, al cierre envía el offline; guarda el offline (fuente de verdad) y registra solo `equal: true|false` y la lista de claves distintas (nunca contenido). Sirve para acumular evidencia antes de cambiar el predeterminado.
- **Progreso hacia el frontend**: `MrcdSessionAnalysis` publica en un `ProgressHub` en memoria por `session_id` (`pending`, `asr_progress`, `finalizing`, `completed`, `unavailable`) y además persiste el documento como hoy (`analysis.py:457-464`).
- **Cliente HTTP** (`app:.../client.py`): se agregan `open_session`, `append`, `close`, `cancel` y `stream_events` con `urllib` (sin dependencias nuevas), mismas defensas (sin redirecciones, tope de respuesta, validación de ids, cancelación por `threading.Event`, `client.py:36-43`, `:103-124`). La espera del resultado deja de ser sondeo (`client.py:107-115`) en modo incremental.

### 2.4 Cambios en el frontend

- `use-session-report.ts` hoy sondea cada 2 s mientras `status == "pending"`, hasta 15 min (`app:frontend/src/contexts/feedback/lib/use-session-report.ts:17-20`, `:38-41`). Con la bandera del backend activa, `ReportPage` abre `GET /api/v1/sessions/{id}/mrcd-analysis/events` con `fetch` + `ReadableStream` (no `EventSource`, porque la API usa `Authorization: Bearer`, `app:frontend/src/shared/api/client.ts:71`), parsea SSE, actualiza la caché de React Query al recibir `completed`/`unavailable` y cierra el flujo. Sin `refetchInterval` en ese modo.
- Reconexión: si el flujo se corta, una sola lectura `GET /sessions/{id}` (por si terminó mientras tanto) y reapertura con `Last-Event-ID`, con espera exponencial acotada; el servidor siempre emite primero el estado actual, así que no hay carrera "terminó antes de suscribirse".
- `SessionDisfluencySection` agrega un indicador de progreso neutral ("Analizando audio: n de m fragmentos"; "Calculando rasgos del audio completo") sin porcentajes de disfluencia, ni cifras que parezcan resultados.
- Con la bandera apagada, el comportamiento actual (sondeo) no cambia.

### 2.5 Diagramas de secuencia

**B original (vigente)**

```mermaid
sequenceDiagram
    participant FE as Navegador
    participant BE as Backend app
    participant MR as MRCD service
    FE->>BE: /ws/session tramas PCM 160 ms (seq, t0_ms)
    BE->>BE: SessionAudioBuffer.append (solo memoria)
    FE->>BE: close_session
    BE->>BE: to_wav_bytes() y sha256
    BE->>MR: POST /v1/jobs (WAV completo)
    MR-->>BE: 202 job queued
    loop sondeo cada 1 s
        BE->>MR: GET /v1/jobs/{id}
    end
    MR->>MR: ASR de todos los fragmentos + rasgos + V0 + reporte
    MR-->>BE: completed + documento v1
    BE->>BE: verifica sha256, sanea y guarda JSON; discard()
    loop sondeo cada 2 s (hasta 15 min)
        FE->>BE: GET /api/v1/sessions/{id}
    end
```

**B incremental (propuesta)**

```mermaid
sequenceDiagram
    participant FE as Navegador
    participant BE as Backend app
    participant MR as MRCD service
    participant W as Hilo trabajador MRCD
    FE->>BE: /ws/session tramas PCM 160 ms
    BE->>MR: POST /v1/sessions (perfil, opciones)
    MR-->>BE: 201 session_id, plan 8 s / 2 s
    loop cada lote de 2 s
        BE->>BE: SessionAudioBuffer.append (fuente de verdad)
        BE->>MR: PUT /v1/sessions/{sid}/audio/{seq} (offset, sha256)
        MR->>MR: ensambla por offset (idempotente por seq)
        MR->>W: fragmento k completo -> transcribe(seg)
        W-->>MR: palabras del fragmento k -> ChunkCache
        MR-->>BE: SSE asr_progress
    end
    FE->>BE: close_session
    BE->>MR: POST /v1/sessions/{sid}/close (total_samples, wav_sha256, last_seq)
    MR->>W: cierre (prioridad 0)
    W->>W: analyze_bytes(WAV canónico) con CachedASR: aciertos de caché + parciales finales + rasgos + V0 + reporte
    W-->>MR: documento v1 (idéntico al offline)
    MR-->>BE: SSE completed (solo estado)
    BE->>MR: GET /v1/sessions/{sid} (documento)
    BE->>BE: verifica sha256, sanea y guarda; discard() de ambos lados
    BE-->>FE: SSE completed (sin sondeo)
    Note over BE,MR: Ante cualquier fallo: DELETE sesión MRCD y POST /v1/jobs con el WAV del búfer (fallback offline)
```

### 2.6 Precarga de Whisper y calentamiento

- **Dónde**: `mrcd/service/cli.py` en `serve`, antes de `serve(...)` (`cli.py:109-126`). Hoy el motor se carga perezosamente en la primera petición (`analyzer.py:646-679`).
- **Cómo**: banderas `--preload p0[,p1_disfluent_prompt]` (vacío por defecto = comportamiento actual) y `--warmup`. La precarga llama a `MrcdAnalyzer._engine(AnalysisOptions(asr_profile=...))` con los MISMOS argumentos que la carga perezosa. El calentamiento ejecuta, en el hilo trabajador y antes de abrir el puerto, `analyze_array` sobre una señal sintética fija de 13 s generada en código (sin datos de personas) y descarta el resultado; `/v1/health` agrega `engines_loaded` y `warmed_up`.
- **Sin cambiar parámetros**: no se toca `WhisperModel(...)`, ni `decode_kwargs`, ni `cpu_threads`. Es la misma llamada que haría la primera petición.
- **Orden obligatorio** (por el hallazgo de hilos de 1.A.1): (1) `--device cpu`; (2) precargar todos los perfiles que el servicio aceptará (en modo incremental, una petición con un perfil no precargado se rechaza con `profile_not_supported_incremental` en lugar de cargarlo a mitad de la vida del proceso); (3) calentamiento, que importa Silero. Así todo análisis del proceso, offline o incremental, ve el mismo `torch.get_num_threads()`, igual al estado estable del servicio B original con un solo perfil (1). Esto fija estado del proceso, no parámetros de Whisper.
- **Expectativa honesta**: 3b midió que el arranque en frío cuesta 1,5 s (carga de modelos) y que la primera pasada no fue más lenta que las siguientes (`COST_REPORT.md:17`). El beneficio esperado de la precarga es ~1,5 s en la primera sesión; el del calentamiento puede ser ~0. El benchmark lo mide; si no aporta, se documenta y se deja apagado.
- **Prueba obligatoria**: el documento de una grabación es idéntico con y sin calentamiento previo (caso 21).

### 2.7 Contrato (rev 1.3, aditiva)

`mrcd.analysis/v1` **no cambia** (mismo esquema, mismo documento). `CONTRACT_REVISION` pasa de `1.2` a `1.3` (`mrcd/service/contract.py:45`) solo por los endpoints nuevos y el esquema nuevo `mrcd.session/v1`; los clientes de 1.2 no se ven afectados. Excepción declarada: `provenance.service.contract_revision` vale `"1.3"` en ambos modos una vez desplegado; la equivalencia se compara siempre entre modos de la MISMA versión.

| Método y ruta                        | Cuerpo / parámetros                                              | Respuesta                                                                                                                                                                                                                 |
| ------------------------------------ | ---------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `POST /v1/sessions`                  | query igual que `/v1/jobs` (`server.py:119-148`)                 | `201 {schema_version:"mrcd.session/v1", session_id, status:"open", plan:{chunk_s:8, overlap_s:2, sample_rate_hz:16000, sample_format:"pcm_s16le", channels:1}, limits:{max_duration_s, idle_timeout_s, max_batch_bytes}}` |
| `PUT /v1/sessions/{sid}/audio/{seq}` | PCM16 LE crudo; cabeceras `X-Offset-Samples`, `X-Content-SHA256` | `200 {received_samples, contiguous_samples, chunks_decoded, chunks_queued, duplicate:false}`; reenvío idéntico: `200 {..., duplicate:true}`                                                                               |
| `POST /v1/sessions/{sid}/close`      | JSON `{total_samples, wav_sha256, last_seq}`                     | `202 {status:"closing"}`; reenvío idéntico: misma respuesta o el resultado si ya terminó                                                                                                                                  |
| `GET /v1/sessions/{sid}`             | --                                                               | estado: muestras, lotes recibidos, `missing_seqs`, fragmentos decodificados, `status`, `result` si terminó                                                                                                                |
| `GET /v1/sessions/{sid}/events`      | `Last-Event-ID` opcional                                         | SSE: `state`, `asr_progress {chunks_decoded, chunks_complete, received_s}`, `finalizing {stage}`, `completed {status, result_available}`, `failed {status, result_available}` (sin documento; el documento sale de `GET /v1/sessions/{sid}`), `expired`, `cancelled`                                                          |
| `DELETE /v1/sessions/{sid}`          | --                                                               | `204`; descarta audio y caché; idempotente                                                                                                                                                                                |

Códigos de error nuevos (se agregan a `ERROR_CODES`, `contract.py`): `incremental_disabled` (404), `session_not_found` (404, no reintentable: usar fallback), `session_limit` (429, reintentable), `sequence_conflict` (409: mismo `seq` con otro sha u offset), `offset_mismatch` (409: lote que pisa bytes distintos), `batch_too_large` (413), `audio_gap` (409 al cerrar: `details.missing_seqs`, `details.first_missing_offset`), `close_mismatch` (409: `total_samples` o `wav_sha256` no coinciden con lo recibido), `session_closed` (409: audio después del cierre), `session_expired` (410), `profile_not_supported_incremental` (422, p. ej. `*_cuda`). Los errores siguen la regla de `ServiceError`: sin contenido de audio ni datos personales (`audio_io.py:20-25`).

### 2.8 Privacidad: qué cambia respecto a B

| Aspecto                      | B original                                                   | B incremental                                                                                                                                                                   |
| ---------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Dónde vive el audio          | Backend durante la sesión; MRCD solo durante el trabajo      | Backend durante la sesión **y MRCD durante toda la sesión** (dos copias en memoria, en dos procesos)                                                                            |
| Tiempo de exposición en MRCD | Duración del análisis (≈ 0,5 × duración)                     | Duración de la sesión + cierre                                                                                                                                                  |
| Disco / registros            | Nunca                                                        | Nunca (la caché de fragmentos es en memoria; contiene palabras, que son dato personal, y se descarta con la sesión)                                                             |
| Transcripción                | No sale de MRCD si `include_text=false`                      | Igual; los eventos SSE **no** llevan texto ni palabras, solo conteos                                                                                                            |
| Descarte                     | `discard()` en todos los caminos                             | Igual en la app; en MRCD, `discard()` de PCM y caché al terminar, cancelar, expirar, fallar y al apagar; la sesión expira por inactividad (`idle_timeout_s`, 120 s por defecto) |
| Límites                      | Tope 600 s, topes de capacidad de la app (`README_S7.md:93`) | Los mismos, más `max_sessions` y `max_session_bytes` en MRCD; superar un tope en MRCD => fallback offline o `unavailable`, nunca resultado parcial                              |
| Consentimiento               | Requerido (`PRIVACY_DECISION.md` §7.2)                       | Igual; el texto de consentimiento debe mencionar que el audio llega al servicio de análisis **durante** la sesión. Requiere ajustar el borrador y la ratificación pendiente     |

Riesgo nuevo: la ventana en la que un volcado de memoria de MRCD expondría audio pasa de "duración del análisis" a "duración de la sesión". Debe registrarse como addendum de `PRIVACY_DECISION.md` y ratificarse junto con la opción B.

### 2.9 Fallos y recuperación

| Situación                                                    | Comportamiento                                                                                                                                                                                    | Garantía                         |
| ------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------- |
| Lote duplicado (mismo `seq`, mismo offset y sha)             | `200 duplicate:true`, no se escribe nada                                                                                                                                                          | Sin duplicar                     |
| Mismo `seq` con otro contenido                               | `409 sequence_conflict`; la app cancela y usa fallback                                                                                                                                            | Sin mezclar audio                |
| Lote fuera de orden                                          | Se guarda por offset (ventana de reordenación acotada, p. ej. 32 lotes); solo se decodifica el prefijo contiguo                                                                                   | Sin perder ni reordenar          |
| Hueco al cerrar                                              | `409 audio_gap` con `missing_seqs`; la app reenvía esos lotes desde su búfer (lo tiene completo) y reintenta el cierre una vez; si persiste, fallback offline                                     | Sin perder                       |
| Fallo de red en `PUT`                                        | Reintento con espera exponencial (3 intentos) del mismo `(seq, offset, sha)`; idempotente                                                                                                         | Sin duplicar                     |
| MRCD reiniciado                                              | Siguiente `PUT` o `close` responde `session_not_found`; la app marca el uploader degradado y al cerrar usa fallback offline con el búfer completo                                                 | Resultado = offline              |
| Reinicio del backend de la app                               | El búfer en memoria se pierde (igual que hoy); la sesión MRCD expira por inactividad y se descarta; el documento `pending` huérfano pasa a `analysis_interrupted` (`app:.../analysis.py:294-328`) | Sin audio huérfano               |
| Cancelación del usuario / retiro de consentimiento / apagado | `DELETE` en MRCD + `discard()` en la app                                                                                                                                                          | Descarte en ambos lados          |
| Fallo de un fragmento en MRCD (excepción de decodificación)  | El fragmento queda sin caché; al cierre `CachedASR` lo decodifica (fallo de caché) o el cierre falla con `analysis_failed`                                                                        | Nunca palabras parciales         |
| Sesión del motor saturada (cola de fragmentos atrasada)      | Los fragmentos se decodifican más tarde, como mucho al cierre                                                                                                                                     | Exactitud intacta; solo latencia |
| SSE cortado                                                  | Reconexión con `Last-Event-ID`; el primer evento es el estado actual                                                                                                                              | Sin perder la finalización       |

### 2.10 Configuración

| Repo        | Bandera                                                                               | Predeterminado                                                                                                              |
| ----------- | ------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| mrcd        | `serve --incremental` (habilita `/v1/sessions`)                                       | apagado                                                                                                                     |
| mrcd        | `serve --preload <perfiles>`, `serve --warmup`                                        | vacío / apagado (con `--incremental` son obligatorios, junto con `--device cpu`; el servicio se niega a arrancar sin ellos) |
| mrcd        | `--max-sessions` (4), `--session-idle-timeout-s` (120), `--max-session-seconds` (600) | indicados                                                                                                                   |
| app backend | `ORATORIA_MRCD_MODE=offline\|incremental\|incremental_verify`                         | `offline`                                                                                                                   |
| app backend | `ORATORIA_MRCD_INCREMENTAL_BATCH_MS`                                                  | 2000                                                                                                                        |
| app backend | `ORATORIA_MRCD_PROGRESS_PUSH` (SSE al frontend)                                       | `false`                                                                                                                     |

Cambiar el predeterminado a `incremental` exige los criterios de la sección 5. Los comandos de arranque se agregan a `comandos-ejecutar/COMANDOS.MD` (fuente única), no aquí.

---

## 3. Casos de prueba de equivalencia exacta

**Comparación bit a bit.** "Igual" significa `==` de Python sobre: (a) lista de palabras como tuplas `(text, norm, repr(start), repr(end), repr(prob), punct_after)`; (b) el documento JSON completo `mrcd.analysis/v1` serializado con `json.dumps(..., sort_keys=True)`, incluidos `events[*]` (`class`, `start_s`, `end_s`, `score.value`, `n_windows`, `validation`, `decision`), `summary`, `transcript` (con `include_text=true` en pruebas), `head_decision`, `provenance` (incluido `asr.profile_fingerprint`, `environment`), `warnings` y `request.audio_sha256`; (c) cuando la prueba usa `keep_windows`, las ventanas con sus probabilidades.

**Únicos campos excluidos (no deterministas, lista cerrada):**

1. `generated_at` del documento `mrcd.analysis/v1` (`analyzer.py:495`, `:131`).
2. Envoltorios de transporte: `job_id`, `created_at` (`server.py:54-60`), `session_id` y las marcas de tiempo de eventos SSE.
3. En la app: `updated_at` (`app:.../analysis.py:109`) y el campo nuevo `binding.mode` (`offline`/`incremental`).
4. Métricas de ejecución (tiempos, aciertos de caché), que nunca forman parte del documento.

Cualquier otra diferencia es un fallo.

**Grabaciones.** Del repo: `results/mejora_objetivos_20261101/S7_app/inputs/e2e_synthetic_es.wav` (45,466 s, sintética), `results/cierre_oe1_oe3_20261031/3b/cost/inputs/*.wav` (13,5-60 s), `results/baselines_20261030/ameresco/inputs/*.wav` (13-17 s). Larga: una grabación de > 10 min no puede pasar por el servicio (tope 600 s, `analyzer.py:48`); se prueba con un tramo de 599,9 s y, a nivel de `DisfluencyEngine`, con un archivo de corpus local de > 10 min (solo en `MRCD_SLOW=1`, nunca en CI). Sintéticas: cortes exactos de una grabación (tabla), más ruido/tono generados en código con semilla.

| #   | Entrada                                                                          | Procedimiento                                                                                                                | Criterio de aceptación                                                                                                                                                                                             |
| --- | -------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 1   | 13,0 s (mínimo)                                                                  | offline vs incremental, bloques de 160 ms                                                                                    | (a) y (b) iguales                                                                                                                                                                                                  |
| 2   | 13,6 s (AMERESCO)                                                                | idem                                                                                                                         | iguales; 1 fragmento completo + 2 parciales                                                                                                                                                                        |
| 3   | 14,0 s = 6+8 (múltiplo exacto de la rejilla)                                     | idem                                                                                                                         | iguales; ningún parcial "de más"                                                                                                                                                                                   |
| 4   | 30 s y 48 s (múltiplos de 6 y de 8)                                              | idem                                                                                                                         | iguales                                                                                                                                                                                                            |
| 5   | 31,0 s y 32,0 s (6k+7 y 6k+8)                                                    | idem                                                                                                                         | iguales                                                                                                                                                                                                            |
| 6   | 37,0 s (último parcial = exactamente 1 s) y 36,999 s (último < 1 s, se descarta) | idem                                                                                                                         | iguales; número de fragmentos según `linguistic.py:280-281`                                                                                                                                                        |
| 7   | 45,466 s (no múltiplo, sintética)                                                | idem                                                                                                                         | iguales                                                                                                                                                                                                            |
| 8   | 60,0 s y 599,9 s                                                                 | idem                                                                                                                         | iguales                                                                                                                                                                                                            |
| 9   | > 10 min (motor, `MRCD_SLOW=1`)                                                  | `CachedASR` vs `transcribe_chunked` directo                                                                                  | (a) igual; `_evaluate` igual                                                                                                                                                                                       |
| 10  | 7,5 s y 8,0 s (motor, sin servicio)                                              | `CachedASR` sobre `transcribe_chunked`                                                                                       | (a) igual; se verifica el atajo `<= 8 s` (`linguistic.py:272-273`)                                                                                                                                                 |
| 11  | Particiones de envío                                                             | misma grabación en bloques de 160 ms, 2 s, tamaños aleatorios con semilla fija (1 muestra a 40 000 muestras), un solo bloque | (b) idéntico en las cuatro y al offline                                                                                                                                                                            |
| 12  | Envío desordenado                                                                | lotes permutados con semilla dentro de la ventana de reordenación                                                            | (b) igual; `decode_calls` = número de fragmentos (sin decodificar dos veces)                                                                                                                                       |
| 13  | Reenvío duplicado                                                                | cada lote enviado 2 veces; también `close` duplicado                                                                         | (b) igual; respuestas `duplicate:true`; `decode_calls` sin cambio                                                                                                                                                  |
| 14  | Conflicto                                                                        | mismo `seq` con bytes distintos                                                                                              | `409 sequence_conflict`; la app usa fallback; (b) final = offline                                                                                                                                                  |
| 15  | Hueco                                                                            | se omite un lote; al cerrar `409 audio_gap`; la app reenvía                                                                  | (b) igual; y si no reenvía: fallback = offline                                                                                                                                                                     |
| 16  | Fallo y reanudación a mitad                                                      | se corta la red (cliente falso) entre lotes 10-20 con reintentos                                                             | (b) igual; ningún lote duplicado en el ensamblado (sha del PCM final = sha del búfer de la app)                                                                                                                    |
| 17  | Reinicio del servicio                                                            | se mata el proceso MRCD a mitad de sesión                                                                                    | la app termina por fallback; (b) = offline; ningún audio en disco                                                                                                                                                  |
| 18  | Cancelación                                                                      | `DELETE` a mitad; consentimiento retirado                                                                                    | `204`; memoria de la sesión en cero (prueba con `bytearray` espía, como B13 de S7); `close` posterior `410/404`                                                                                                    |
| 19  | Perfil p0 y p1                                                                   | casos 1-8 con `asr_profile=p0` y `p1_disfluent_prompt`                                                                       | iguales en ambos                                                                                                                                                                                                   |
| 20  | `p0_noloop` / `p1_noloop`                                                        | casos 2 y 7                                                                                                                  | iguales si `loop_redecode.n_non_reproducible == 0` (incluidos `events[*].chunk_offset_s` y su orden); si es > 0 se reporta "no comparable" y se repite offline-vs-offline para mostrar que tampoco es reproducible |
| 21  | Calentamiento                                                                    | mismo archivo con y sin `--warmup`, con y sin precarga                                                                       | (b) igual                                                                                                                                                                                                          |
| 22  | Interferencia entre sesiones                                                     | 3 sesiones simultáneas intercaladas + un `POST /v1/jobs` en medio                                                            | (b) de cada una igual a su offline                                                                                                                                                                                 |
| 23  | Opciones del documento                                                           | `include_text`, `include_out_of_scope`, `decision`, `head=vc`, `text_route=rep_ctx_p1`                                       | (b) igual en cada combinación (las opciones solo se aplican en `_run`/`build_analysis`)                                                                                                                            |
| 24  | Errores equivalentes                                                             | 9 s (`audio_too_short`), > 600 s (`audio_too_long`), estéreo no aplica (PCM mono)                                            | mismo `error.code` que el offline                                                                                                                                                                                  |
| 25  | Clave de caché                                                                   | se fuerza un `start` desplazado 1 muestra en el planificador incremental (mutante)                                           | (b) sigue igual (fallo de caché, decodificación al cierre) y `misses > 0`                                                                                                                                          |
| 26  | Mutante de fusión                                                                | el envoltorio devuelve palabras de otro fragmento                                                                            | la prueba 7 falla (detecta el mutante)                                                                                                                                                                             |
| 27  | Estado de hilos del proceso                                                      | servicio arrancado con precarga p0+p1 y calentamiento; offline y luego incremental de p1 en el mismo proceso, y al revés     | (b) idéntico, incluido `provenance.environment.cpu_threads.torch`; además, prueba que demuestra la diferencia si se carga un motor DESPUÉS del primer análisis (reproduce el hallazgo de 1.A.1)                    |

**Pruebas de privacidad.**

| #   | Prueba                                                                                                                                     | Aceptación                                 |
| --- | ------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------ |
| P1  | Ningún archivo nuevo en cwd/tmp del servicio ni del backend tras 5 sesiones (`builtins.open`, `os.open`, `tempfile` vigilados, como en S7) | 0 escrituras                               |
| P2  | Registros de ambos procesos sin `RIFF`, `WAVE`, `pcm`, texto de palabras                                                                   | 0 coincidencias                            |
| P3  | Tras `completed`, `failed`, `cancelled`, `expired` y apagado: `bytearray` de la sesión en ceros y caché vacía                              | verificado con espía                       |
| P4  | Eventos SSE sin campos `text`, `words`, `transcript`                                                                                       | validación de esquema                      |
| P5  | Sin consentimiento: no se abre sesión MRCD                                                                                                 | 0 llamadas (contador del cliente falso)    |
| P6  | Expiración por inactividad                                                                                                                 | sesión descartada a `idle_timeout_s` + 1 s |

**Pruebas de mensajes de progreso.**

| #   | Prueba                                                                    | Aceptación                                                                                                                    |
| --- | ------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| M1  | Orden: `state`, `asr_progress*` no decreciente, `finalizing`, `completed` | orden estricto; `completed` exactamente una vez                                                                               |
| M2  | Suscripción tardía (después de `completed`)                               | primer evento = `completed` solo con estado y conteos (el documento sale de `GET`)                                                                                  |
| M3  | Reconexión con `Last-Event-ID`                                            | sin eventos perdidos ni repetidos después de ese id                                                                           |
| M4  | Frontend                                                                  | con la bandera activa no hay `refetchInterval`; con la bandera apagada el sondeo actual no cambia (pruebas vitest existentes) |
| M5  | Fallback                                                                  | si el SSE se cae y la app usa offline, el frontend recibe igualmente `completed` por su propio SSE                            |

---

## 4. Plan de benchmark

- **Condiciones**: misma máquina, inactiva (como 3b, `COST_REPORT.md:3-4`), `OMP_NUM_THREADS=4` antes de numpy, `HF_HUB_OFFLINE=1`, CPU int8, perfiles p0 y p1, sin GPU. Una ronda fría descartada + 3 medidas; mediana (mín-máx).
- **Entradas**: las mismas del caso de prueba (13 a 600 s), reproducidas en tiempo real (reloj de pared, lotes de 2 s) para simular la sesión; y una variante "lo más rápido posible" para separar el costo del ritmo.
- **Métricas por modo**: (1) **latencia post-sesión** = desde `close_session` hasta `completed` en el backend; (2) tiempo total de CPU del proceso MRCD (`psutil` `cpu_times`) y tiempo total de pared; (3) pico de memoria (working set) de MRCD y del backend; (4) carga de CPU durante la sesión (incremental la reparte); (5) tiempo de arranque con y sin `--preload` y `--warmup` y latencia de la primera sesión; (6) equivalencia (b) en cada corrida; (7) con 1, 2 y 3 sesiones simultáneas: retraso de la cola de fragmentos al cerrar.
- **Salida**: `results/<fecha>_b_incremental/BENCHMARK.md` + `runs.json`, mismo formato que `3b/cost`. Sin cifras en este documento hasta medir.

## 5. Criterios de aceptación y riesgos

**Para pasar `incremental` a predeterminado (todos):**

1. Casos 1-27 en verde en CI (los que no requieren modelo, con ASR falso) y con Whisper real en `MRCD_SLOW=1`; 0 diferencias fuera de la lista cerrada.
2. Pruebas P1-P6 y M1-M5 en verde.
3. `incremental_verify` en el E2E real (Chromium + micrófono falso, `scripts/mejora_s7/`) con `equal: true` en todas las sesiones.
4. Benchmark: latencia post-sesión menor que offline en todas las duraciones ≥ 30 s, sin aumento del pico de memoria de MRCD mayor que el audio de las sesiones abiertas.
5. Addendum de privacidad registrado y ratificación del asesor (incluye el cambio de exposición de 2.8).

**Riesgos:**

| Riesgo                                                                                                                                                           | Mitigación                                                                                                                                  |
| ---------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| Equivalencia exacta imposible para `*_noloop` cuando se usa el peldaño muestreado                                                                                | Declarado; no forma parte de los predeterminados; caso 20 lo hace visible                                                                   |
| Una actualización de faster-whisper/ctranslate2 introduce estado entre llamadas                                                                                  | La clave de caché incluye versiones y `model.bin`; `incremental_verify` lo detectaría; casos 1-8 en CI con Whisper real antes de actualizar |
| `OMP_NUM_THREADS` distinto en el proceso que reabre tras reinicio                                                                                                | El reinicio no reutiliza caché (está en memoria): siempre fallback offline                                                                  |
| Saturación con varias sesiones (ASR ≈ 0,45 de un proceso de 4 hilos por segundo de audio, `COST_REPORT.md:14-16`)                                                | Cola con prioridad; la exactitud no depende del retraso; `max_sessions`; el benchmark mide la latencia con 2-3 sesiones                     |
| Mayor exposición del audio en MRCD                                                                                                                               | 2.8; consentimiento y ratificación                                                                                                          |
| La app y MRCD discrepan en la cabecera WAV                                                                                                                       | `close_mismatch` si el sha no coincide; caso 11 y verificación de enlace existente                                                          |
| El calentamiento no reduce latencia                                                                                                                              | Se mide; si no aporta, queda apagado                                                                                                        |
| Dos copias en memoria (app y MRCD)                                                                                                                               | Topes en ambos lados; el benchmark mide el pico                                                                                             |
| Hilos de torch dependientes de la historia del proceso (medido: 4 frente a 1 en `provenance`; también afecta al offline B original si se cargan varios perfiles) | Arranque determinista (2.6); caso 27; evaluar aparte si conviene corregirlo también en B original (fuera del alcance de este diseño)        |
| Carga de un segundo motor con `device="auto"` detenida en esta máquina (observado una vez, no diagnosticado)                                                     | `--device cpu` obligatorio en modo incremental; investigar antes de usar `--preload` con varios perfiles en `auto`                          |
| La equivalencia de puntuaciones con distinto número de hilos de torch se observó (p1, 2 archivos) pero no está garantizada                                       | No depender de ello: se fija el estado del proceso                                                                                          |

Si en algún momento una de las verificaciones de equivalencia diera diferencias para p0/p1, la alternativa más cercana (no implementada) sería: mantener el incremental solo para mostrar progreso y adelantar el ASR, pero producir el documento final SIEMPRE con el offline sobre el WAV completo (equivale a `incremental_verify` sin usar el resultado incremental), aceptando la latencia del offline.

## 6. Plan de implementación (pasos pequeños, un commit por paso)

Cada paso deja el sistema funcionando con el offline como predeterminado. Commits convencionales, sin atribuciones.

**Repo `mrcd`:**

1. `test(asr): pin per-chunk decode independence and merge equivalence with real Whisper (slow)` - casos 1-10 a nivel de motor con `CachedASR` aún en el test.
2. `feat(service): CachedASR wrapper and per-chunk cache key (no behaviour change)` - `incremental/cached_asr.py`, `cache.py`, `plan.py` + pruebas unitarias con ASR falso, mutantes 25-26.
3. `feat(service): engine worker with a priority queue` - `JobRunner` -> `EngineWorker`; `/v1/jobs` sin cambios (pruebas existentes en verde).
4. `feat(service): incremental session store with idempotent appends` - ensamblado por offset, duplicados, conflictos, huecos, expiración, `discard()`; sin endpoints.
5. `feat(service): /v1/sessions endpoints behind --incremental (contract rev 1.3)` - endpoints, códigos de error, esquema `mrcd.session/v1`, `docs/contracts/` actualizados.
6. `feat(service): SSE progress stream for incremental sessions` - M1-M3.
7. `feat(service): optional preload and warm-up at serve start` - `--preload`, `--warmup`, orden determinista (device cpu -> precarga -> calentamiento), casos 21 y 27.
8. `test(service): exact equivalence suite incremental vs offline` - casos 11-24, P1-P6.
9. `docs(contracts): B incremental variant, privacy addendum and commands` - este documento pasa a "implementado", `PRIVACY_DECISION.md` addendum, `COMANDOS.MD`.

**Repo app (`feat/mrcd-integration`, por separado):**

10. `feat(backend): MRCD incremental client (sessions, append, close, cancel, SSE)` - stdlib, mismas defensas.
11. `feat(backend): incremental uploader beside the in-memory buffer (opt-in)` - `ORATORIA_MRCD_MODE`, cancelación en hueco/límite, fallback offline.
12. `feat(backend): incremental_verify mode and progress hub` - comparación sin contenido.
13. `feat(backend): SSE endpoint for the session analysis status`.
14. `feat(report): subscribe to analysis progress instead of polling (flagged)` - frontend, M4-M5.
15. `test(e2e): incremental vs offline on the real stack` - Chromium + micrófono falso, `incremental_verify`.

**Evidencia y cierre:**

16. Benchmark (sección 4) y reporte en `results/`.
17. Revisión independiente del diff de ambos repos antes de cambiar cualquier predeterminado.
