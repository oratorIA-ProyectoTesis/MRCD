# OE2_RESUMEN: archivos creados, hashes y discrepancias abiertas (2026-10-31)

Alcance: fase 2 del plan (2.1 contratos, 2.2 criterio de desambiguación, 2.3 C4, 2.4 captura de la aplicación, 2.5 paquete de aprobación). Escribió solo en `mrcd\docs\architecture\`, `mrcd\docs\contracts\` y esta carpeta. No tocó `mrcd\service\`, `mrcd\docs\microphone-check\`, documentos de tesis OE1 ni la aplicación (lectura únicamente). Sin push. Idioma: documentos técnicos en inglés; portada y resumen del paquete de aprobación en español neutro.

Hash = SHA-256 del contenido con LF (los PNG, en bruto). Estado de aprobación: **PENDIENTE** (ver the advisor approval sheet (kept local, not published)); el agente no simuló ninguna aprobación.

## Commits locales (no subidos)

| Commit      | Contenido                                                                                                                    |
| ----------- | ---------------------------------------------------------------------------------------------------------------------------- |
| `bff4c2a`   | `docs(architecture)`: modelo C4, renderizador SVG determinista, verificador de código                                        |
| `7e9f8e8`   | `docs(contracts)`: contratos actuales, especificación de desambiguación, contrato de integración, esquemas y ejemplos reales |
| (siguiente) | esta carpeta: the advisor approval sheet (kept local, not published) y este resumen (forzado con `git add -f` porque `results/` está en `.gitignore`)    |

## Archivos creados (`mrcd\`)

### docs/architecture

| Archivo                                 | SHA-256                                                                                                                                                                                                      |
| --------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `README.md`                             | `6c10ab46a74a21a25f82c329ceea8665a2ddedc6397b76c4ec9eeece3846877c`                                                                                                                                           |
| `c4_model.py`                           | `bb164fbb6aee722177f41d69b582074a79cfacdcd94f96680378ff219c8a2c41`                                                                                                                                           |
| `verify_architecture.py`                | `5d256126aaa84977d1e4913a141f4798bf24f7af88183cabbfb4c09adf9107b8`                                                                                                                                           |
| `render_png.sh`                         | `3c91c53cdd34c0421a4c43c5f30f87b00cf9d3c23f2f8df0c1c3dcfa863f9b72`                                                                                                                                           |
| `c4_1_context.svg` / `.mmd` / `.png`    | `1f71b52d6db3880dd2e6a5f1a251b28d1495ab061bcee371d8ca5c859dba8e95` / `4297c3d08fc51fce8fe4805c89a6a5567dc496c71bd949f0cb14badedd41879d` / `6a540cde3e99c767a6c76dbb6488149ac1a582fb4847addca42b3b51cfa1cc9d` |
| `c4_2_containers.svg` / `.mmd` / `.png` | `dc77327d226f9741b3ce2e8dff9a74120e069c7fff681a0ac957aa58527becc2` / `67fa74352d006d3112a02353a806d6e3f5d4348d61d8ff595f005b8d73af4c65` / `aab5c54e546dfbcf428253ff55b36da0efb97a7841e0fdb5c06d85954d98c208` |
| `c4_3_components.svg` / `.mmd` / `.png` | `c7dcd4f658515454f00f2f9876c4ea51325f7cd80ed8ffb08dc645c44a042b2e` / `34cfaaaffb1529579c7eb0fd1d7c1e2277dce62aa11b241ee5a1885b9996603c` / `ad930bddffafe69cc5016ff070f095db411b948d28a696498f24cb12ddfcf501` |
| `c4_4_code.svg` / `.mmd` / `.png`       | `9f39a9d1598588f3de1e7754c9d5078606ad54efb0a7f6ec35e115a062bbaac8` / `4a0384c20ed32e9aa55f7116b5ade087aa80ecccc8589b368ce38f08864e9a33` / `002e963bfec877c2ca5bd7387ac8ce19f0edd7fe1d1f6d1ba783ec2c255542bf` |

### docs/contracts

| Archivo                                      | SHA-256                                                            |
| -------------------------------------------- | ------------------------------------------------------------------ |
| `README.md`                                  | `633a9465beb0ad62dfb24cbaad655981dfb1baf55643c95093f080790352de99` |
| `EVENT_CONTRACT.md`                          | `ba2c2328793da6a7720c6ff8976318b868de47802db61c528a2f6f5e3e8620b7` |
| `LIVE_CONTRACT.md`                           | `223b0accbcc56c8818970aba0335f6ff27c52618b4d7c21e5fe825e75471c2ce` |
| `ASR_PROFILES_CONTRACT.md`                   | `435941b6c77effc095af550e4ca7b920c38376cf78182b6f8603e4e65693cae6` |
| `HEAD_CONTRACT.md`                           | `68ec4c790b085728354a73f5fd53aeb70a0674161a1ece6fe0fc7bc558add13c` |
| `DISAMBIGUATION_SPEC.md`                     | `a3b064691f99c402280754dd67470fe7bac3aa563b4a45c52bbe121f8a196b6c` |
| `APP_INTEGRATION_CONTRACT.md`                | `ccef3ec877e895b30f56d39735a25f61f83cfbcf8620201d7d4e77dfe1769e97` |
| `disambiguation_states.mmd`                  | `4a55f622653934030e44646f6b2fb861260c49b1b6c8c91746e2d962c9749465` |
| `validate_examples.py`                       | `f09f76c10aea08a719f9e0a3705156ed8ae1a26b8ab7e6422447bd1325e91f2a` |
| `build_report_section.py`                    | `ae695c25815be95f3fef0f14a74025342a08c007f0d8b9dae54808ea02c7d818` |
| `schemas/engine_result.current.schema.json`  | `0430ea01d8506523985528c3879bbe817d37fd65ec30172284387d406ec109bc` |
| `schemas/engine_windows.current.schema.json` | `c4b8822a7bf9493b9920dd8f653d542fe0c47e5927e16b4e64185400a37cb70b` |
| `schemas/live_row.current.schema.json`       | `e5931a802543107f34e6d92419a5c45da89d2b9dd6012a4589dc845ca1f9b655` |
| `schemas/app_integration.v1.schema.json`     | `05e5a53006ff00118b5cb9294faf65776382b2257c0551cf5cc5fc2677eaa9aa` |

### docs/contracts/examples (salidas reales del motor y del servicio; comando exacto en `provenance.json` y en `generate_examples.py`)

| Archivo                                  | SHA-256                                                                                                                   |
| ---------------------------------------- | ------------------------------------------------------------------------------------------------------------------------- |
| `generate_examples.py`                   | `4b59b61b29f42bf99fc62d747884841054e005f59bbb4582a37b6c785435902e`                                                        |
| `provenance.json`                        | `32412d74261831efc789f3261cd2cc7fa948e652bdfa08f91c0369eb90d12bb6`                                                        |
| `batch_p0.json`                          | `7f3e5ef96dba089811b775cddce0e60c68b7a50d2fdef868140896df4069346b`                                                        |
| `batch_p0_vc.json`                       | `0384cea0298e3fdd85b6b6ee6bc143ef8e8c2676fbd87350cf6c91923a0bc643`                                                        |
| `batch_p0_windows.json`                  | `53a277efbfeac5f6dd2088602c9e4e9194c0b43a9c6592ef57979b3345a3f161`                                                        |
| `batch_p1.json`                          | `f94bd37a4427c89727aebb452e889960a894584f63c024eff3d8d6a75244996c`                                                        |
| `error_short_audio.json`                 | `c4bd857386ae0d2444b454716e44ea521ad10530eda46391f28be42c274ddbb6`                                                        |
| `live_rows_v1.json`                      | `1d6af956950d30c86cb202a63c7e44bc8175dfb51672f7fecb28e0544e97f2e6`                                                        |
| `live_rows_v2.json`                      | `b503d202fd96706425cc78d7bf7fb7d731093786e75d3d3fcdf68562573badac`                                                        |
| `live_rows_v3.json`                      | `1c88a3bf1a86303b7555fb7a7eee0a2b0909b796480f372037f13ebe2fafa95e`                                                        |
| `service_analysis_v1.json`               | `b980321d3aa21d27594e1fe13b3370e69904e24a0025449dbbef51680b749d92`                                                        |
| `service_error_audio_too_short.json`     | `f6938d4e5567820032a4e40c2e651a600fabb2791eafea8877dc977e779372b2`                                                        |
| `app_report_section_example.json`        | `9097a4c6bc9e9ef6a2c0b5a25a01a4b0038559099b87c59ddfc105593424eec6`                                                        |
| `app_report_section_failed_example.json` | `76fbcc0f4a8f047251add341a49008ab19b99af78fb2c5cf29c59cacd0294a1f`                                                        |
| `app_session_binding_example.json`       | `95e696a719b1d208dc65e4d0f2ffe5bb64c865928613399082158a85736fa373` (ilustrativo: identificadores de marcador de posición) |

Fragmento usado: segundos 60-80 de `data/clips/FIwdDFht7FE.wav` (git-ignorado, charla pública; audio no versionado), sha256 del fragmento `a19c7f534dfc29e440666b918b93aae89a560b95a7522bd56592f2ff4afa1763`. Checkpoint `87f95108a765bfac...`, Vc `776c72f8e064e42c...`, Whisper `small` (`model.bin` `3e305921506d8872...`), `OMP_NUM_THREADS=4`, `HF_HUB_OFFLINE=1`.

### results/cierre_oe1_oe3_20261031

the advisor approval sheet (kept local, not published) (hoja de aprobación, estado PENDIENTE) y este `OE2_RESUMEN.md` (su propio hash no se lista aquí; ver `git hash-object` o calcúlelo tras el último cambio).

## Verificación (resultado literal)

- `docs/architecture/verify_architecture.py`: **113 comprobaciones aprobadas, 0 fallidas** (rutas de módulos, símbolos por AST, evidencia de imports por arista, afirmaciones negativas, ausencia de cliente MRCD en la aplicación, cobertura de módulos).
- `docs/contracts/validate_examples.py`: **67 comprobaciones aprobadas, 0 fallidas** (esquemas actuales y de integración, contrato `mrcd.analysis/v1` del servicio, semántica de eventos/ventanas/filas en vivo, "no calibrado", "no disponible con motivo").
- `c4_model.py` re-ejecutado: salida byte a byte idéntica (determinista).
- No se ejecutó la suite completa de pruebas de MRCD en esta tarea (no se modificó código de `mrcd/`).

## Hallazgos principales (para el informe de cierre)

1. `DisfluencyEvent` y el estado "disfluente/funcional/incierto" no existen en el motor; existe `Event`/fila en vivo y `decision.three_state` solo para investigación.
2. Pausas y bloqueo se emiten sin referencia humana y dependen del relleno/duración; no se derivan clases de segundos en inferencia, pero las etiquetas de entrenamiento sí usaron umbrales en segundos preliminares.
3. Un 20 s real produjo 8 eventos; el intervalo de `filler_word` (0,0-2,0 s) cubre 10 palabras: ejemplo de franja de decisión frente a frontera humana. En ese fragmento `live_cfg_v2` y `live_cfg_v3` dieron filas idénticas (no dice nada sobre sesiones largas).
4. Existe una "cabeza" V0 (`net.head`, sin calibración) y una Vc experimental (sigmoide + umbrales congelados, `calibrated:false`); la Vc marcó las tres clases como presentes en una unidad de 20 s, consistente con su propia nota de saturación.
5. La aplicación captura PCM16 mono 16 kHz en bloques de 160 ms con `t0_ms` en ms de época del navegador, con cancelación de eco/supresión de ruido/AGC activos; el video no sale del navegador; no hay desfase A/V medido ni verificación de identidad.

## Discrepancias abiertas

1. Plan vs código: `DisfluencyEvent` y tres estados (ver arriba).
2. `mrcd/service/contract.py`: `score.kind = max_window_class_probability` para un softmax sin calibrar; recomendado `max_window_softmax_score` en la siguiente versión (no modificado; código de otra tarea).
3. NaN: `three_state` → incierto; `unit_decision` de Vc → `present=False` (inalcanzable vía `analyze`).
4. Sin pruebas dedicadas: cola < 1 s no transcrita en `transcribe_chunked`; ASR sin palabras; saturación de Vc en unidades largas; empate exacto del softmax.
5. `README.md` raíz de MRCD desactualizado (estructura `core/`, "27 pruebas", "sin API web").
6. `mrcd/service` apareció en el árbol (commit `955d27c`) mientras esta tarea se ejecutaba; los diagramas y el verificador se actualizaron a ese árbol. `tests/test_service_server.py` tenía una edición sin confirmar de esa tarea al cerrar.
7. Integración app → MRCD no existe; requiere un búfer temporal de audio (decisión de privacidad pendiente, pregunta P6 de la hoja) y manejo de huecos de chunks, que el servicio v1 no representa.
8. Los `.mmd` de Mermaid no se probaron con un renderizador (ninguno instalado: sin `mmdc`, `dot` ni PlantUML); el SVG es el artefacto renderizado.
9. Sin `HF_HUB_OFFLINE=1`, un segundo `DisfluencyEngine.load` en el mismo proceso se colgó más de 10 min sin uso de CPU; causa no investigada (los ejemplos se generaron con la variable).
10. Efecto de DSP del navegador y remuestreo lineal sobre las características acústicas: no medido. Tiempos de pared de `provenance.json`: una corrida con la máquina compartida; no son benchmark.
11. COMANDOS.MD no define la suite completa ni comandos del servicio/micrófono (lo señalan `ESTADO_INICIAL.md` y el plan de la fase 4); no se editó (fuera de `mrcd\`, sin VCS).

## Pendiente por terceros

- Asesor: decisión sobre OE2 (hoja lista, campos vacíos). Ningún estado fue afirmado como aprobado.
- Responsable del servicio: nombre del tipo de puntaje (pregunta P11).
- Propietarios de la aplicación: decisión de retención temporal de audio y cualquier cambio en el repositorio de la aplicación (autorización explícita requerida; este trabajo fue solo lectura).
