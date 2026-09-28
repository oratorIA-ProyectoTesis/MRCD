# MRCD — protocolo técnico, operación y estado

Implementación del _Plan técnico de MRCD: motor, revisión humana y producto de prueba_ (28-09-2026) sobre la rama `refactor/mrcd-architecture`. Este documento separa lo que ya está construido y verificado de lo que depende de personas, datos o decisiones del equipo.

## 1. Arquitectura

Monolito modular con un proceso de trabajo separado:

| Pieza          | Archivo              | Responsabilidad                                                                                                                                                                                                                                                        |
| -------------- | -------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Núcleo         | `core/engine.py`     | `PreparedRecording`: ASR y pista acústica una sola vez por medio y configuración; ventanas vectorizadas; decodificación con umbrales y abstención; refinamiento de fronteras opcional; tiempos por etapa, RTF, hardware; caché por hash de medio, parámetros y versión |
| Evaluación     | `core/evaluation.py` | Emparejamiento uno a uno, métricas por clase, pausas aparte, fronteras, tolerancia, acuerdo A/B, bootstrap por hablante, calibración, cobertura/riesgo, selección activa, SUS                                                                                          |
| API            | `app/api.py`         | Rutas del §13 del plan bajo `/api`, autorización por rol y asignación, cegamiento, concurrencia por revisión esperada                                                                                                                                                  |
| Worker         | `app/worker.py`      | Preparación de medios y ejecuciones; recuperación de trabajos abandonados con intentos limitados                                                                                                                                                                       |
| Sistemas       | `app/systems.py`     | MRCD, reglas (B0) y GPT audio (G), cada uno con estado propio por segmento                                                                                                                                                                                             |
| Medios         | `app/media.py`       | Original intacto, derivado PCM 16 kHz mono con comando registrado, picos a 10 y 100 por segundo                                                                                                                                                                        |
| Datos          | `app/store.py`       | Documentos versionados con historial append-only; SQLite por defecto, PostgreSQL con `DATABASE_URL`                                                                                                                                                                    |
| Web            | `app/static/`        | Bandeja, editor ciego/asistido, adjudicación, playground, comparación, SUS                                                                                                                                                                                             |
| Administración | `app/cli.py`         | Usuarios, proyecto, importador del piloto, tareas, casos, particiones, cola de desarrollo, promoción, respaldo verificado, reporte de calidad                                                                                                                          |

### Desviaciones justificadas respecto del plan

- **Cola en la base de datos en vez de Celery + Redis.** Celery no soporta Windows nativo y el equipo trabaja en Windows. La base ya es la fuente de verdad del estado; el reclamo de trabajos es atómico (`UPDATE … WHERE status = 'queued'`). Se conserva lo que el plan exige: proceso separado, un análisis pesado por worker, idempotencia, reintentos limitados y error por etapa.
- **Cliente sin build (JavaScript de módulos + WaveSurfer 7.12.12 fijado) en vez de React + TypeScript.** No existía frontend que reutilizar y el objetivo es el mínimo de código. El estado remoto se resuelve con `fetch`; el estado de edición es local y explícito (selección, deshacer, borrador pendiente).
- **SQLite por defecto.** Suficiente para el piloto; `DATABASE_URL=postgresql+psycopg://…` usa PostgreSQL con el mismo SQL (instalar `psycopg`). Pasar a PostgreSQL antes de la anotación principal con varias personas concurrentes.

## 2. Puesta en marcha

```powershell
pip install -r requirements.txt
python -m app.cli add-user datos admin            # imprime el token una sola vez
python -m app.cli add-user ana annotator
python -m app.cli add-user beto annotator
python -m app.cli add-user carla adjudicator
python -m app.cli add-user rita researcher
python -m app.cli add-project tesis --purpose "Disfluencias en español" --guide guia-anotacion-v1
python -m app.cli import-pilot C:\zArnes-tesis\gpt\piloto-daniel --speaker spk_daniel --project tesis
python -m app.cli make-tasks <rec_id> --annotators <id_ana>,<id_beto> --split pilot --segments s001,s012,s024
uvicorn app.api:app --port 8000          # web: http://127.0.0.1:8000
python -m app.worker                     # en otra terminal
```

Solo el primer usuario administrador necesita el CLI. Lo demás (personas y tokens, tareas, casos de adjudicación, conjuntos congelados y evaluación) se hace desde la web: **Administración** y **Experimentos**. Cada vista abre con una guía «¿Qué es esta vista y cómo se usa?».

`MRCD_CHECKPOINT` indica el checkpoint del worker (por defecto `models/ckpt_audio_text.pt`). El micrófono del navegador requiere `localhost` o HTTPS.

## 3. Roles

| Rol               | Puede                                                         | No puede                                                     |
| ----------------- | ------------------------------------------------------------- | ------------------------------------------------------------ |
| `annotator`       | Sus tareas, audio recortado de su región, su anotación        | Ver ejecuciones, predicciones, exportaciones o tareas ajenas |
| `adjudicator`     | Sus casos A/B anónimos y la decisión final                    | Ver modelos o nombres de anotadores                          |
| `researcher`      | Grabaciones, ejecuciones, comparaciones, tareas, evaluaciones | Crear snapshots                                              |
| `admin` (datos)   | Todo, incluidos casos y snapshots                             | —                                                            |
| `user` (producto) | Grabar/cargar, analizar con MRCD, feedback, SUS               | Ejecutar GPT/reglas                                          |

## 4. Datos, particiones y separación

1. **Depuración inmediata**: piloto Daniel (3 × 30 s). Ya inspeccionado: nunca será prueba final.
2. **Piloto de guía**: 10–15 min variados, ampliables a 40–60. Todo es material de desarrollo del protocolo.
3. **Corpus principal**: meta logística de 30 participantes × 2 sesiones de ~3 min.
4. **Partición antes de entrenar**: `python -m app.cli assign-splits tesis --dev 6 --test 6 --exclude-test <hablantes_del_piloto>`. Agrupa por hablante; se niega a rehacerse sin `--force`.
5. **Expansión** si las clases de interés son raras.

Controles automáticos: el snapshot rechaza (HTTP 409) un hablante o un audio presente en dos particiones; el conjunto de prueba solo admite tareas ciegas; la selección activa rechaza casos de prueba; el endpoint de evaluación registra cuántas evaluaciones previas tuvo el mismo snapshot.

| Uso           | Sugerencias                                           | Entrena                                          |
| ------------- | ----------------------------------------------------- | ------------------------------------------------ |
| Entrenamiento | Sí, con procedencia `human_verified_model_suggestion` | Sí (`scripts/snapshot_windows.py --split train`) |
| Desarrollo    | Referencia independiente primero                      | Solo si se crea otro desarrollo independiente    |
| Prueba        | Nunca                                                 | Nunca                                            |

## 5. Protocolo de evaluación (fijar antes de ver prueba)

- Emparejamiento uno a uno, misma clase, máxima cardinalidad; IoU principal **0,50**, sensibilidad **0,30 y 0,70**.
- Error absoluto de inicio y fin en pares emparejados junto con la proporción emparejada.
- Métrica secundaria con tolerancia de **100 ms** en inicio y fin, reportada aparte.
- Recall por duración: <300 ms, 300–1000 ms, ≥1000 ms.
- Falsas alarmas de disfluencia por minuto evaluable y sobre usos legítimos.
- Grupos separados: disfluencias (5 clases) y pausas (2). Macro con lista explícita de clases con soporte.
- Solo cuenta audio con cobertura explícita. Inciertos y no evaluables se informan, no se puntúan.
- IC 95 % por remuestreo de hablantes; diferencia pareada cuando se comparan dos sistemas.
- Acuerdo humano sobre anotaciones independientes previas a la adjudicación: existencia, clase (κ sobre unidades comunes que incluyen omisiones como «none») y fronteras por separado.

El protocolo viaja en cada `evaluation` y el snapshot es inmutable (hash SHA-256 de su contenido).

## 6. Comparadores

| ID      | Cómo se obtiene                                                                                                                                                |
| ------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| B0      | Sistema `rules` (cascada heurística sobre la misma preparación)                                                                                                |
| B1–B3   | `scripts/baselines.py`: texto, acústica+prosodia, concatenación, sobre las mismas ventanas                                                                     |
| M0 / M1 | Sistema `mrcd` con checkpoint y configuración registrados (`model_version`)                                                                                    |
| G       | Sistema `gpt`: requiere `external_processing` en el proyecto y `OPENAI_API_KEY`; prompt versionado en `config/gpt_prompt_v1.txt` (SHA-256 igual al del piloto) |
| V       | Pendiente: solo cuando la integración de video supere sus puertas                                                                                              |

## 7. Ciclo de mejora

1. Ejecutar la versión actual sobre desarrollo (playground o API).
2. Clasificar errores: `evaluate` separa clase, intervalo, sin soporte y omisión; ASR, hablante y segmentación se etiquetan en revisión.
3. `python -m app.cli queue-dev-cases <run_a> <run_b> --n 20`: 40 % desacuerdos, 30 % inciertos, 30 % aleatorio, en ronda por hablante.
4. Barrido completo de regiones aleatorias (tareas ciegas normales).
5. Corregir/adjudicar; `POST /api/dataset-snapshots`.
6. Cambiar **una** hipótesis (por ejemplo `asr_chunk_s`, `refine_boundaries`, `thresholds`, `temperature`).
7. Evaluar en desarrollo con el mismo protocolo (`POST /api/evaluation-runs`).
8. `python -m app.cli promote <mdl_id> --reason …`. Los checkpoints anteriores se conservan.

Umbrales y temperatura se ajustan en desarrollo (`evaluation.pick_threshold`, `evaluation.fit_temperature`), nunca en prueba ni para imitar a GPT.

## 8. Puertas para streaming y video

No se implementan todavía, como indica el plan: primero debe funcionar el flujo posterior a la grabación y el núcleo debe pasar su evaluación. La demo de micrófono existente (`scripts/live_demo.py`) sigue siendo experimental. Condiciones para abrirlas: métricas de desarrollo estables, RTF ≤ 1 medido y, para video, captura sincronizada con desfase medido al inicio, centro y final.

## 9. Operación y protección de datos

- Medios por hash, sin URLs públicas; el token viaja en cabecera o en la URL del medio (solo `localhost`).
- `python -m app.cli backup D:\respaldo` copia la base y los medios y **verifica** la copia (conteo de documentos y hash de cada original). Hacer y restaurar una copia antes de la anotación principal. Con PostgreSQL: `pg_dump` + copia de `data/app/media`.
- Salir de la web borra los datos locales del navegador.
- GPT es opt-in por proyecto; sus respuestas se guardan por hash de audio+modelo+prompt y no se reintentan automáticamente.
- Las trazas no guardan claves ni audio en base64.

## 10. Estado del backlog

| ID      | Estado                | Evidencia o pendiente                                                                                                                                                                                                 |
| ------- | --------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| P01     | Hecho                 | `import-pilot` reproduce MRCD 57 eventos (30 pausas), 413,36 s + 3,81 s; reglas `partial` (s024 sin ejecutar); GPT 6 eventos + 1 uso legítimo; cero parejas GPT–MRCD misma clase con IoU ≥ 0,5; reimportar no duplica |
| P02     | Borrador              | `docs/guia-anotacion-v1.md`; falta aprobación del equipo y clips de ejemplo                                                                                                                                           |
| P03     | Hecho                 | Milisegundos enteros `[inicio, fin)`, desplazamiento del recorte en `X-Media-Offset-Ms`; prueba de ida y vuelta EAF/CSV                                                                                               |
| P04     | Hecho                 | WAV estéreo 44,1 kHz y WebM/Opus → 16 kHz mono verificados con ffmpeg real                                                                                                                                            |
| P05     | Hecho                 | Perfil en s001: ruta histórica por ventana 78,4 s solo en extracción; ruta preparada 8,57 s en total (RTF 0,21); diferencia máxima de probabilidades 4,5·10⁻⁸                                                         |
| P06     | Hecho                 | Reintento sin duplicados, error con etapa, recuperación de trabajos abandonados, fallo de reglas no afecta a MRCD                                                                                                     |
| P07–P10 | Hecho                 | Prueba de recorrido completo con dos anotadores, adjudicación, snapshot y evaluación                                                                                                                                  |
| P11     | Hecho                 | Playground: grabar/cargar/ejemplo → analizar con progreso SSE → escuchar → exportar → feedback                                                                                                                        |
| P12     | Hecho                 | Revisión asistida con cinco acciones y procedencia; bloqueada en prueba                                                                                                                                               |
| P13     | Listo para datos      | `scripts/snapshot_windows.py` + `scripts/baselines.py`                                                                                                                                                                |
| P14–P15 | Listo para datos      | `asr_chunk_s`, `refine_boundaries`, `thresholds`, `temperature`; evaluación en desarrollo pendiente de anotación humana                                                                                               |
| P16     | Hecho                 | Vista de comparación con pistas, conteos, acuerdo simétrico en segmentos comunes, coste y versión                                                                                                                     |
| P17     | Instrumentado         | Formulario SUS, telemetría de apertura/guardado/reproducción, `quality-report`; falta el estudio con usuarios                                                                                                         |
| P18–P19 | Bloqueado por puertas | Ver §8                                                                                                                                                                                                                |

### Verificación realizada

- `tests/test_app.py` y `tests/test_evaluation_engine.py`: 18 pruebas del §19 (tiempos y exportación EAF/CSV, ingesta real WAV/WebM con ffmpeg, duplicados e idempotencia, 409 entre pestañas, cegamiento de endpoints y exportaciones, fugas por hablante y por alias, evaluador uno a uno, recuperación y cancelación, pausas aparte, equivalencia de la refactorización, recorrido completo subir → preparar → analizar → A/B → adjudicar → snapshot → evaluar). La suite completa: 192 pruebas pasan; 3 fallos previos a este cambio (falta `cv2`, lectura cp1252 en Windows, credenciales del juez).
- Navegador headless (Chromium) sobre el piloto importado: playground, comparación, dos anotaciones ciegas con autosave confirmado, revisión asistida y adjudicación, sin errores de consola.
- Worker real con Whisper `small` y extracción acústica sobre un recorte de 40 s del piloto: MRCD y reglas terminan; reglas reutiliza la preparación de MRCD (`cache_hit`); GPT se rechaza sin autorización del proyecto.
- `snapshot_windows.py` con ASR real sobre la adjudicación de la prueba de interfaz: 25 ventanas dentro de la región revisada, contrato 1.2.0 validado.
- Revisión adversarial independiente: tres defectos confirmados (caída del worker ante un conflicto, cancelación que podía perderse, recorte sin validar) corregidos con pruebas de regresión.

### Brechas menores conocidas

- La bandeja filtra por estado, no por proyecto; el playground filtra por clase y tipo, no por hablante ni tramo.
- La recuperación de borradores locales del navegador (`pending:*`) se probó manualmente, no con una prueba automática.
- Alias de un mismo participante con dos IDs: `python -m app.cli alias-speaker <proyecto> <alias> <id_canónico>`; los metadatos de variedad, dispositivo y entorno se envían en `meta` (JSON) al subir.

### Limitación conocida

El checkpoint auditado (`C:\zArnes-tesis\mrcd\models\ckpt_audio_text.pt`) no es compatible con la fusión reconstruida de esta rama: `load_ckpt` lo rechaza. Para analizar con MRCD en esta rama hay que reentrenar con `scripts/train_checkpoints.py` (o, cuando exista, con ventanas humanas de `snapshot_windows.py`). La medición del §10/P05 usa una red de pesos aleatorios: vale para tiempo y equivalencia, no para exactitud. En este entorno `silero_vad` no está instalado y la acústica usa VAD por energía; instalarlo para reproducir la configuración de entrenamiento.
