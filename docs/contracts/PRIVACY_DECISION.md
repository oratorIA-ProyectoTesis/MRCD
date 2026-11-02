# Decisión de privacidad del audio (app ↔ MRCD)

**ESTADO (2026-11-01): opción B decidida por el equipo; ratificación del asesor PENDIENTE** (ver el registro de la sección 6). Las secciones 1 a 5 conservan el análisis previo a la decisión. Fecha de preparación: 2026-11-01. Autorrevisión de los autores; no es asesoría legal ni sustituye la consulta al asesor o al comité de ética.

## 1. Por qué hay que decidir

- MRCD necesita un WAV mono de 16 kHz, continuo y de al menos 13 s (`APP_INTEGRATION_CONTRACT.md`, parte A). La aplicación recibe el audio en trozos de 160 ms y **hoy no conserva audio** (propiedad "no se retiene audio crudo").
- Para llamar a MRCD, el backend tendría que **armar el audio** en algún lugar. Eso cambia esa propiedad y exige una decisión explícita y un texto de consentimiento (`APP_INTEGRATION_CONTRACT.md`, B.8).
- La hoja de OE2 ya lo declara como decisión abierta (punto 7 de la sección 1).
- El kit de micrófono (`docs/microphone-check/`) usa por defecto `record=False`: ni audio ni texto se guardan; solo estadísticas de nivel. La demo en vivo del navegador, en cambio, conserva la sesión completa en RAM y la expone en `/download/audio.wav` (limitación documentada, sin cambio de código).
- La voz y el rostro de personas reales son datos personales. El `README.md` del repositorio remite a la Ley 29733 y a consultar al asesor o comité de ética; **no se verificó aquí su aplicación concreta** ni existe aprobación ética (el Capstone local lo declara: sin aprobación ética ni registros de consentimiento).

## 2. Hechos del sistema que condicionan la decisión

| Hecho | Fuente |
|---|---|
| El servicio MRCD decodifica en memoria, no escribe en disco y guarda solo resultados JSON acotados en memoria | `APP_INTEGRATION_CONTRACT.md` B.8 |
| `include_text=false` por defecto: el texto de la transcripción no sale salvo petición explícita | idem |
| El servidor de desarrollo no tiene autenticación ni TLS: solo `loopback`; rechaza otras direcciones sin `--allow-non-loopback` | idem |
| Los registros (`logs`) no deben contener audio ni texto | idem |
| El vínculo app-resultado usa el hash SHA-256 del audio (`request.audio_sha256`); el hash no es el único dato que se guarda: ver la sección 6 (qué se persiste exactamente) | idem, B.10 |
| El video nunca entra al motor; el video no sale del navegador | `docs/architecture/README.md`, límites de confianza |
| Un audio menor de 13 s o con huecos produce `audio_too_short` o `not_run`, nunca un resultado rellenado | `APP_INTEGRATION_CONTRACT.md` B.10 |
| La transcripción y el audio pueden revelar contenido sensible aunque no se guarden nombres | observación general, sin cifras |

## 3. Opciones

| | A. Sin retención (no integrar audio) | B. Búfer en memoria con descarte | C. Almacenamiento temporal cifrado | D. Consentimiento explícito con retención opcional |
|---|---|---|---|---|
| Qué hace | La app no llama a MRCD con audio real; MRCD solo se usa fuera de la app (archivos de prueba, corpus) | El backend arma el WAV en memoria por segmento y lo envía a MRCD; descarta al terminar el trabajo (`finally`); nada del audio se escribe en disco ni en la base de datos; se guarda el resultado JSON acotado (eventos con clase, inicio, fin y puntaje; resumen; procedencia; advertencias) más el hash SHA-256 del audio enviado, nunca audio ni transcripción | El backend escribe un archivo temporal cifrado (clave por sesión) y lo borra al terminar o por expiración | Como B o C, más una casilla de consentimiento informado y una opción separada y desactivada por defecto para conservar audio con un fin declarado (p. ej. auditoría o reprocesamiento) |
| Riesgo de privacidad | Mínimo | Bajo: el audio vive en RAM del backend durante el análisis; un volcado de memoria o un fallo que persista datos lo expone | Medio: aparece un archivo en disco; si la clave o el borrado fallan, queda audio recuperable | Mayor: hay conservación real; exige base legal, finalidad, plazo, derechos de acceso y supresión |
| Costo de implementación | Ninguno | Bajo a medio: búfer por sesión, manejo de huecos, tope de memoria y de duración | Medio a alto: gestión de claves, limpieza ante caídas, pruebas de borrado | Alto: UI de consentimiento, registro de consentimientos, política de retención y supresión |
| Efecto en OE3 (integración real) | **Bloquea** la integración de extremo a extremo con audio de usuarios; solo caben pruebas con archivos | Habilita la integración con la menor huella | Habilita; protege ante caídas, pero añade superficie | Habilita y permite reprocesar; más cargas de gobernanza |
| Efecto en el reporte | La sección de disfluencias queda "no disponible: integración no habilitada" | Normal | Normal | Normal |
| Efecto en el kit de micrófono | Ninguno (ya no guarda nada) | Ninguno; el kit sigue siendo una prueba local independiente | Ninguno | Si se ofrece conservar, debe usar el mismo texto de consentimiento y el opt-in `--keep-audio` |
| Efecto en el kit SUS | La aplicación no podría mostrar disfluencias; el SUS evaluaría un producto sin la función | Compatible: los participantes consienten un análisis efímero | Compatible | Compatible; el consentimiento del SUS puede ser el mismo documento |
| Qué NO resuelve | -- | Un servidor MRCD sin autenticación ni TLS no debe exponerse; la red entre app y MRCD debe ser local o protegida | Idem | Idem; no sustituye la aprobación ética |

## 4. Riesgos transversales (cualquier opción distinta de A)

1. **Datos de terceros en el audio** (otras voces, contenido personal del discurso).
2. **Fuga por registros** (logs con texto o rutas de audio): prohibida por contrato, pero no verificada en la app.
3. **Transcripción como dato personal:** `include_text=false` reduce la exposición; si se activa, el reporte pasa a contener texto del usuario.
4. **Servidor sin autenticación:** el servicio de desarrollo es solo para `loopback`; una integración real requiere decisión sobre autenticación y TLS.
5. **Fallos parciales:** si el trabajo falla, el audio debe descartarse igual; hay que probar el descarte en el camino de error.
6. **Cumplimiento y ética:** la aprobación ética, la base legal y los textos de consentimiento no existen hoy; ninguna opción los sustituye.
7. **El video no se envía a MRCD** y nunca se infiere un estado psicológico a partir de él; esta decisión es solo sobre audio.

## 5. Recomendación (para discusión; no es una decisión)

**Opción B (búfer en memoria con descarte)** como mínimo viable para el piloto de integración y el SUS, con estas condiciones: (1) descarte garantizado también en errores, con prueba; (2) solo persiste el resultado JSON acotado y el hash SHA-256 del audio enviado (nunca audio, transcripción ni palabras; ver la sección 6); (3) `include_text=false` por defecto; (4) sin registros de contenido; (5) servidor MRCD en loopback o red privada; (6) texto de consentimiento informado leído y aceptado antes del análisis (el kit de la carpeta `docs/sus-kit/` incluye un borrador); (7) sin conservar audio. Si el asesor o el comité exigen más garantías, pasar a C; si se desea conservar audio, **D** requiere antes aprobación ética y política de retención. Mientras no haya decisión, la opción vigente de hecho es **A** (la app no retiene audio y no llama a MRCD).

Compensación: B es la menor huella que aún permite OE3 real y el SUS; no elimina el riesgo de exposición en memoria durante el análisis, y no sustituye el consentimiento ni la aprobación ética.

## 6. Registro de la decisión

| Campo | Valor |
|---|---|
| Opción elegida | **B. Búfer en memoria con descarte** |
| Condiciones adicionales | (1) El audio PCM16 de 16 kHz mono de la sesión se acumula solo en memoria del backend de la app, con tope de duración configurable (`ORATORIA_MRCD_MAX_AUDIO_SECONDS`, 600 s por defecto) y error claro al excederlo; (2) al cerrar la sesión se envía una vez al servicio MRCD (loopback o red interna, URL por variable de entorno, sin claves en el repositorio); (3) solo se guarda el JSON del resultado vinculado a la sesión (más el hash SHA-256 del audio enviado); la lista exacta está en el addendum de abajo; (4) el búfer se borra de forma explícita al terminar, ante error, ante límite excedido, ante desconexión abrupta y al apagar el servicio; (5) nunca se escribe audio crudo en disco ni en registros; (6) `include_text=false` por defecto; (7) si MRCD falla o no está disponible, el reporte dice "no disponible" con el motivo, nunca 0; (8) el servicio MRCD sigue sin autenticación ni TLS: no exponerlo fuera de loopback o red interna |
| Decide (nombre) | El equipo del proyecto (decisión del equipo, no del asesor) |
| Fecha | 2026-11-01 |
| Ratificación del asesor | **PENDIENTE** |
| Firma | (no aplica: decisión registrada por el equipo; ratificación y firma del asesor pendientes) |

Lo que esta decisión **no** resuelve y sigue abierto: texto de consentimiento informado que mencione el análisis de audio en memoria (el actual de registro solo remite a los Términos y a la Política de privacidad), aprobación ética, base legal bajo la Ley 29733, autenticación y TLS entre la app y MRCD, y el efecto de un volcado de memoria del backend durante el análisis. Si el asesor o el comité exigen más garantías, la alternativa es la opción C.

Implementación (rama local `feat/mrcd-integration` de `ORATORIA-PROYECTO`, no mezclada): `backend/src/oratoria/platform/mrcd/` (búfer, cliente, orquestación), `platform/ws/connection_handler.py` (acumulación y entrega al cerrar o desconectar), columna JSONB `practice_session.mrcd_analysis`. Evidencia: `results/mejora_objetivos_20261101/S7_app/README_S7.md`. Autorrevisión de los autores; sin revisión independiente.

## 7. Addendum 2026-11-01 (correcciones tras la revisión independiente de S7)

**Autorrevisión de los autores; pendiente de ratificación del asesor y de revisión ética.** El texto de consentimiento y la política de privacidad del análisis son un BORRADOR.

### 7.1 Qué se persiste exactamente (corrige la contradicción entre las secciones 5 y 6)

Se persiste, vinculado a la sesión (`practice_session.mrcd_analysis`), un documento JSON **permitido por lista blanca** (`platform/mrcd/result_schema.py`):

- por evento: `id`, `class`, `start_s`, `end_s`, `interval_semantics`, `score` (valor, tipo, `calibrated:false`), `n_windows`, `modalities_used`, `validation`, `experimental`, `in_scope`, `decision` si se pidió;
- `summary` (total, eventos por minuto, conteos por clase, fuera de alcance), `request` (hash SHA-256 del audio, duración, perfil ASR, bandera `include_text`), `modalities`, `provenance`, `experimental`, `warnings`;
- el vínculo con la sesión (`binding`: tramos recibidos, huecos, duración, hash) y la versión del texto de consentimiento.

**Nunca** se persiste audio, transcripción, texto de eventos ni palabras: `events[].text` y `transcript` se guardan siempre como `null` aunque el servicio los enviara, y cualquier campo fuera de la lista se descarta (se registra solo el nombre del campo, jamás su contenido). La respuesta del servicio se lee con tope de tamaño (2 MiB) y el documento guardado tiene tope (512 KiB, 5000 eventos).

### 7.2 Consentimiento específico (opt-in)

El análisis requiere una casilla propia, **no premarcada**, en el registro (y un interruptor en "Privacidad" para activarlo o retirarlo, con opción de borrar los resultados ya guardados). La decisión se guarda por cuenta con fecha y versión del texto (`mrcd_consent`); una decisión solo vale para la versión vigente del texto. Sin consentimiento activo no se acumula audio, no se llama a MRCD y el reporte dice "No disponible: sin consentimiento para análisis de audio". El consentimiento se vuelve a comprobar al cerrar la sesión (puede retirarse durante la práctica). Si el almacén de consentimientos falla, se trata como "sin consentimiento".

### 7.3 Garantía de descarte, dicha con precisión

"Descartar" significa: el `bytearray` acumulado se sobrescribe con ceros y se vacía, y el búfer rechaza audio nuevo. **No** alcanza las copias inmutables que Python hace en el camino (cada trama recibida, el WAV que se construye al cerrar y el cuerpo de la petición): se liberan cuando nada las referencia y las recupera el intérprete, pero no se sobrescriben. Un volcado de memoria del proceso durante el análisis podría exponer audio. La afirmación correcta es "liberado de la memoria tras el análisis", no "borrado forense". Se mantiene el mínimo de copias (una acumulada y una sola copia WAV mientras se envía).

### 7.4 Límites y robustez

Tope global de búferes simultáneos, tope por cuenta, tope de bytes en memoria y tope de análisis pendientes (`ORATORIA_MRCD_MAX_BUFFERS`, `..._PER_ACCOUNT`, `..._MAX_BUFFERED_MB`, `..._MAX_PENDING_ANALYSES`); al excederlos la sesión sigue, el audio no se guarda y el reporte lo dice. Al apagar, los análisis en curso se cancelan con plazo acotado (5 s) y el hilo del cliente HTTP se detiene en su siguiente comprobación; un análisis `pending` huérfano (caída del proceso) pasa a "No disponible: análisis interrumpido". El cliente no sigue redirecciones, valida el `job_id` y rechaza `http://` hacia hosts que no sean loopback salvo opt-in explícito (`ORATORIA_MRCD_ALLOW_INSECURE_REMOTE_HTTP=true`).

### 7.5 Sigue abierto

Ratificación del asesor de la opción B y del texto de consentimiento; aprobación ética y base legal (Ley 29733); texto de Términos de servicio; autenticación y TLS entre la app y MRCD; prueba con micrófono real; las rutas Postgres no tienen prueba automatizada propia (se ejercitaron en la prueba de extremo a extremo).

## 8. Addendum 2026-11-01 (variante "B incremental" frente a B original)

**Autorrevisión de los autores; la ratificación del asesor y la revisión ética están PENDIENTES.** El texto de consentimiento sigue siendo un BORRADOR; su versión pasó a `2026-11-01-draft-2-incremental`, de modo que **las cuentas que aceptaron la versión anterior vuelven a ver el aviso y deben consentir de nuevo** (una decisión solo vale para la versión vigente).

| Aspecto | B original (vigente por defecto) | B incremental (opt-in, `ORATORIA_MRCD_MODE=incremental`) |
| --- | --- | --- |
| Dónde vive el audio | Backend durante la sesión; MRCD solo mientras dura el análisis | Backend durante la sesión **y MRCD durante toda la sesión**: **dos copias en memoria, en dos procesos** |
| Exposición en MRCD | Duración del análisis | Duración de la sesión más el cierre |
| Disco / registros | Nunca | Nunca; la caché de fragmentos (palabras = dato personal) es solo memoria y se descarta con la sesión |
| Transcripción | No sale de MRCD si `include_text=false` | Igual; los eventos de progreso llevan solo conteos |
| Descarte | `discard()` en todos los caminos | Además, en MRCD: `DELETE` de la sesión (ceros + caché vacía) al terminar, cancelar, fallar, expirar por inactividad (`--session-idle-timeout-s`, 120 s) y apagar |
| Límites | Tope 600 s y topes de capacidad de la app | Los mismos, más `--max-sessions` (4) y el tope de duración por sesión en MRCD |

Riesgo nuevo, declarado: la ventana en la que un volcado de memoria del proceso de MRCD expondría audio pasa de "duración del análisis" a "duración de la sesión", y existen dos copias simultáneas. Las mismas salvedades de 7.3 aplican a ambas copias (los `bytes` inmutables intermedios no se sobrescriben). Garantía de borrado en MRCD, exacta: se sobrescriben con ceros el búfer PCM de la sesión, el WAV canónico armado al cerrar, el acumulador del cuerpo de cada petición y las copias `raw`/float32 de cada fragmento. NO se pueden sobrescribir (solo se liberan): los `bytes` inmutables que entrega el servidor HTTP al leer un cuerpo, los `BytesIO` y arreglos float32 creados dentro del análisis offline (que no se modifica), las copias internas de Whisper/CTranslate2 y lo que el asignador de Python conserve tras liberar. Por eso se habla de "minimizar y descartar", no de "borrado forense". Mitigaciones: modo apagado por defecto, consentimiento por versión, descarte garantizado en todos los finales, MRCD solo en loopback / red controlada.

Ante cualquier fallo del camino incremental (hueco, tope, red, reinicio de MRCD, conflicto) la app **cancela la sesión en MRCD** y usa el modo offline con el búfer completo que ella conserva; nunca se entrega un resultado parcial. El resultado persistido es el mismo documento sanitizado (lista blanca de 7.1); el campo `binding.mode` registra `offline` o `incremental` (y `fallback_from`/`fallback_reason` si hubo vuelta atrás). El modo `incremental_verify` guarda el resultado offline y registra solo `equal` y la lista de rutas distintas, nunca contenido.

Sigue abierto, además de 7.5: ratificación del asesor de este cambio de exposición y aprobación ética del texto nuevo; decidir si el modo incremental pasa a ser el predeterminado (criterios en `DESIGN_B_INCREMENTAL.md`, sección 5).

### 8.1 Correcciones tras la revisión S8 (2026-11-01; autorrevisión)

- **Acceso al progreso y a la sesión.** Cada sesión de MRCD emite al abrirse un `subscription_token` aleatorio (solo memoria, un único uso de entrega). Audio, cierre, estado, eventos y borrado lo exigen; sin él o con otro, la respuesta es la misma que para una sesión inexistente. Sigue sin haber TLS ni cuentas entre app y MRCD: el token evita que quien conozca o adivine un identificador lea o cancele una sesión ajena, no sustituye una red controlada.
- **El flujo de eventos es "solo conteos" también en el evento final**: ya no lleva el documento ni texto (con `include_text=true` la transcripción viajaba en `completed`). El documento se obtiene por `GET /v1/sessions/{id}` con el token.
- **Tope global de memoria**: `max_total_audio_bytes` (128 MiB por defecto) suma los búferes PCM y el WAV canónico de todas las sesiones vivas; al superarlo, `429 session_limit` y la app vuelve al modo offline.
- **Cancelar durante el cierre** descarta el audio de inmediato y el análisis deja el hilo en el siguiente fragmento por decodificar; el fragmento en curso (o las etapas posteriores al ASR) no se puede interrumpir, y su resultado se descarta.

## 9. Addendum 2026-11-01 (S9: texto breve de cada disfluencia y grabaciones con pausas)

Autorrevisión de los autores; pendiente de revisión del asesor y del comité de ética.

### 9.1 Texto breve de cada evento (decisión del equipo, 2026-11-01)

- **Qué cambia.** Hasta S8 el documento guardado llevaba `events[].text = null` siempre. Con esta decisión, una segunda preferencia, **separada** del consentimiento del análisis de audio, permite guardar por evento como máximo **8 palabras** del texto que MRCD reconoció en la ventana de ese evento (`include_text=true` en la petición a MRCD, offline o incremental, por tramo). La transcripción completa (`transcript`) sigue en `null` y nunca se guarda ni se registra; el flujo de eventos de MRCD sigue llevando solo conteos (el texto llega por el `GET` del resultado).
- **Casilla premarcada (decisión del equipo).** La casilla "Guardar el texto breve de cada disfluencia detectada" viene **marcada por defecto** en el registro y en Privacidad. El valor inicial es configurable con `ORATORIA_SNIPPETS_DEFAULT` (por defecto `true`) y el formulario lo lee de `GET /api/v1/mrcd/snippets-config`. Solo rellena el formulario: el backend guarda texto únicamente para cuentas con una decisión **guardada** (`PUT /api/v1/users/me/mrcd-snippets`, con fecha y versión del texto `2026-11-01-draft-1-snippets`) y con el análisis de audio activo. Sin decisión guardada, o con una versión de texto antigua, no se pide ni se guarda texto.
- **Retiro.** Desmarcar la casilla guarda `enabled=false` y **borra** los textos breves de todos los resultados guardados de la cuenta (los conteos y tiempos se conservan). Si se retira entre el inicio y el cierre de una sesión, el texto se descarta antes de guardar.
- **Advertencia legal y ética.** Una casilla premarcada **puede no calificar como consentimiento expreso** bajo la Ley 29733 (que exige un consentimiento libre, previo, expreso, informado e inequívoco). Por eso el valor por defecto es configurable, queda registrado como decisión del equipo y **sigue pendiente de revisión del asesor y de ética**; si el comité lo rechaza, basta `ORATORIA_SNIPPETS_DEFAULT=false` (sin cambio de código) o retirar la función.
- **Riesgo declarado.** El texto de las primeras 8 palabras de una ventana es contenido hablado de la persona (puede incluir datos personales dichos en voz alta). Se mitiga con el tope de 8 palabras, el opt-in guardado, el borrado al desmarcar y el borrado de la cuenta, pero no desaparece. El texto del consentimiento del análisis de audio (`2026-11-01-draft-2-incremental`) NO se modificó: la preferencia nueva lleva su propio texto, que declara lo que se suma.

### 9.2 Grabaciones con pausas

- La app ya no rechaza la sesión con `audio_gap`. El búfer en memoria registra los tramos continuos (cortes de secuencia o saltos del reloj de captura mayores de 1.5 s); cada tramo se analiza por separado en MRCD (en el modo incremental, con su propia sesión y su propio token) y los eventos se combinan sumando el desfase real de cada tramo dentro de la sesión. No se rellena ni se une audio.
- Un tramo menor que el mínimo del motor (13 s) se informa como "tramo no analizado: demasiado corto"; en el modo incremental su sesión de MRCD se cancela (la copia remota se pone en ceros) sin analizarlo. El documento guardado conserva por tramo su posición, duración, estado, modo, hash y procedencia.
- Privacidad sin cambios de fondo: el búfer sigue solo en memoria y se descarta siempre; en el modo incremental pueden coexistir más de una sesión de MRCD por sesión de la app (una por tramo), con las mismas garantías de descarte.
