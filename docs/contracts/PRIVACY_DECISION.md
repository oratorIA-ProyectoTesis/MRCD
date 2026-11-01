# Decisión de privacidad del audio (app ↔ MRCD)

**ESTADO: PENDIENTE DE DECISIÓN del usuario y del asesor.** Este documento prepara la decisión; no la toma. Ninguna opción está aprobada, implementada ni probada. Fecha de preparación: 2026-11-01. Autorrevisión de un agente de software; no es asesoría legal ni sustituye la consulta al asesor o al comité de ética.

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
| El vínculo app-resultado conserva solo el hash SHA-256 del audio (`request.audio_sha256`) | idem, B.10 |
| El video nunca entra al motor; el video no sale del navegador | `docs/architecture/README.md`, límites de confianza |
| Un audio menor de 13 s o con huecos produce `audio_too_short` o `not_run`, nunca un resultado rellenado | `APP_INTEGRATION_CONTRACT.md` B.10 |
| La transcripción y el audio pueden revelar contenido sensible aunque no se guarden nombres | observación general, sin cifras |

## 3. Opciones

| | A. Sin retención (no integrar audio) | B. Búfer en memoria con descarte | C. Almacenamiento temporal cifrado | D. Consentimiento explícito con retención opcional |
|---|---|---|---|---|
| Qué hace | La app no llama a MRCD con audio real; MRCD solo se usa fuera de la app (archivos de prueba, corpus) | El backend arma el WAV en memoria por segmento y lo envía a MRCD; descarta al terminar el trabajo (`finally`); nada se escribe en disco ni en la base de datos; se guarda solo el hash | El backend escribe un archivo temporal cifrado (clave por sesión) y lo borra al terminar o por expiración | Como B o C, más una casilla de consentimiento informado y una opción separada y desactivada por defecto para conservar audio con un fin declarado (p. ej. auditoría o reprocesamiento) |
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

**Opción B (búfer en memoria con descarte)** como mínimo viable para el piloto de integración y el SUS, con estas condiciones: (1) descarte garantizado también en errores, con prueba; (2) solo el hash SHA-256 persiste; (3) `include_text=false` por defecto; (4) sin registros de contenido; (5) servidor MRCD en loopback o red privada; (6) texto de consentimiento informado leído y aceptado antes del análisis (el kit de la carpeta `docs/sus-kit/` incluye un borrador); (7) sin conservar audio. Si el asesor o el comité exigen más garantías, pasar a C; si se desea conservar audio, **D** requiere antes aprobación ética y política de retención. Mientras no haya decisión, la opción vigente de hecho es **A** (la app no retiene audio y no llama a MRCD).

Compensación: B es la menor huella que aún permite OE3 real y el SUS; no elimina el riesgo de exposición en memoria durante el análisis, y no sustituye el consentimiento ni la aprobación ética.

## 6. Registro de la decisión (vacío a propósito)

| Campo | Valor |
|---|---|
| Opción elegida | |
| Condiciones adicionales | |
| Decide (nombre) | |
| Fecha | |
| Firma | |

El agente no completa esta tabla. Hasta que se complete, el estado es **PENDIENTE DE DECISIÓN**.
