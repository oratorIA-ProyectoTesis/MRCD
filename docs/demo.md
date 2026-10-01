# Probar la demo

La demo en vivo corre sobre el motor real y se habilita en sesiones programadas. Pide al equipo el enlace y un token de acceso de rol «Usuario del producto».

## Qué vas a poder hacer

1. **Grabar** entre 20 y 60 s con el micrófono del navegador, **subir** un archivo (WAV, MP3, M4A, WebM, OGG o MP4) o abrir un **ejemplo** ya cargado.
2. Pulsar **Analizar con MRCD** y seguir el progreso por etapas: preparar el audio, transcribir y extraer rasgos, detectar, guardar el reporte.
3. Escuchar cada detección haciendo clic en su franja de color sobre la forma de onda.
4. Filtrar por clase o por tipo (disfluencias, pausas, inciertos) y descargar el resultado en JSON, CSV o como reporte legible.
5. Marcar una detección como «Está bien» o «No corresponde», o avisar «Falta algo» en un punto del audio. Esos comentarios van a revisión humana; no cambian el modelo al instante.
6. Responder la encuesta de usabilidad SUS (10 preguntas).

## Requisitos

- Navegador actual (Chrome, Edge, Firefox o Safari).
- Al menos 13 s de audio: el modelo mira 10 s de contexto para decidir cada tramo.
- Para grabar se necesita permiso de micrófono. Usa auriculares para que el altavoz no se grabe.

## Privacidad

El audio que grabes o subas se guarda en el servidor de la demo para poder analizarlo y revisarlo. No subas grabaciones de otras personas sin su consentimiento.

## Correr la demo en tu propia máquina

Si prefieres no depender del enlace, sigue la [instalación rápida](instalacion.md): en unos minutos tendrás la misma aplicación en `http://127.0.0.1:8000`.
