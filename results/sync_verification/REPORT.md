# Verificación de sincronización e integración — MRCD

Fecha de ejecución: 2026-10-20. Todas las cifras provienen de `results.json` (y `raw/`). Nada se estimó ni se interpoló.

## Entorno

Windows 11, Intel i7-13650HX, 23,7 GB RAM. Motor en CPU (torch 2.9.1+cpu; la GPU no se usó). Python 3.12.10. faster-whisper 1.2.1 (Whisper small, int8, trozos de 8 s), Silero VAD 6.2.2. Referencias independientes: WebRTC VAD (webrtcvad-wheels 2.0.14, agresividad 2 fijada antes de medir; 1 y 3 se reportan) y torchaudio MMS_FA (MMS-300m). Semillas: subconjunto 20260101; bootstrap `default_rng(0)`.

## Corpus

87 grabaciones, 161,5 min, máximo 300 s, 18 videos fuente. Los mp4 de segmentos son recortes de rostro a 10 fps, sin pista de audio.

## Resultados

| Prueba                              | Ejecutada | N                                   | Métricas medidas                                                                                                                                                                      | Meta adoptada                                               | Resultado                                                                                           |
| ----------------------------------- | --------- | ----------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------- | --------------------------------------------------------------------------------------------------- |
| (f) Integración                     | Sí        | 12 grab., 26,97 min, 2 corridas     | 22,9 y 21,0 min de pared por hora de audio; RTF 0,38 y 0,35; 0 fallos; hash A=B en 48/48 archivos; EAF 12/12 válidos contra EAFv3.0.xsd; `events.json` con start<end estricto en 7/12 | ≤20 min/h; fallos ≤5 %; hash 100 %; esquema ≥95 %; RTF ≤1,0 | Tiempo por hora **no cumplido** (22,9 y 21,0). Fallos, hash, esquema EAF y RTF cumplidos            |
| (c) Pausas vs WebRTC VAD            | Sí        | 87 grab., 596 eventos               | Agresividad 2: IoU de frames 0,35; F1 0,52 (IC95 0,46–0,56); F1 de evento IoU≥0,5 0,49. Agresividad 1: 0,46. Agresividad 3: 0,71                                                      | F1 temporal ≥0,70                                           | **No cumplido** con la agresividad fijada antes de medir. Con 3 (elegida después) llega a 0,71      |
| (a) Offset inyectado                | Sí        | 696 ensayos                         | Todas las grabaciones: 80,5 % dentro de ±40 ms, MAE 53,5 ms. Sin envolventes degeneradas (N=544, regla definida después de verlas): 92,1 %, MAE 24,7 ms                               | MAE ≤40 ms y ≥90 % dentro de ±40 ms                         | **No cumplido** con todas las grabaciones. Cumplido solo tras excluir 19/87 envolventes degeneradas |
| (e) Deriva                          | Sí        | 27 grab. ≤300 s; 1 clip de 38,7 min | Segmentos: pendiente media −1,1 ms/min (IC −12 a +9). Clip largo: +4,9 ms/min (ventana 60 s), +7,4 (120 s), +5,4 (300 s)                                                              | ≤10 ms/min                                                  | Segmentos **inconclusos** (el IC supera la meta). Clip largo dentro de la meta, pero N=1            |
| (b) Whisper vs MMS_FA               | Sí        | 4332 palabras, 12 grab.             | Error de borde mediana 54,7 ms (inicio 74,0; fin 34,5). Dentro de 20/50/100 ms: 22,6 / 46,7 / 71,5 %. Whisper adelanta el inicio ≈66 ms (mediana con signo)                           | Mediana ≤50 ms; ≥90 % dentro de 100 ms                      | **No cumplido**                                                                                     |
| (d) Sincronía audio-video (SyncNet) | **No**    | —                                   | Un proxy jaw_open vs RMS (N=85) quedó inconcluso: 21/85 pegados al borde de ±500 ms                                                                                                   | —                                                           | No ejecutada: los mp4 no tienen audio y están a 10 fps                                              |

## Hallazgos adicionales

- Offset nativo palabras-vs-VAD: mediana −129 ms (WebRTC) y −180 ms (Silero). Es un sesgo sistemático de la combinación ASR+VAD, muy por encima de ±40 ms.
- 91 eventos de duración cero en 17 de las 87 grabaciones del corpus (de 1465 eventos).
- 19/87 envolventes de WebRTC saturadas (≥98 % de habla) por ruido o música.
- Las corridas A y B fueron idénticas entre sí; A difiere de la v4 original en 2 archivos (`events.json` y `eaf`, 11/12 iguales).

## Limitaciones

- El aligner y los VAD son referencias con error propio: se reporta acuerdo entre sistemas, no verdad absoluta.
- (a) es una prueba de autoconsistencia del estimador.
- La regla de exclusión de envolventes degeneradas y la elección de la agresividad 3 son posteriores a ver los datos; solo los resultados con todas las grabaciones y con la agresividad 2 son confirmatorios.
- (e): un solo clip largo; no se separa deriva de reloj de sesgo ASR/VAD. El segundo clip largo se detuvo por tiempo.
- Durante parte de las corridas de (f) hubo instalaciones de paquetes en segundo plano.
