# Kit de grabación (Windows)

**NO EJECUTADO.** Nadie ha grabado con este kit. No usar con participantes sin aprobación (`../README.md`). Todo lo que sigue se verificó contra el código del repositorio, **no** contra una sesión real; el primer piloto debe confirmar cada paso.

Carpetas de trabajo (fuera del repositorio, en un disco cifrado):

```
D:\viab-datos\
  raw\        VB_S01_PRES1.mkv  ->  VB_S01_PRES1.mp4
  verif\      VB_S01_PRES1.report.json
  runs\       resumenes de corrida del motor (JSON)
  informe\    technical_summary.json / .md
```

## A. Antes de la sesión (10 min)

1. Cuarto silencioso, puerta cerrada, sin otra persona en la imagen, sin pantallas ni espejos detrás.
2. Cámara fija a la altura de los ojos, a 60 a 80 cm; luz frontal suave, sin ventana detrás.
3. Windows: Configuración > Privacidad y seguridad > Micrófono y Cámara: acceso activado.
4. Windows: Configuración > Sistema > Sonido > Entrada: elegir el micrófono definido y **desactivar mejoras de audio** (Propiedades del dispositivo > Mejoras de audio > desactivar).
5. Cerrar otras aplicaciones que usen la cámara o el micrófono.

## B. OBS Studio (grabador recomendado)

Ajustes, una sola vez (el piloto los valida):

| Ajuste                                                       | Valor                                                                                                 |
| ------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------- |
| Configuración > Vídeo > Resolución base y de salida          | 1280x720 (o 1920x1080 si la cámara lo da)                                                             |
| Configuración > Vídeo > FPS comunes                          | **25**                                                                                                |
| Configuración > Audio > Frecuencia de muestreo               | 48 kHz                                                                                                |
| Configuración > Audio > Canales                              | Mono                                                                                                  |
| Configuración > Salida > Modo avanzado > Grabación > Formato | mkv (seguro ante cierres)                                                                             |
| Configuración > Salida > Grabación > Codificador de vídeo    | x264, CRF 20 (o el codificador por hardware del equipo)                                               |
| Configuración > Salida > Pista de audio                      | 1, 192 kbps o más                                                                                     |
| Fuentes                                                      | "Captura de dispositivo de vídeo" (la webcam) + "Captura de entrada de audio" (el micrófono definido) |
| Filtros del micrófono en OBS                                 | **Ninguno** (ni supresión de ruido, ni ganancia, ni compresor)                                        |

**Nivel:** pida al participante que hable con voz normal y mire el medidor del mezclador: el pico debe oscilar entre **-12 y -6 dB** (verde a amarillo), nunca llegar a rojo. Ajuste la ganancia del micrófono en Windows, no en OBS.

**Alternativa sin OBS:** la aplicación Cámara de Windows (modo video) graba video y micrófono, pero no muestra nivel ni deja fijar los fps; solo úsela si OBS no está disponible y verifique todo con el paso D.

## C. La sesión (de 30 a 45 min)

1. Recibir al participante, entregar el consentimiento, **esperar la firma** y asignar el código `VB_S__` (hoja de sesión). Hacer la pregunta de elegibilidad en voz baja y anotar solo sí/no.
2. Iniciar la grabación en OBS **antes** de dar instrucciones y ponerle nombre al archivo después (OBS: Archivo > Mostrar grabaciones). Leer las instrucciones de `../PROTOCOLO_GRABACION.md`, sección 4, sin improvisar.
3. **Claqueta inicial:** el participante da una palmada con las manos visibles a la altura de la cara, se cuenta 3 s en silencio y se hace la seña de inicio.
4. `PRES1` (3 a 5 min); pausa de 1 min; `ESP1` (2 a 3 min). Una grabación (archivo) por tarea: detener y reiniciar OBS entre ellas.
5. **Claqueta final:** palmada visible 3 s después de terminar.
6. Detener la grabación. Agradecer y recordar el derecho de retiro y el plazo.

## D. Verificación el mismo día

1. Remuxar si grabó en mkv: OBS > Archivo > Remuxar grabaciones, mkv a mp4. Renombrar a `VB_S01_PRES1.mp4`.
2. Verificar:

```powershell
cd <repo>
$env:OMP_NUM_THREADS = "4"
.\.venv\Scripts\python.exe -m mrcd.recording_check verify D:\viab-datos\raw\VB_S01_PRES1.mp4 --out-dir D:\viab-datos\verif
```

La compuerta de rostro corre a unos 11 ms por cuadro (una toma de 5 min a 25 fps tarda cerca de 1 min). Con `--no-face` se omite y los criterios de rostro y compuerta quedan "?" (desconocidos, nunca "ok"). El comando imprime `OK` o `REVISAR` y cada chequeo: duración, frecuencia de muestreo, pico, recorte, piso de ruido, resolución, fps, diferencia de duración audio-video, desfase de la claqueta, deriva entre las dos palmadas, fracción admitida por la compuerta S6 y sincronía de la compuerta. **No guarda audio, imágenes ni texto: solo números y el sha256 del archivo.**

3. Qué hacer con cada falla (repetir la toma una vez; si vuelve a fallar, anotarlo y seguir):

| Falla                                               | Causa probable                                   | Acción                                                  |
| --------------------------------------------------- | ------------------------------------------------ | ------------------------------------------------------- |
| `audio_peak_level` / `audio_clipping`               | Ganancia alta o baja                             | Bajar o subir la ganancia en Windows y repetir          |
| `audio_noise_floor`                                 | Ruido de fondo, ventilador                       | Cambiar de cuarto, cerrar ventanas                      |
| `video_resolution` / `video_fps`                    | OBS mal configurado                              | Revisar ajustes de la sección B                         |
| `av_duration`                                       | Cuadros perdidos                                 | Cerrar programas pesados; bajar la resolución de salida |
| `clap_offset` / `clap_drift` desconocido            | No se encontró la palmada                        | Repetir con la palmada más fuerte y manos en cuadro     |
| `admitted_fraction` baja, `face_not_visible_enough` | Rostro fuera de cuadro, de perfil, mal iluminado | Recolocar la cámara o la persona y repetir              |
| `second_face_present...`                            | Otra cara en la imagen                           | Quitar pósters, espejos u otra persona                  |

**No** modifique los umbrales de la compuerta ni del verificador (`mrcd/av/config.py`, `mrcd/recording_check/`): están congelados.

## E. Corrida del motor e informe

1. Con la versión del motor registrada en la hoja de sesión, analice cada grabación (ver `../../contracts/APP_INTEGRATION_CONTRACT.md` para el servicio) y guarde un JSON por grabación en `D:\viab-datos\runs\` con el esquema de `../PREREGISTRO_TECNICO.md`, sección 6. Ninguna salida se usa como etiqueta.
2. Informe:

```powershell
.\.venv\Scripts\python.exe -m mrcd.recording_check report --reports D:\viab-datos\verif --runs D:\viab-datos\runs\runs.json --out D:\viab-datos\informe
```

El informe lleva en su encabezado que **no hay evidencia de exactitud**. No lo edite a mano.

## F. Cierre

- Borrado de crudos en el plazo; llenar la hoja de custodia.
- Copiar al repositorio solo `technical_summary.json` y `.md`, nunca crudos ni reportes por archivo.
