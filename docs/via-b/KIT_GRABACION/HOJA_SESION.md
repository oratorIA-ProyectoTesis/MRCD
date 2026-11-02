# Hoja de sesión (plantilla)

Una hoja por participante. **No escribir el nombre del participante aquí**: solo el código. La tabla código-persona va en el archivo cifrado aparte.

| Campo                                                           | Valor    |
| --------------------------------------------------------------- | -------- |
| Código del participante                                         | `VB_S__` |
| Fecha y hora de la sesión                                       |          |
| Responsable de la sesión                                        |          |
| Consentimiento firmado (sí/no) y archivado en la carpeta aparte |          |
| Elegible (sí/no; sin motivo)                                    |          |
| Rango de edad (18-24, 25-34, 35-44, 45+)                        |          |
| Variedad de español (país o región, opcional)                   |          |
| Usa lentes (sí/no)                                              |          |
| Commit del repositorio y versión del motor                      |          |
| Micrófono (modelo)                                              |          |
| Cámara (modelo, resolución, fps configurados)                   |          |
| Versión de OBS                                                  |          |

## Tomas

| Archivo                           | Tarea        | Hora inicio | Duración aprox. | Claqueta inicial (sí/no) | Claqueta final (sí/no) | Segunda voz audible (sí/no) | Incidentes | Resultado de `verify` (OK/REVISAR) | ¿Repetida? |
| --------------------------------- | ------------ | ----------- | --------------- | ------------------------ | ---------------------- | --------------------------- | ---------- | ---------------------------------- | ---------- |
| `VB_S__PRES1`                     | Presentación |             |                 |                          |                        |                             |            |                                    |            |
| `VB_S__ESP1`                      | Espontánea   |             |                 |                          |                        |                             |            |                                    |            |
| `VB_S__PRES2` (solo si se repite) |              |             |                 |                          |                        |                             |            |                                    |            |

Motivo de cada repetición (falla técnica, no calidad del habla): ________________________________

## Hoja de custodia

| Evento                                  | Fecha | Responsable | Detalle                     |
| --------------------------------------- | ----- | ----------- | --------------------------- |
| Archivos copiados al contenedor cifrado |       |             |                             |
| Informe técnico generado                |       |             |                             |
| Borrado seguro de crudos                |       |             | Método:                     |
| Retiro de consentimiento (si ocurrió)   |       |             | Borrado en <= 7 días: sí/no |

## Lista de verificación de la sesión

- [ ] Consentimiento firmado antes de grabar
- [ ] Audio con mejoras de Windows desactivadas; sin filtros en OBS
- [ ] Pico entre -12 y -6 dB en el medidor
- [ ] Rostro completo y frontal; una sola persona en la imagen
- [ ] 25 fps y 720p o más
- [ ] Claqueta inicial y final visibles y audibles
- [ ] Archivos con el nombre `VB_S__<TAREA><k>.mp4`, sin nombres propios
- [ ] `verify` ejecutado y resultado anotado
- [ ] Archivos solo en `D:\viab-datos\`, nada en el repositorio, la nube ni el correo
