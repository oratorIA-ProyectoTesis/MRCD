# Vía B: kit OPCIONAL de validación técnica con grabaciones propias

**ESTADO: preparado, NO EJECUTADO, OPCIONAL.** Nadie ha grabado ni verificado nada con este kit. No se graba a ninguna persona hasta tener la aprobación del asesor y, si corresponde, del comité de ética (ver `SOLICITUD_ETICA.md`). Fecha de preparación: 2026-11-01. Autorrevisión del trabajo preparado; no hubo revisión independiente.

## Qué es y qué NO es

Decisión final del equipo (01/11/2026): **nadie anota nada**, ni dos personas ni una. Por eso este kit **no produce evidencia de exactitud**: no hay etiquetas humanas, así que no se puede calcular ni afirmar precisión, recall ni F1 con estas grabaciones. Tampoco se usan salidas automáticas del motor como referencia.

Lo que sí mide, sobre grabaciones propias de hablantes nuevos con video frontal:

| Qué                                                                      | Para qué sirve                                                                                                         |
| ------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------- |
| Admisión de la compuerta de video S6 (porcentaje de segmentos admitidos) | Acción 4.1 del plan: saber si el canal de video es evaluable con un montaje controlado (en EPIC solo se admitió 4,5 %) |
| Sincronía audio-video (claqueta y compuerta)                             | Saber si el desfase es verificable y menor de 120 ms                                                                   |
| Cadena de audio con micrófono real (nivel, recorte, ruido)               | Complementa el kit `docs/microphone-check/`                                                                            |
| Robustez y latencia del análisis sobre hablantes no vistos               | El motor termina sin fallar, tiempo de análisis frente a la duración                                                   |

Lo que **no** resuelve (queda abierto): acciones 1.1 a 1.3 (evaluación confirmatoria), 2.2, 2.3, 3.1, 4.2 (ablación A+T frente a A+T+V con exactitud) y 5.3 del plan, porque todas necesitan etiquetas humanas. Con este kit se puede decir, como máximo, "el video es admisible técnicamente en X % de los segmentos", nunca "el video mejora la detección".

## Contenido

| Archivo                                                      | Qué es                                                                                                                                 |
| ------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------- |
| `PROTOCOLO_GRABACION.md`                                     | Objetivo, tamaño y precisión esperada, participantes, tarea de habla, condiciones técnicas, nomenclatura, gestión de datos, cronograma |
| `CONSENTIMIENTO_PARTICIPANTE.md`                             | Borrador de consentimiento (pendiente de aprobación)                                                                                   |
| `SOLICITUD_ETICA.md`                                         | Resumen para el asesor y el comité (borrador)                                                                                          |
| `PREREGISTRO_TECNICO.md` + `PREREGISTRO_TECNICO.freeze.json` | Criterios técnicos fijados con sha256 antes de grabar                                                                                  |
| `KIT_GRABACION/`                                             | Pasos en Windows (OBS), hoja de sesión, lista de verificación                                                                          |
| `_descartado/`                                               | Código y pruebas de anotación y acuerdo entre anotadores que se escribieron antes de la decisión final; **no forman parte del flujo**  |

Código: `mrcd/recording_check/` (verificación por archivo, informe agregado, congelamiento). Pruebas: `tests/test_via_b_recording_check.py`.

## Flujo

1. **Ética** (equipo → asesor): enviar `SOLICITUD_ETICA.md` y `CONSENTIMIENTO_PARTICIPANTE.md`. Sin aprobación escrita no se continúa.
2. **Verificar el congelamiento** (equipo): con la aprobación en mano y antes de la primera sesión real, ejecutar `python -m mrcd.recording_check verify-freeze docs/via-b/PREREGISTRO_TECNICO.freeze.json`. Debe decir INTACTO. Si el texto o el código cambiaron, redactar una enmienda fechada y congelar de nuevo (nuevo archivo, el anterior se conserva).
3. **Sesión piloto** (equipo, 1 a 2 personas del equipo, no cuentan): probar OBS, encuadre, claqueta y verificación. Los pilotos no entran en el informe.
4. **Reclutamiento** (equipo): al menos 10 hablantes nuevos que cumplan los criterios del protocolo; consentimiento firmado antes de grabar.
5. **Grabación** (equipo): una sesión por persona con `KIT_GRABACION/`.
6. **Verificación** (equipo, el mismo día): `python -m mrcd.recording_check verify ARCHIVO --out-dir RUTA`. Repetir la toma si falla algo corregible.
7. **Corrida del motor** (equipo, opcional): analizar cada grabación con la versión del motor registrada en la hoja de sesión y guardar un resumen por grabación (esquema en `PREREGISTRO_TECNICO.md`, sección 6). Ninguna salida se muestra a nadie como "correcta" ni se usa como etiqueta.
8. **Informe técnico** (equipo): `python -m mrcd.recording_check report --reports ... --out ...`. Se publica solo el informe agregado, con la frase de que no hay evidencia de exactitud.
9. **Cierre de datos** (equipo): borrar los archivos crudos en el plazo del protocolo; conservar solo informes numéricos.

## Responsables

| Rol                       | Responsabilidad                                                                                                          |
| ------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| Equipo (<author 1>, <author 2>)  | Reclutar, obtener consentimientos, grabar, verificar, correr el motor, redactar el informe, custodiar y borrar los datos |
| Asesor (<advisor>) | Aprobar o derivar al comité de ética; ratificar el preregistro técnico; decidir si el kit se ejecuta                     |
| Comité de ética           | Dictamen si el asesor o la universidad lo exigen                                                                         |

## Lista de verificación general

- [ ] Aprobación ética o aval escrito del asesor archivado (fecha: ____)
- [ ] Preregistro técnico congelado (sha256 verificado) antes de la primera sesión real
- [ ] Versión del motor y commit registrados antes de grabar
- [ ] Disco cifrado y carpeta de datos fuera del repositorio (`D:\viab-datos\`, nunca dentro de `<workspace>`)
- [ ] Piloto hecho y descartado del informe
- [ ] >= 10 hablantes nuevos con consentimiento firmado
- [ ] Cada archivo verificado; tomas fallidas repetidas o declaradas como excluidas con motivo
- [ ] Informe técnico generado sin editar a mano ninguna cifra
- [ ] Datos crudos borrados en el plazo; constancia en la hoja de custodia

## Línea de tiempo realista

Los tiempos son estimaciones del equipo de preparación; la espera de la aprobación ética es la incertidumbre dominante y no se puede acotar desde aquí.

| Semana           | Actividad                                                                       |
| ---------------- | ------------------------------------------------------------------------------- |
| 0                | Entregar solicitud y consentimiento al asesor                                   |
| 1 a 3 (variable) | Espera de aprobación; mientras tanto, piloto interno sin participantes externos |
| 3 a 4            | Congelar preregistro, reclutar y agendar 10 a 12 sesiones                       |
| 4 a 6            | Sesiones (de 30 a 45 min cada una) y verificación el mismo día                  |
| 6 a 7            | Corrida del motor e informe técnico                                             |
| 7 a 8            | Revisión con el asesor; borrado de datos crudos al cierre                       |

Si el calendario de la tesis no admite 8 semanas contando la aprobación, la conclusión honesta es no ejecutar el kit y declarar el canal de video como no evaluable (acción 4.4 del plan).
