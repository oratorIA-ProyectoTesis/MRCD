# Protocolo de grabación (validación técnica, opcional)

**BORRADOR, NO APROBADO.** No se graba a ninguna persona hasta contar con la aprobación descrita en `README.md`. Autorrevisión; sin revisión independiente.

## 1. Objetivo

Medir, con grabaciones propias de hablantes nuevos y video frontal, **propiedades técnicas** del flujo audio-video del motor: admisión de la compuerta de video S6 congelada, sincronía audio-video, calidad de la cadena de audio con micrófono real, y robustez y tiempo de análisis. **No se anota nada** y no se afirma exactitud de ninguna clase (ver `PREREGISTRO_TECNICO.md`).

## 2. Tamaño y precisión esperada

Mínimos fijados antes de grabar (`mrcd/recording_check/config.py`): **10 hablantes nuevos** y **30 min** de habla neta en total; objetivo 45 min. Cada hablante aporta una presentación de 3 a 5 min y una parte espontánea de 2 a 3 min (5 a 8 min por persona; 50 a 80 min de grabación en total, de los cuales se espera al menos el 60 % de habla neta).

Qué precisión permite ese tamaño (aproximación normal con efecto de diseño `1 + (m - 1) * ICC`, `m` = segmentos de 3 s por hablante, ~100; el ICC es la correlación intra-hablante, desconocida):

| Hablantes | Mitad del IC 95 % si la admisión real es 0,80 (ICC 0,1 / 0,3 / 0,5) | Si la admisión real es 0,95 |
| --------- | ------------------------------------------------------------------- | --------------------------- |
| 10        | 0,08 / 0,14 / 0,18                                                  | 0,05 / 0,08 / 0,10          |
| 12        | 0,08 / 0,13 / 0,16                                                  | 0,04 / 0,07 / 0,09          |
| 15        | 0,07 / 0,11 / 0,14                                                  | 0,04 / 0,06 / 0,08          |

Lectura: con 10 hablantes, para que el límite inferior del IC supere 0,80 hace falta que la admisión real sea del orden de 0,92 a 0,97. Con una admisión real de 0,85 el resultado típico será "cumple solo en la estimación puntual". Es una limitación del tamaño, no un error de método. **Si se alcanzan 12 a 15 hablantes sin costo alto, conviene hacerlo.** Si no se alcanzan 10, el informe lo declara y no se concluye sobre T1 a T5.

Esta sección **reemplaza** el criterio de "≥ 40 positivos por clase" del plan original: ese criterio exige etiquetas humanas y queda fuera de este kit.

## 3. Participantes

Inclusión: mayores de 18 años; hispanohablantes con dominio fluido del español; sin relación de dependencia académica con el equipo (no compañeros que dependan del asesor, para evitar presión); capaces de leer y firmar el consentimiento; **hablante nuevo**: no aparece en ningún corpus de desarrollo ni fue usado para ajustar el motor o la compuerta.

Exclusión: personas que declaran **diagnóstico clínico de tartamudez** (se mantiene la población no clínica, que es el alcance del motor); quienes no aceptan que se grabe el rostro; quienes no pueden estar solos frente a la cámara durante la toma (segunda persona visible); miembros del equipo que desarrollaron o ajustaron el motor (solo pueden hacer el piloto).

**Cribado:** una única pregunta verbal, "¿Tienes un diagnóstico clínico de tartamudez?". Solo se anota "elegible: sí/no"; no se registra el motivo ni ningún otro dato de salud. No se pregunta nada más sobre diagnósticos.

Datos que sí se registran: código seudónimo (`VB_S01`...), edad en rango (18-24, 25-34, 35-44, 45+), variedad de español por autodeclaración (país/región, texto libre opcional) y si usa lentes. Nada más.

## 4. Tarea de habla

Instrucciones leídas por el equipo, sin improvisar:

> "Vas a hablar frente a la cámara, como si presentaras un tema a un público. No hace falta que sea perfecto ni que no te equivoques; habla con naturalidad. Puedes detenerte cuando quieras. Primero daré una palmada frente a la cámara, esperaré tres segundos y te haré una seña para empezar."

1. **Presentación (3 a 5 min), `PRES1`.** El participante elige **uno** de seis temas entregados en tarjetas y tiene 2 minutos para pensar (puede tomar notas en papel; no lee un guion): (a) un lugar que recomendarías visitar, (b) un oficio o profesión que admires, (c) cómo se prepara un plato que conoces, (d) una tecnología que cambió tu rutina, (e) un problema de tu ciudad y cómo lo resolverías, (f) un recuerdo de estudio o trabajo. Se detiene por sí mismo; si pasa de 5 min, se le indica.
2. **Parte espontánea (2 a 3 min), `ESP1`.** Sin preparación; el equipo hace una pregunta abierta ("¿Qué hiciste el fin de semana pasado?" u otra de una lista fija) y el participante responde. El equipo no interrumpe. Si interviene, queda audible la voz de una segunda persona; esas partes no afectan a la compuerta pero se anotan en la hoja de sesión.
3. Cierre con una segunda palmada, visible, 3 segundos después de terminar de hablar.

## 5. Condiciones técnicas

| Elemento    | Requisito                                                                                                                                                                                                                           |
| ----------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Micrófono   | Un solo micrófono definido por el equipo (diadema USB o lavalier); se registra el modelo en la hoja. Sin supresión de ruido, ganancia automática ni eco de OBS/Windows (distorsionan la relación labios-audio y lo que ve el motor) |
| Audio       | 48 kHz mono en la grabación (mínimo exigido 16 kHz); pico entre -20 y -1 dBFS; recorte ≤ 0,1 %; piso de ruido ≤ -50 dBFS                                                                                                            |
| Cámara      | Frontal, fija (trípode o monitor), a la altura de los ojos, 60 a 80 cm; **720p o más**, **25 fps** (aceptables 24 a 30; la compuerta S6 se congeló pensando en 25)                                                                  |
| Encuadre    | Rostro completo y frontal (±15°), ocupando al menos un cuarto de la altura de la imagen (rostro ≥ ~150 px); manos visibles al dar la palmada; una sola persona en la imagen; fondo sin pantallas ni espejos con otro rostro         |
| Iluminación | Luz frontal suave; sin contraluz (ventana detrás); exposición manual si la cámara lo permite, para que no cambie la tasa de cuadros                                                                                                 |
| Sincronía   | Claqueta: palmada visible y audible a los 3 a 10 s del inicio y otra tras terminar; la verificación estima el desfase de cada una y la deriva entre ambas                                                                           |
| Ambiente    | Cuarto silencioso, puerta cerrada, sin música ni TV                                                                                                                                                                                 |

Los pasos exactos están en `KIT_GRABACION/README.md`.

## 6. Nomenclatura y seudónimos

- Participante: `VB_S01`, `VB_S02`, ... (asignado en orden de sesión).
- Grabación: `VB_S01_PRES1`, `VB_S01_ESP1` (tarea `PRES` o `ESP` y número de toma). Archivo: `VB_S01_PRES1.mp4`. **Ningún nombre propio en ningún archivo ni carpeta.**
- La tabla que une el código con la persona (nombre, contacto, firma) vive en un archivo cifrado distinto, con una sola persona responsable, y nunca junto a las grabaciones.
- El piloto usa `VB_S00_...` y se excluye del informe.

## 7. Gestión de datos

| Tema                         | Regla                                                                                                                                                                                                                                                         |
| ---------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Dónde                        | Disco o contenedor cifrado (BitLocker o VeraCrypt) en un solo equipo, ruta `D:\viab-datos\` (fuera del repositorio). `*.mp4`, `*.wav` y `*.mkv` ya están ignorados en `.gitignore`, pero la carpeta de datos no debe estar dentro del repositorio             |
| Nube                         | Prohibido subir audio, video o capturas a nubes, chats o correo                                                                                                                                                                                               |
| Quién accede                 | Los dos integrantes del equipo; nadie más                                                                                                                                                                                                                     |
| Qué se conserva              | Informes numéricos (JSON y markdown), sin audio, video ni texto                                                                                                                                                                                               |
| Qué se hace con el contenido | El motor procesa el audio en memoria; no se guardan transcripciones ni cuadros ni puntos faciales (la verificación solo emite números). Nadie escucha ni mira el contenido salvo para diagnosticar una falla técnica puntual, durante el menor tiempo posible |
| Retención                    | Crudos: hasta la entrega del informe técnico más 30 días, y como máximo 6 meses desde la grabación; luego borrado seguro (suprimir el contenedor cifrado)                                                                                                     |
| Retiro                       | El participante puede retirarse hasta antes de generar el informe agregado: borrado de sus archivos en un máximo de 7 días y constancia. Después de agregado, su aporte numérico no se puede separar (se le informa)                                          |
| Publicación                  | Solo cifras agregadas. Nunca video, audio, cuadros ni resultados individuales identificables                                                                                                                                                                  |
| Incidentes                   | Pérdida o acceso indebido: avisar al asesor el mismo día                                                                                                                                                                                                      |

## 8. Cronograma estimado

Ver la tabla de `README.md`. Las sesiones se agendan de 30 a 45 min (5 min de recepción y consentimiento, 5 de montaje y prueba de nivel, 15 de grabación, 5 de verificación inmediata, 5 de margen).

## 9. Qué hacer si no se alcanza el tamaño o fallan los criterios

- Menos de 10 hablantes o menos de 30 min netos: el informe lo declara y no emite decisiones T1 a T5; se reporta lo observado como descriptivo.
- Admisión de la compuerta muy baja (< 0,5) con un montaje conforme a la lista de verificación: se concluye "el canal de video no es evaluable con este montaje y esta compuerta" y se informa al asesor para la reformulación del alcance (acción 4.4). **No** se baja el umbral de la compuerta ni se cambia el criterio después de ver los datos.
- Fallas del motor: se reportan tal cual (estado `failed` o `not_run`), nunca se rellenan ni se cuentan como cero eventos.
