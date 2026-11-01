# Protocolo de evaluación de usabilidad (SUS)

**ESTADO: NO REALIZADO. Borrador listo para aplicar, pendiente de revisión del asesor.** Versión 0.1, 2026-11-01.

## 1. Objetivo

Estimar la usabilidad percibida de la aplicación web al mostrar el reporte de disfluencias del motor, con la escala SUS (Brooke, 1996). Es una medición de usabilidad de la interfaz; **no** mide la exactitud del motor ni la eficacia pedagógica.

## 2. Participantes

- **Tamaño mínimo:** 10 participantes con respuestas completas (indicador original de OE4: SUS >= 70 con >= 10 usuarios). Se recomienda reclutar 12-14 para cubrir pérdidas. Con menos de 10 el indicador no se evalúa.
- **Inclusión:** mayores de 18 años; estudiantes universitarios; hablantes nativos o dominantes de español; que hayan dado su consentimiento informado; que puedan grabar audio con un micrófono y ver un reporte en pantalla.
- **Exclusión:** autores del proyecto, el asesor y personas que participaron en el desarrollo de la aplicación o del motor; quien no acepte el consentimiento; quien haya probado versiones previas de la aplicación.
- **Variedad:** registrar (sin datos personales) universidad y región de procedencia para describir la muestra; no generalizar a "el español".
- **Anonimato:** cada participante recibe un código (`P01`, `P02`, ...). No se guardan nombres, correos ni audio.

## 3. Materiales y condiciones

- Aplicación con el parche de integración aplicado y el servicio MRCD local en ejecución (comandos en `comandos-ejecutar/COMANDOS.MD`).
- Micrófono y sala silenciosa; navegador y dispositivo estándar; el mismo montaje para todos.
- Decisión de privacidad resuelta (`../contracts/PRIVACY_DECISION.md`) y consentimiento firmado **antes** de grabar.
- Reporte configurado para mostrar solo `filler_word`, `prolongation` (alargamiento observable) y `repetition` (literal); la aplicación muestra "no calibrado" junto a cada puntaje y "no disponible: motivo" cuando falta una medición.

## 4. Tareas en la aplicación

Cada tarea se lee en voz alta; el moderador no ayuda salvo bloqueo de más de 3 minutos (se registra la ayuda).

| # | Tarea | Éxito |
|---|---|---|
| T1 | Iniciar una práctica de habla libre de **al menos 20 s** continuos (MRCD necesita >= 13 s de audio) y finalizarla | La sesión queda guardada |
| T2 | Abrir el reporte de la sesión y localizar la sección de disfluencias | El participante señala la sección |
| T3 | Elegir un evento de la lista y decir su clase, su marca de tiempo y qué significa la leyenda "no calibrado" | Dice clase y tiempo; explica que el puntaje no es una probabilidad |
| T4 | Encontrar un indicador marcado "no disponible" y decir por qué no está disponible | Cita el motivo mostrado |
| T5 | Repetir una práctica de **menos de 13 s** y describir qué muestra el reporte | Reconoce que el reporte indica "no disponible: audio demasiado corto" y no un valor cero |

Si la aplicación no mostrara alguna de estas pantallas, la tarea se registra como **no ejecutable** (no como fallida) y se anota el motivo.

## 5. Procedimiento (aprox. 30-40 min por persona)

1. Bienvenida y lectura del consentimiento (`CONSENTIMIENTO.md`); firma antes de continuar.
2. Explicación breve: se evalúa la aplicación, no al participante. Ninguna respuesta es correcta o incorrecta.
3. Tareas T1 a T5, en el mismo orden, con registro de: éxito (sí/no/no ejecutable), tiempo, ayudas, comentarios espontáneos.
4. Cuestionario SUS (`CUESTIONARIO_SUS_ES.md`) inmediatamente después de las tareas, sin consultar a otros ni a la pantalla.
5. Preguntas abiertas: qué fue lo más confuso; qué cambiaría.
6. Cierre: agradecimiento; recordatorio de que puede retirar sus datos.

## 6. Qué se registra

- `responses.csv` (plantilla en `responses_template.csv`): código, fecha, 10 ítems (1-5), éxito de T1-T5, comentarios.
- Un valor **ausente** se deja vacío; nunca se escribe 0. Una fila incompleta se excluye del SUS y se informa el motivo.
- No se guarda audio, texto de transcripción ni datos personales fuera de lo declarado en el consentimiento.

## 7. Análisis

- `sus_score.py` calcula el SUS de cada participante (ítems impares: respuesta menos 1; pares: 5 menos la respuesta; suma por 2,5) y la media, mediana, desviación y un IC por bootstrap sobre participantes.
- El umbral 70 es el declarado en `OBJETIVOS.MD`; con n < 10 el script informa que el indicador no puede evaluarse. No se redondea ni se descartan datos para alcanzarlo.
- Reportar también éxito por tarea, tiempos y comentarios, y todas las limitaciones: muestra de conveniencia, una universidad o pocas, sin aleatorización, usabilidad percibida.

## 8. Riesgos y límites

- El SUS mide percepción de usabilidad de la interfaz, no la calidad del motor.
- Un puntaje alto no valida las clases; un puntaje bajo puede deberse a la interfaz y no al motor.
- Muestra pequeña y de conveniencia: sin inferencia poblacional.
- La integración depende de decisiones pendientes (autorización del parche, privacidad del audio, aprobación ética).
- El video no se usa en esta evaluación ni se infiere estado psicológico alguno.
