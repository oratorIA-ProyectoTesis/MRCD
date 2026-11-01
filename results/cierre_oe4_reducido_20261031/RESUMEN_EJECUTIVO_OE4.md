# Resumen ejecutivo: OE4 reducido (propuesta)

**Versión 1.0, 2026-10-31. Estado: PENDIENTE de ratificación del asesor.** Documento preparado por los autores; es una autorrevisión, sin revisión independiente. Nada se ha firmado, enviado ni aprobado.

## Declaración de cumplimiento

> **OE4 reducido: paquete de evidencia completo con el alcance propuesto, pendiente de ratificación del asesor; NO cumple los indicadores originales de OBJETIVOS.MD.**

## Qué se propone

El 2026-10-31 el equipo decidió retirar del alcance evaluado de OE4 las clases sin referencia humana compatible (`block`, `rhetorical_pause`, `neutral_pause`) y completar OE4 con la evidencia disponible. **Es una reducción de alcance y requiere la ratificación del asesor**; los autores no pueden afirmar que fue aprobada. El código de esas clases no se borra: quedan como "fuera del alcance evaluado, sin referencia, salida técnica no validada".

Clases evaluadas: pausa llena (`filler_word`); alargamiento observable (`prolongation`, sin demostrar que sea disfluencia); repetición literal; revisión solo como aproximación NO confirmada, sin métricas de exactitud afirmadas. Unidad: turno o segmento, no eventos con fronteras humanas. Sin anotación humana nueva.

## Qué se midió (todo exploratorio)

- Corpus con etiquetas preexistentes: DIHANA (225 hablantes, 5 particiones por hablante; el fold 5 es histórico, ya visto dos veces), AMERESCO (4 conversaciones), EPIC (desarrollo), disfluentES y ES-Port (solo texto).
- Mejores F1 con audio y ASR del motor, frente al umbral original de 0,80: pausa llena 0,51 a 0,58 (faltan 0,22 a 0,29); prolongación 0,54 a 0,63 (faltan 0,17 a 0,26); repetición 0,35 a 0,54 (faltan 0,26 a 0,45); revisión 0,09 a 0,29 (faltan 0,51 a 0,71). Los intervalos superiores no llegan a 0,80.
- El motor no supera a una regla léxica de una línea sobre el texto del ASR en pausa llena ni en repetición. En prolongación sí supera a las líneas base simples en DIHANA, pero no a "siempre presente" en AMERESCO.
- Robustez: determinismo condicionado a fijar `OMP_NUM_THREADS=4` antes de numpy y sin carga concurrente; paridad offline exacta; paridad en vivo NO exacta (v2 difiere en 7/12 sesiones sintéticas, v3 en 1/12).
- Costo (fase 3b, CPU): factor de tiempo real de 0,45 a 0,51 en la ruta por defecto para audios de 17 a 60 s. Streaming y WER del ASR en streaming: pendientes de 3b.
- Video: no evaluable (sin identidad ni sincronía verificadas). Ablación existente: solo frente a un juez automático con 18 unidades; no cuenta como evidencia.
- Incertidumbre: no existe abstención calibrada.
- Usabilidad (SUS >= 70, >= 10 usuarios): **no medida**; requiere aplicación operable, consentimiento y usuarios reales. No existe un protocolo local preparado.

## Qué no se cumple

Ninguno de los indicadores originales de OE4: los F1 están por debajo de los umbrales; DM, F1_context, kappa/alfa e IoU no son medibles sin anotación nueva ni fronteras humanas; coverage/risk no es medible sin abstención calibrada; SUS no se midió.

## Qué se pide al asesor

Ver the advisor approval sheet (kept local, not published): ratificar o no la reducción, qué métricas se reportan, eliminar o diferir las clases sin referencia, tratamiento del SUS y uso de los corpus.

Informe técnico completo (en inglés): `INFORME_OE4_REDUCIDO.md`.
