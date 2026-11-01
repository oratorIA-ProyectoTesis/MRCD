# Matriz de trazabilidad — avance técnico, no conformidad académica

| Requisito | Componente | Evidencia ejecutada | Objetivo relacionado | Brecha vigente |
|---|---|---|---|---|
| Preservar procedencia y significado de etiquetas | `references.py` | 136 etiquetas/textos cotejados con miembros originales; hashes conservados | OE1, OE4 | No equivale a siete clases funcionales |
| Distinguir ausencia de anotación y negativo | `Reference` | Rechazo de negativos con cobertura desconocida en pruebas | OE1, OE2 | Auditar otros importadores del motor local |
| Contrato explícito de decisión e incertidumbre | `SelectivePolicy` | Abstención nula, score sin calibración probabilística y bloqueo por hash/tarea | OE2, OE3 | Integración con `Engine` 0.4.0 pendiente |
| Evaluar incertidumbre sin ocultar rechazos | `execute_plan.py` | Validación agrupada anidada: 102 segmentos, 16 IDs; cobertura, matrices, recuperación total e intervalos | OE4 | Resultado retrospectivo; beneficio insuficiente para promoción |
| Retirar evidencia visual no verificada | `VisualPolicy`, `GuardedHead` | 136 respaldos exactos de audio/texto; controles sintéticos de cortes, offset, deriva y ambigüedad | OE2, OE3 | No hay verificador de hablante activo/sincronía nuevo ni beneficio visual acreditado |
| Preparar contexto sin confundir candidato y disfluencia | `context.py` | Pruebas de candidatos, límites, contexto externo y procedencia ASR | OE2, OE3 | Falta clasificador contextual y referencia compatible de función |
| Detectar incompatibilidad de extracción | `parity.py` | 136 pares de caché numéricamente iguales, todos marcados con procedencia no verificable | OE3, OE4 | Repetir extracción real de v4; no cerrar `PARITY_DIFFERENCE` |
| Instalación e integración portable | Rueda y CLI | Verificación fuera del árbol de desarrollo en `results/installation_smoke.json` | OE3 | No mide decodificación, ASR, seguimiento ni rendimiento completo |
| Importar corpus sin convertir ausencia en negativo | `mrcd/corpora` (DIME, ESLORA, DIHANA) | Guarda de mapeos prohibidos normalizada y validada a nivel de Example; pruebas con variantes; datos reales importados sin mapeos prohibidos | OE1, OE4 | Revisión independiente de estas correcciones pendiente; ESLORA sin audio; DIME sin etiquetas de reparación (403) |
| Comparar contexto frente a reglas y audio sobre referencia existente | `mrcd/context_eval`, `scripts/rep_rev` | DIHANA 1-4, validación anidada por hablante: repetición B ctx_LR 0,537 frente a regla 0,541 (sin diferencia); revisión B 0,286 frente a 0,146 (en gran parte largo del turno); exploratorio | OE4 | Un solo corpus, condición A circular, sin partición de prueba independiente; no confirmatorio |
| Paridad en vivo con el cálculo offline | `mrcd/live`, `live_cfg_v1/v2/v3` | 12 sesiones nuevas × 3 particiones: v2 difiere en 7/12 (IC 95 % [27,7; 84,8] %), v3 en 1/12 ([0,2; 38,5] %); causa aislada y corregida en v3 | OE3 | Sesiones sintéticas, sin micrófono real ni ASR en streaming equivalente; v3 no es el valor por defecto |
| Recuperación de repeticiones por decodificación del ASR (EXPLORATORIO) | `mrcd/asr_recovery` | DIHANA 1-4: prompt disfluente P1 supervivencia 105→144 de 214, F1 ponderado ctx_LR 0,581→0,690; control fluido C1 no lo reproduce (0,304; qué rasgo del prompt importa no está probado, P0 ya es un prompt con muletillas y no hay control sin prompt); O2/O3 sin recuperación | OE4 | Un solo corpus y modelo, exploratorio; no cierra OE4 |
| Clases sin referencia (bloqueo, pausas) | motor actual | Medición técnica de sensibilidad al relleno de cola, duración y contexto (240 turnos DIHANA 1-3) | OE3, OE4 | No es validez ni F1; falta referencia compatible |

## OE4 propuesto para ratificación

Validar técnicamente el motor de reconocimiento contextual mediante experimentos reproducibles sobre corpus de habla en español con etiquetas preexistentes compatibles, comparándolo con líneas base y ablaciones de las modalidades mediante métricas de detección, cobertura, robustez y costo de procesamiento, según las unidades y categorías respaldadas por las fuentes.

No se incluye creación, doble anotación, adjudicación o reclutamiento de anotadores. La propuesta no acredita aprobación del asesor ni altera el título vigente. La localización temporal y las siete categorías solo podrán declararse evaluadas con evidencia compatible; esta tanda no las acredita.

La evidencia para OE4 que sí se agrega es la auditoría de etiquetas, los comparadores, la validación agrupada, el análisis de abstenciones, las limitaciones y los controles reproducibles del software. No constituye un acta firmada ni el cierre del objetivo.

Actualización 31/10/2026: las filas anteriores sobre importadores, comparación contextual, paridad en vivo y clases sin referencia son evidencia técnica exploratoria del motor mrcd. OE4 NO se marca como completo: faltan una partición de prueba independiente con referencia compatible, el desempeño por clase para bloqueo y pausas, y la ratificación del asesor.
