# Integración del avance MRCD

Fecha de revisión: 30 de octubre de 2026. Este documento distingue cambios del motor, experimentos auxiliares y evidencia aún pendiente.

## Decisión de integración

El motor de esta carpeta conserva su API y su clasificación de siete categorías. El paquete `<plan-package>/MRCD_Plan_Avance_20261030` trabaja principalmente con **presencia binaria de la notación FPAUSE por segmento**. Esa etiqueta no equivale a una clase funcional, a una frontera humana de evento ni a ausencia de todas las disfluencias. No se copian sus pesos o umbrales al clasificador multicategoría.

La variante experimental v4 de `<LEGACY_REPO_ROOT>\experiments\epic_f1_v4_20261029` reporta F1 positivo de 73,68 % sobre 41 segmentos externos, frente a 52,63 % de su v2. Los siete positivos corresponden a un solo hablante. La comparación con una regla ASR estrecha es cercana (F1 positivo 72,73 %; macro-F1 84,25 % frente a 82,87 % de v4). Audio/texto y trimodal empatan en 61,54 %. Además, la extracción desde medios todavía registra `PARITY_DIFFERENCE`. Estos resultados se conservan como experimento binario de EPIC, sin presentarlos como mejora validada del MRCD de siete clases o como aporte demostrado del video.

## Mejoras incorporadas al motor actual

| Área | Cambio | Alcance de la comprobación |
| --- | --- | --- |
| Inferencia de archivo | La señal acústica y los rasgos lingüísticos se extraen una vez; las ventanas se construyen conjuntamente. Se mantiene `bs=1` para la inferencia. | Pruebas con extractores simulados comprueban paridad de tensores y predicciones respecto al camino anterior. Falta medir latencia y paridad con un checkpoint real. |
| Contrato de entrada | Se rechazan arreglos no monofónicos, valores no finitos, WAV distintos de PCM16 y modos que requieren video en la API que solo recibe audio. | Pruebas de errores y formatos. No se afirma soporte audiovisual en `analyze`. |
| Eventos | `Event.text` deja de depender de `keep_windows`. | Prueba de equivalencia de eventos con y sin ventanas de depuración. |
| Referencia humana | `label_source="human"` exige cobertura completa declarada y procedencia no vacía. Un tier ausente, etiqueta desconocida o intervalo inválido se rechaza. | Pruebas de cobertura y lectura EAF. La declaración de procedencia debe comprobarse contra los documentos del corpus. |

Para registros con anotación humana completa, el manifiesto debe incluir `human_annotation: {"coverage": "complete", "provenance": "fuente o registro verificable"}`. Un tier vacío puede significar un negativo solo cuando esa cobertura está documentada. Las anotaciones parciales o desconocidas no se convierten en ventanas `fluent`.

## Camino para medir una mejora de F1

1. **Definir unidad y referencia:** conservar las etiquetas originales de corpus publicados, su licencia, cobertura, tiempos y procedencia. Mapear únicamente categorías equivalentes. La restricción del proyecto es reutilizar anotaciones existentes, sin crear una nueva campaña manual.
2. **Congelar comparadores:** guardar código, dependencias, rasgos, particiones por hablante, umbrales y predicciones de reglas, audio, texto, audio/texto y video. Seleccionar hiperparámetros solo en desarrollo. Los segmentos ya inspeccionados no constituyen una nueva prueba independiente.
3. **Resolver paridad de v4:** ejecutar la extracción del mismo medio en procesos separados, con hashes, versión del encoder, parámetros de hilos y configuración registrados. No promover el bundle mientras persista la diferencia de rasgos.
4. **Conseguir evaluación externa compatible:** buscar unidades publicadas con positivos de varios hablantes, negativos con cobertura acreditada y clases realmente anotadas. Informar soporte por clase, matriz de errores y F1 por hablante además del agregado. Si una fuente solo marca FPAUSE por segmento, limitar la conclusión a esa tarea binaria.
5. **Medir video de forma emparejada:** admitirlo solo cuando correspondencia de hablante, cobertura y sincronía estén verificadas. Comparar las mismas unidades con audio/texto; reportar respaldo y ausencias por motivo. Los umbrales visuales del complemento son parámetros de ingeniería aún no calibrados.

Los intervalos emitidos por el clasificador de ventanas son franjas de decisión de 0,5 s; no deben llamarse fronteras humanas de disfluencia. Las probabilidades softmax tampoco están calibradas. Cualquier mejora de F1 se declarará solo después de una comparación sobre la **misma tarea, referencia y partición**.
