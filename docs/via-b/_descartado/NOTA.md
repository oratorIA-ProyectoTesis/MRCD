# Material descartado (no forma parte del flujo)

**Decisión final del equipo (01/11/2026): nadie anota nada** (ni dos personas ni una). El material de esta carpeta se escribió antes de esa decisión, bajo las instrucciones de una Vía B con doble anotación humana y una evaluación confirmatoria con F1. Se conserva por si el equipo cambia de decisión; **no se mantiene, no se prueba y no se debe usar para afirmar resultados**.

## Qué hay

| Ruta                                                      | Qué es                                                                                                                                           | Estado                                                                                                                                                                   |
| --------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `annotation_eval/eaf.py`, `schema.py`                     | Plantilla y validador de archivos ELAN (EAF 3.0): pistas por anotador y clase, vocabularios controlados, solapes, fronteras, cobertura declarada | Escrito, **sin pruebas ejecutadas**; nunca abierto en ELAN                                                                                                    |
| `annotation_eval/importer.py`                             | Importa EAF al esquema `mrcd.corpora` con procedencia humana y cobertura por clase y grabación                                                   | Necesitaba una extensión de `ALLOWED_MAPPINGS` en `mrcd/corpora/base.py` (el tipo de fuente `via_b_elan_tag`); **esa extensión se revirtió** y no está en el repositorio |
| `annotation_eval/agreement.py`, `adjudication.py`         | Kappa de Cohen, alfa de Krippendorff, IoU y F1 entre anotadores con IC por hablante; adjudicación                                                | Escrito                                                                                                                                                                  |
| `annotation_eval/evaluate.py`, `config.py`                | Evaluación confirmatoria con F1 por evento, Holm y ablación A+T frente a A+T+V                                                                   | Escrito, **sin pruebas propias**; sustituido por `mrcd/recording_check/`                                                                                                 |
| `annotation_eval/power.py`, `cli.py`, `metrics.py`        | Cálculo de tamaño, línea de comandos y utilidades                                                                                                | Escrito; `metrics.py` fue la base de `mrcd/recording_check/stats.py`                                                                                                     |
| `tests/test_via_b_agreement.py`, `tests/via_b_helpers.py` | Pruebas de kappa, alfa, IoU, Holm, bootstrap y acuerdo                                                                                           | Se ejecutaron solo hasta la primera falla: 4 pasaron y 1 falló (tolerancia de punto flotante en `segment_positive`); no se corrigió ni se repitió la batería                                                                                                                       |

## Por qué no se usa

Sin etiquetas humanas no hay evaluación de exactitud posible, y la decisión del equipo es no crearlas. Las salidas del motor, de reglas o de modelos no pueden servir de referencia ni para preanotar.

## Si se retomara

1. Restituir `mrcd/annotation_eval/` desde esta carpeta y la extensión de `ALLOWED_MAPPINGS`, con revisión.
2. Escribir las pruebas del evaluador y del importador, y abrir una plantilla en ELAN real: no se hizo.
3. Elegir de nuevo cuántos anotadores habrá y redactar la guía de anotación (no llegó a escribirse), el preregistro confirmatorio y el consentimiento adecuados a la decisión.
4. Rehacer el tamaño de muestra con las prevalencias del corpus (las estimaciones usadas, a partir de las marcas de DIHANA y AMERESCO, eran aproximadas: dependían de la duración media de turno de DIHANA, que no está medida en el repositorio).

Los archivos `.py` de esta carpeta no son importables tal como están (importan `mrcd.annotation_eval`, que ya no existe) y `conftest.py` impide que `pytest` los recoja.
