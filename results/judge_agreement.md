# Fiabilidad del juez multimodal

- Juicios: 512 · modelo: gpt-4o
- Kappa heurística vs juez: **0.422** (bajo/moderado = el juez aporta una opinión independiente; muy alto = solo repite las reglas)

## Matriz heurística vs juez

| heurística \ juez | filler_word | prolongation | repetition | block | revision | rhetorical_pause | neutral_pause | fluent |
|---|---|---|---|---|---|---|---|---|
| filler_word | 33 | 1 | 2 | 0 | 0 | 1 | 14 | 9 |
| prolongation | 0 | 12 | 1 | 0 | 0 | 1 | 31 | 51 |
| repetition | 1 | 0 | 51 | 0 | 0 | 1 | 1 | 6 |
| block | 1 | 0 | 0 | 0 | 0 | 4 | 23 | 0 |
| revision | 11 | 0 | 3 | 0 | 8 | 0 | 0 | 11 |
| rhetorical_pause | 1 | 0 | 0 | 1 | 1 | 7 | 50 | 0 |
| neutral_pause | 4 | 0 | 0 | 1 | 0 | 3 | 50 | 0 |
| fluent | 1 | 0 | 3 | 0 | 1 | 3 | 8 | 101 |

## Muestreo negativo (117 ventanas sin candidato heurístico)

- El juez encontró fenómenos en 16 de ellas: {'neutral_pause': 8, 'repetition': 3, 'rhetorical_pause': 3, 'revision': 1, 'filler_word': 1}
- Esto estima lo que las heurísticas NO ven (su recall).

## Validación humana

- El CSV existe pero no tiene etiquetas válidas todavía.

