# Fiabilidad del juez multimodal

- Juicios: 506 · modelo: gpt-4o
- Kappa heurística vs juez: **0.344** (bajo/moderado = el juez aporta una opinión independiente; muy alto = solo repite las reglas)

## Matriz heurística vs juez

| heurística \ juez | filler_word | prolongation | repetition | block | revision | rhetorical_pause | neutral_pause | fluent |
|---|---|---|---|---|---|---|---|---|
| filler_word | 23 | 1 | 1 | 0 | 0 | 1 | 13 | 11 |
| prolongation | 1 | 4 | 7 | 2 | 1 | 1 | 28 | 54 |
| repetition | 1 | 4 | 40 | 0 | 4 | 0 | 2 | 9 |
| block | 1 | 0 | 0 | 1 | 0 | 4 | 19 | 0 |
| revision | 14 | 1 | 6 | 0 | 5 | 0 | 1 | 6 |
| rhetorical_pause | 2 | 0 | 0 | 1 | 0 | 12 | 45 | 0 |
| neutral_pause | 2 | 0 | 2 | 0 | 0 | 6 | 50 | 0 |
| fluent | 3 | 2 | 6 | 0 | 10 | 2 | 6 | 91 |

## Muestreo negativo (120 ventanas sin candidato heurístico)

- El juez encontró fenómenos en 29 de ellas: {'revision': 10, 'neutral_pause': 6, 'rhetorical_pause': 2, 'filler_word': 3, 'repetition': 6, 'prolongation': 2}
- Esto estima lo que las heurísticas NO ven (su recall).

## Validación humana

- **PENDIENTE**: sin ella no se puede afirmar que las etiquetas del juez aproximen la verdad.
- Genera la plantilla: `python scripts/judge_agreement.py --make-template 60`

