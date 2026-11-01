# Sensibilidad funcional de `audio_text` v1

> Baseline congelado: se reutilizan las predicciones existentes; no hubo reentrenamiento ni modificación de probabilidades.

## Matriz binaria de producto: disfluencia vs. fluente

| juez \ modelo | disfluencia | fluente/no disfluencia |
|---|---|---|
| disfluencia | 12 | 11 |
| fluente | 16 | 5 |

- Sensibilidad: **52.2%** (12/23).
- Precisión positiva: **42.9%** (12/28).
- Especificidad: **23.8%** (5/21).
- Exactitud: **38.6%** (17/44).

La clase positiva agrupa `filler_word`, `repetition`, `revision`, `prolongation` y `block`. Se excluyen las 16 ventanas cuyo juez indicó pausa.

## Familia solicitada: léxica/repetición vs. fluente

| juez \ modelo | filler/repetition/revision | fluente/otra |
|---|---|---|
| filler/repetition/revision | 10 | 13 |
| fluente | 8 | 13 |

- Sensibilidad: **43.5%** (10/23).
- Exactitud: **52.3%** (23/44).

## Pausas

| juez \ modelo | neutral_pause | rhetorical_pause | fuera de familia |
|---|---|---|---|
| neutral_pause | 4 | 2 | 9 |
| rhetorical_pause | 0 | 0 | 1 |

- Acierto fino de pausa: **4/16 (25.0%)**.

## Diagnóstico `revision` vs. `filler_word`

- `filler_word` juzgados como `revision`: **3/15**.
- Todas las predicciones `revision`: **3**; ninguna coincidió con una revisión adjudicada.
- Solapamiento literal de léxicos: `o sea`.
- En el corpus, `o sea` originó **13/33** eventos débiles de `revision` (39.4%).
- `o sea` activa simultáneamente `is_filler` e `is_edit_term`. Además, la heurística etiqueta revisión cuando encuentra un término de edición tras hasta tres palabras sin puntuación fuerte; no verifica semánticamente que exista reparación.
- `bueno` no pertenece a `EDIT_TERMS`. Sus errores como `revision` son una correlación aprendida de la supervisión débil/contexto, no una regla directa.

- Este baseline no consumía embeddings RoBERTa: su entrada lingüística era el vector manual de 10 rasgos más `zone` y `dangling` (12 dimensiones).

### Casos

| t | texto | p(revision) |
|---|---|---|
| [00:16](https://www.youtube.com/watch?v=Ka_okSSytes&t=16s) | [text withheld] | 0.9876 |
| [03:24](https://www.youtube.com/watch?v=Ka_okSSytes&t=204s) | [text withheld] | 0.5988 |
| [00:24](https://www.youtube.com/watch?v=Ka_okSSytes&t=24s) | [text withheld] | 0.5645 |
