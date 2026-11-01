# Diagnostico de sesion

- Audio: `mrcd_sesion_1790263526.wav` (90.8 s)
- Checkpoint: `ckpt_audio_text.pt` (modo audio_text)
- Whisper: `small` (VerbatimASR del pipeline)

> **PRUEBA DE HUMO — etiquetas heurísticas, NO válida como evidencia**

## Test de memorizacion lexica

Cada ventana reevaluada con la **acustica intacta** y el texto sustituido por
palabras neutras que conservan los tiempos originales. La unica variable que
cambia es la identidad lexica.

- Si la clase **se mantiene**, la decision venia de la senal acustica/temporal.
- Si **colapsa**, el modelo estaba reconociendo palabras, no escuchando.

Importa porque la cascada heuristica que genero las etiquetas de entrenamiento
decide por lexico (`FILLER_LEXICON`, `EDIT_TERMS`): el modelo pudo aprender la
regla en vez del fenomeno.

| t | texto original | clase real | conf | clase sin lexico | conf | sobrevive |
|---|---|---|---|---|---|---|
| 8.0 | — | prolongation | 43% | prolongation | 43% | si |
| 12.0 | — | neutral_pause | 94% | neutral_pause | 94% | si |
| 14.1 | — | neutral_pause | 93% | neutral_pause | 93% | si |
| 16.2 | [text withheld] | neutral_pause | 68% | neutral_pause | 81% | si |
| 18.5 | [text withheld] | rhetorical_pause | 67% | neutral_pause | 47% | **NO** |
| 20.8 | [text withheld] | neutral_pause | 72% | neutral_pause | 74% | si |
| 23.4 | [text withheld] | prolongation | 51% | fluent | 79% | **NO** |
| 26.0 | [text withheld] | neutral_pause | 73% | neutral_pause | 65% | si |
| 28.6 | [text withheld] | neutral_pause | 90% | neutral_pause | 88% | si |
| 31.4 | [text withheld] | fluent | 94% | fluent | 95% | si |
| 34.1 | [text withheld] | neutral_pause | 89% | neutral_pause | 89% | si |
| 36.6 | [text withheld] | rhetorical_pause | 80% | rhetorical_pause | 80% | si |
| 39.2 | [text withheld] | rhetorical_pause | 80% | rhetorical_pause | 80% | si |
| 41.9 | [text withheld] | prolongation | 89% | prolongation | 88% | si |
| 45.8 | [text withheld] | neutral_pause | 72% | neutral_pause | 72% | si |
| 48.4 | [text withheld] | rhetorical_pause | 56% | rhetorical_pause | 73% | si |
| 51.1 | [text withheld] | neutral_pause | 61% | neutral_pause | 55% | si |
| 53.7 | [text withheld] | prolongation | 42% | prolongation | 39% | si |
| 56.2 | [text withheld] | rhetorical_pause | 54% | rhetorical_pause | 54% | si |
| 58.8 | [text withheld] | neutral_pause | 37% | neutral_pause | 43% | si |
| 61.3 | [text withheld] | prolongation | 92% | prolongation | 83% | si |
| 63.7 | [text withheld] | rhetorical_pause | 48% | rhetorical_pause | 52% | si |
| 66.3 | [text withheld] | rhetorical_pause | 60% | rhetorical_pause | 64% | si |
| 68.8 | [text withheld] | rhetorical_pause | 97% | rhetorical_pause | 82% | si |
| 71.8 | [text withheld] | filler_word | 98% | fluent | 52% | **NO** |
| 74.4 | [text withheld] | fluent | 57% | fluent | 84% | si |
| 77.0 | [text withheld] | neutral_pause | 73% | neutral_pause | 80% | si |
| 79.5 | [text withheld] | neutral_pause | 79% | neutral_pause | 78% | si |
| 82.1 | [text withheld] | rhetorical_pause | 81% | rhetorical_pause | 85% | si |
| 84.6 | [text withheld] | neutral_pause | 77% | neutral_pause | 67% | si |

Clases que sobreviven sin lexico: **27/30** (90%)

### Dependencia lexica por clase

| clase | ventanas | sobreviven | lectura |
|---|---|---|---|
| neutral_pause | 13 | 13/13 (100%) | acustica |
| rhetorical_pause | 9 | 8/9 (89%) | acustica |
| prolongation | 5 | 4/5 (80%) | acustica |
| fluent | 2 | 2/2 (100%) | acustica |
| filler_word | 1 | 0/1 (0%) | **lexica** |

Una clase con dependencia **lexica** no esta midiendo el fenomeno acustico
que su definicion describe. Es el patron esperado si el modelo reaprendio la
cascada heuristica en vez de generalizar.

