# Estudio de caso — Entrevista_Desing_Thinking

> **PRUEBA DE HUMO — etiquetas heurísticas, NO válida como evidencia**
>
> Los checkpoints se entrenaron con etiquetas heurísticas (supervisión débil).
> Las predicciones de abajo sirven para inspeccionar el comportamiento del
> pipeline. No son evidencia de exactitud: eso exige el conjunto de prueba
> con etiquetas independientes y su validación humana.

- Vídeo: https://www.youtube.com/watch?v=Ka_okSSytes
- Duración: 12.7 min · 1514 ventanas (3.0 s, paso 0.5 s)
- Etiquetas de entrenamiento: `auto` (18927 ventanas)
- Rostro seguido presente en 100% de los frames · margen de dominancia 1.00

## a) Eventos detectados

Ventanas contiguas de la misma clase fundidas en un evento. Con paso de 0.5 s y ventana de 3.0 s cada instante cae en unas seis ventanas, así que contar ventanas multiplicaría los eventos por seis.

| clase | audio_only | audio_text | trimodal |
|---|---|---|---|
| filler_word | 136 | 49 | 52 |
| prolongation | 90 | 95 | 93 |
| repetition | 46 | 16 | 24 |
| block | 15 | 14 | 31 |
| revision | 10 | 27 | 24 |
| rhetorical_pause | 85 | 70 | 57 |
| neutral_pause | 136 | 129 | 121 |
| **total** | **518** | **400** | **402** |

Tasa por minuto (trimodal): **31.7** eventos/min.

## c) Discrepancia entre modos

| par de modos | ventanas discrepantes | % del total |
|---|---|---|
| audio_only vs audio_text | 989 | 65.3% |
| audio_text vs trimodal | 482 | 31.8% |
| audio_only vs trimodal | 996 | 65.8% |

Ventanas donde los tres modos coinciden: **25.4%**.

### Pausas naturales clasificadas como disfluencia

Una pausa con silencio largo y reinicio tonal claro es, prosódicamente, una frontera de discurso. Si el modelo la marca como disfluencia de suspensión (bloqueo, prolongación), es un falso positivo probable. No es prueba de error —el rasgo no es infalible— pero sí la cola que conviene auditar.

| modo | ventanas sospechosas | % del total |
|---|---|---|
| audio_only | 5 | 0.3% |
| audio_text | 8 | 0.5% |
| trimodal | 8 | 0.5% |

### Reinicio tonal por clase predicha (trimodal)

Comprobación de coherencia interna: las clases de pausa deberían mostrar un reinicio tonal mayor que las de suspensión. Si no se separan, el modelo no está usando el rasgo.

| clase predicha | ventanas | f0_reset mediano | rango intercuartílico |
|---|---|---|---|
| neutral_pause | 404 | +0.00 | +0.00 … +0.00 |
| prolongation | 302 | +0.00 | -1.22 … +0.00 |
| filler_word | 182 | +0.00 | +0.00 … +0.80 |
| revision | 168 | +0.00 | +0.00 … +1.16 |
| fluent | 166 | +0.00 | +0.00 … +0.37 |
| rhetorical_pause | 146 | +0.00 | +0.00 … +0.00 |
| repetition | 98 | +0.00 | +0.00 … +0.00 |
| block | 48 | +1.47 | +0.00 … +5.59 |

## b) ¿Corrige el trimodal al bimodal?

Adjudicadas **60** ventanas discrepantes por el juez a ciegas.

> De ellas, **2** se resolvieron SIN fotogramas porque el modelo rechazó las imágenes. Quedan **excluidas** del recuento: un árbitro que no vio vídeo no puede decidir si el vídeo aporta. Aparecen aparte más abajo.

Recuento sobre las **58** que sí vieron fotogramas:

| modo | aciertos | % de las adjudicadas |
|---|---|---|
| audio_only | 22 | 37.9% |
| audio_text | 12 | 20.7% |
| trimodal | 21 | 36.2% |

Ventanas donde ningún modo acertó o acertaron todos: 21.

Sólo como referencia, el recuento de las que no vieron fotogramas (mide el aporte del texto, no el del vídeo):

| modo | aciertos | %  |
|---|---|---|
| audio_only | 0 | 0.0% |
| audio_text | 0 | 0.0% |
| trimodal | 1 | 50.0% |

### Coincidencias de `audio_text` con el juez

Desglose completo de las ventanas adjudicadas en que el modo bimodal coincidió con el juez. Es una línea base descriptiva, no evidencia de generalización fuera de esta cola.

**12** coincidencias de `audio_text`:

| t | juez | transcripción | p(audio_text) | f0_reset | colgante |
|---|---|---|---|---|---|
| [00:03](https://www.youtube.com/watch?v=Ka_okSSytes&t=3s) | filler_word | justamente, Daniel, eh, ¿podréis pre… | 0.846 | +5.18 | no |
| [00:04](https://www.youtube.com/watch?v=Ka_okSSytes&t=4s) | filler_word | Daniel, eh, ¿podréis presentarte por… | 0.730 | +5.18 | no |
| [00:09](https://www.youtube.com/watch?v=Ka_okSSytes&t=9s) | neutral_pause | y... ¿Qué te, a qué te dedicas? | 0.897 | -21.81 | no |
| [04:23](https://www.youtube.com/watch?v=Ka_okSSytes&t=263s) | filler_word | más cómodo. O bueno, eso es lo que c… | 0.425 | +8.38 | no |
| [04:27](https://www.youtube.com/watch?v=Ka_okSSytes&t=267s) | fluent | uno de mis grandes debilidades | 0.516 | +4.73 | no |
| [04:28](https://www.youtube.com/watch?v=Ka_okSSytes&t=268s) | fluent | uno de mis grandes debilidades podrí… | 0.534 | +5.62 | no |
| [04:55](https://www.youtube.com/watch?v=Ka_okSSytes&t=295s) | neutral_pause | genial. ¿Y cómo te preparas antes de… | 0.562 | +9.65 | no |
| [04:55](https://www.youtube.com/watch?v=Ka_okSSytes&t=295s) | neutral_pause | ¿Y cómo te preparas antes de realiza… | 0.905 | +9.65 | no |
| [04:58](https://www.youtube.com/watch?v=Ka_okSSytes&t=298s) | neutral_pause | exposición, digamos, a algún | 0.410 | +9.44 | no |
| [07:59](https://www.youtube.com/watch?v=Ka_okSSytes&t=479s) | repetition | pero pero | 0.938 | +4.94 | no |
| [08:00](https://www.youtube.com/watch?v=Ka_okSSytes&t=480s) | repetition | pero | 0.858 | +4.94 | no |
| [12:33](https://www.youtube.com/watch?v=Ka_okSSytes&t=753s) | filler_word | todas esas preguntas, y bueno, graci… | 0.474 | +5.38 | no |

**Contraste pareado (McNemar).** Los tres modos se evalúan sobre las mismas ventanas, así que comparar porcentajes sueltos no basta. McNemar mira sólo los pares en que un modo acierta y el otro falla.

| comparación | a favor del 1º | a favor del 2º | p | ¿significativa? |
|---|---|---|---|---|
| trimodal vs audio_text | 21 | 12 | 0.163 | no |
| trimodal vs audio_only | 8 | 9 | 1.000 | no |
| audio_only vs audio_text | 17 | 7 | 0.064 | no |

Ninguna diferencia entre modos es estadísticamente significativa en esta muestra. La diferencia aparente de aciertos cabe dentro del azar.

> **Sesgo de la cola, a tener en cuenta al leer lo anterior.** Estas ventanas no son una muestra aleatoria de las discrepancias: se ordenaron por fuerza de evidencia (`|f0_reset| + 4 × tensión labial`) y se cortaron por arriba. Es decir, están escogidas entre las que más favorecen al modo trimodal. Un resultado favorable al trimodal aquí NO se generaliza; uno desfavorable pesa más de lo que sugiere su tamaño.

### Casos donde el trimodal acertó y el bimodal no (veredicto del juez)

| t | transcripción | f0_reset | Δ tensión labial | bimodal dijo | juez dijo |
|---|---|---|---|---|---|
| [09:57](https://www.youtube.com/watch?v=Ka_okSSytes&t=597s) | cuando tienes que realiciar alguna expos… | +12.38 | 0.083 | filler_word | fluent |
| [09:56](https://www.youtube.com/watch?v=Ka_okSSytes&t=596s) | sientes cuando tienes que realiciar algu… | +12.38 | 0.077 | prolongation | fluent |
| [00:16](https://www.youtube.com/watch?v=Ka_okSSytes&t=16s) | bueno, ehh, | +7.67 | 0.615 | revision | filler_word |
| [04:00](https://www.youtube.com/watch?v=Ka_okSSytes&t=240s) | por ahí de repente te intimidan o te | +8.39 | 0.268 | repetition | fluent |
| [07:49](https://www.youtube.com/watch?v=Ka_okSSytes&t=469s) | bueno no en un trabajo del trabajo. | +6.80 | 0.242 | neutral_pause | filler_word |
| [00:24](https://www.youtube.com/watch?v=Ka_okSSytes&t=24s) | tengo años y bueno me | +7.28 | 0.056 | revision | filler_word |
| [02:55](https://www.youtube.com/watch?v=Ka_okSSytes&t=175s) | que el motivo que podría | +5.47 | 0.402 | block | neutral_pause |
| [07:19](https://www.youtube.com/watch?v=Ka_okSSytes&t=439s) | bien ya contaste cuál fue tu peor experi… | +6.59 | 0.088 | prolongation | fluent |
| [00:35](https://www.youtube.com/watch?v=Ka_okSSytes&t=35s) | bueno eso sería el problema. Perfecto. | +5.33 | 0.312 | filler_word | neutral_pause |
| [00:36](https://www.youtube.com/watch?v=Ka_okSSytes&t=36s) | sería el problema. Perfecto. | +5.33 | 0.299 | block | neutral_pause |

## Acuerdo entre jueces (GPT-4o vs Gemini)

Se comparan solo ventanas con clasificación válida y `n_frames > 0` en ambos jueces. Los juicios sin fotogramas quedan fuera del conjunto comparable.

- Ventanas comparadas: **14**
- Acuerdo bruto: **57.1%**
- Kappa de Cohen: **0.488**

Matriz de confusión (filas GPT-4o, columnas Gemini):

| GPT-4o \ Gemini | filler_word | prolongation | repetition | block | revision | rhetorical_pause | neutral_pause | fluent |
|---|---|---|---|---|---|---|---|---|
| filler_word | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| prolongation | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| repetition | 0 | 0 | 1 | 0 | 2 | 0 | 0 | 0 |
| block | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| revision | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| rhetorical_pause | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| neutral_pause | 0 | 0 | 0 | 0 | 1 | 1 | 1 | 1 |
| fluent | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 3 |

## Acuerdo entre jueces (GPT-4o vs Ollama local)

Se comparan solo ventanas con clasificación válida y `n_frames > 0` en ambos jueces. Los juicios sin fotogramas quedan fuera del conjunto comparable.

- Ventanas comparadas: **58**
- Acuerdo bruto: **19.0%**
- Kappa de Cohen: **-0.028**

Matriz de confusión (filas GPT-4o, columnas Ollama local):

| GPT-4o \ Ollama local | filler_word | prolongation | repetition | block | revision | rhetorical_pause | neutral_pause | fluent |
|---|---|---|---|---|---|---|---|---|
| filler_word | 2 | 0 | 0 | 1 | 1 | 2 | 8 | 1 |
| prolongation | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| repetition | 0 | 0 | 0 | 0 | 0 | 0 | 4 | 1 |
| block | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| revision | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 1 |
| rhetorical_pause | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 |
| neutral_pause | 0 | 0 | 3 | 1 | 0 | 1 | 7 | 3 |
| fluent | 0 | 0 | 1 | 2 | 0 | 3 | 12 | 2 |

> **Advertencia crítica:** el acuerdo entre jueces es bajo; este gold set no es confiable como ground truth.

## Qué falta para que esto sea citable

1. Adjudicar la cola con el juez multimodal (apartado b).
2. Validar al juez contra anotación humana (`data/judge/human_check.csv`). Sin el kappa juez-humano, el juez es un árbitro sin credenciales.
3. Reentrenar los checkpoints con etiquetas limpias: los actuales salen de heurísticas cuya tasa de error en `prolongation` ronda el 85 % según el propio juez.

