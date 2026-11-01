# Ablación con test independiente (juez multimodal)

> El conjunto de prueba fue etiquetado por un VLM a ciegas, sin acceso a las heurísticas. El acuerdo juez–humano se reporta en `results/judge_agreement.md` y condiciona la lectura de esta tabla.

## Ablación multimodal — entrenamiento: **auto** · test: **gold_llm**

- Entrenamiento con etiquetas 'auto' (supervisión débil) y evaluación con etiquetas 'gold_llm' independientes; GroupKFold por hablante (sin fuga) (k=5, unidades=18, hablantes=18)
- IC 95 %: bootstrap de clúster por hablante (B=1000)
- Ventanas: 499

| Clase | Soporte | audio_only | audio_text | trimodal | kinesic_only |
|---|---|---|---|---|---|
| fluent | 170 | 0.433 [0.298, 0.552] | 0.594 [0.471, 0.705] | 0.617 [0.512, 0.715] | 0.426 [0.356, 0.510] |
| filler_word | 46 | 0.204 [0.094, 0.296] | 0.424 [0.212, 0.568] | 0.432 [0.260, 0.559] | 0.114 [0.055, 0.165] |
| prolongation | 12 | 0.043 [0.000, 0.128] | 0.077 [0.000, 0.163] | 0.053 [0.000, 0.143] | 0.000 [0.000, 0.000] |
| repetition | 62 | 0.104 [0.042, 0.189] | 0.568 [0.436, 0.640] | 0.537 [0.403, 0.629] | 0.067 [0.012, 0.147] |
| block | 4 | sin datos suficientes | sin datos suficientes | sin datos suficientes | sin datos suficientes |
| revision | 20 | 0.097 [0.000, 0.179] | 0.148 [0.059, 0.250] | 0.167 [0.053, 0.304] | 0.000 [0.000, 0.000] |
| rhetorical_pause | 25 | 0.133 [0.043, 0.205] | 0.222 [0.091, 0.347] | 0.091 [0.000, 0.218] | 0.092 [0.000, 0.165] |
| neutral_pause | 160 | 0.559 [0.402, 0.651] | 0.587 [0.468, 0.668] | 0.559 [0.477, 0.605] | 0.234 [0.118, 0.320] |
| **macro-F1 (disfluencias)** | | 0.190 [0.144, 0.224] | 0.338 [0.271, 0.382] | 0.306 [0.258, 0.346] | 0.085 [0.053, 0.110] |

**Diferencias pareadas de macro-F1 (IC 95 %)**

- trimodal-audio_only: [+0.069, +0.166]
- audio_text-audio_only: [+0.095, +0.183]
- trimodal-audio_text: [-0.057, +0.014]
- trimodal-kinesic_only: [+0.166, +0.274]
- kinesic_only-audio_only: [-0.157, -0.048]
