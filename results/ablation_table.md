# Ablación con test independiente (juez multimodal)

> El conjunto de prueba fue etiquetado por un VLM a ciegas, sin acceso a las heurísticas. El acuerdo juez–humano se reporta en `results/judge_agreement.md` y condiciona la lectura de esta tabla.

## Ablación multimodal — entrenamiento: **auto** · test: **gold_llm**

- Entrenamiento con etiquetas 'auto' (supervisión débil) y evaluación con etiquetas 'gold_llm' independientes; GroupKFold por hablante (sin fuga) (k=2, unidades=18, hablantes=18)
- IC 95 %: bootstrap de clúster por hablante (B=2)
- Ventanas: 503

| Clase | Soporte | audio_only | audio_text | trimodal | kinesic_only |
|---|---|---|---|---|---|
| fluent | 177 | 0.316 [0.379, 0.405] | 0.567 [0.529, 0.647] | 0.565 [0.537, 0.661] | 0.189 [0.122, 0.190] |
| filler_word | 50 | 0.138 [0.094, 0.150] | 0.322 [0.384, 0.411] | 0.340 [0.403, 0.412] | 0.157 [0.190, 0.234] |
| prolongation | 13 | 0.000 [0.000, 0.000] | 0.036 [0.001, 0.037] | 0.044 [0.001, 0.044] | 0.000 [0.000, 0.000] |
| repetition | 57 | 0.087 [0.002, 0.084] | 0.694 [0.633, 0.690] | 0.705 [0.619, 0.702] | 0.000 [0.000, 0.000] |
| block | 2 | sin datos suficientes | sin datos suficientes | sin datos suficientes | sin datos suficientes |
| revision | 10 | 0.033 [0.039, 0.064] | 0.240 [0.225, 0.338] | 0.250 [0.235, 0.347] | 0.065 [0.027, 0.043] |
| rhetorical_pause | 20 | 0.164 [0.097, 0.147] | 0.191 [0.169, 0.179] | 0.172 [0.162, 0.172] | 0.060 [0.052, 0.187] |
| neutral_pause | 174 | 0.338 [0.335, 0.382] | 0.205 [0.194, 0.291] | 0.201 [0.169, 0.203] | 0.063 [0.073, 0.102] |
| **macro-F1 (disfluencias)** | | 0.126 [0.106, 0.126] | 0.281 [0.285, 0.307] | 0.286 [0.287, 0.291] | 0.058 [0.067, 0.084] |

**Diferencias pareadas de macro-F1 (IC 95 %)**

- trimodal-audio_only: [+0.162, +0.185]
- audio_text-audio_only: [+0.159, +0.201]
- trimodal-audio_text: [-0.016, +0.003]
- trimodal-kinesic_only: [+0.207, +0.220]
- kinesic_only-audio_only: [-0.059, -0.022]
