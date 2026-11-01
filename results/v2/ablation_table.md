# Ablación con test independiente (juez multimodal)

> El conjunto de prueba fue etiquetado por un VLM a ciegas, sin acceso a las heurísticas. El acuerdo juez–humano se reporta en `results/judge_agreement.md` y condiciona la lectura de esta tabla.

## Ablación multimodal — entrenamiento: **auto** · test: **gold_llm**

- Entrenamiento con etiquetas 'auto' (supervisión débil) y evaluación con etiquetas 'gold_llm' independientes; GroupKFold por hablante (k=5, hablantes=18)
- IC 95 %: bootstrap de clúster por hablante (B=1000)
- Ventanas: 503

| Clase | Soporte | audio_only | audio_text | trimodal |
|---|---|---|---|---|
| fluent | 177 | 0.399 [0.251, 0.531] | 0.599 [0.430, 0.741] | 0.615 [0.471, 0.738] |
| filler_word | 50 | 0.103 [0.020, 0.170] | 0.474 [0.337, 0.581] | 0.491 [0.345, 0.611] |
| prolongation | 13 | 0.024 [0.000, 0.071] | 0.088 [0.000, 0.148] | 0.110 [0.000, 0.200] |
| repetition | 57 | 0.126 [0.000, 0.206] | 0.721 [0.667, 0.813] | 0.740 [0.660, 0.870] |
| block | 2 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| revision | 10 | 0.000 [0.000, 0.000] | 0.235 [0.082, 0.375] | 0.261 [0.095, 0.405] |
| rhetorical_pause | 20 | 0.056 [0.000, 0.130] | 0.192 [0.040, 0.316] | 0.105 [0.000, 0.189] |
| neutral_pause | 174 | 0.524 [0.449, 0.628] | 0.519 [0.438, 0.582] | 0.469 [0.400, 0.546] |
| **macro-F1 (disfluencias)** | | 0.119 [0.096, 0.145] | 0.318 [0.281, 0.376] | 0.311 [0.270, 0.370] |

**Diferencias pareadas de macro-F1 (IC 95 %)**

- trimodal-audio_only: [+0.157, +0.244]
- audio_text-audio_only: [+0.170, +0.244]
- trimodal-audio_text: [-0.044, +0.033]
