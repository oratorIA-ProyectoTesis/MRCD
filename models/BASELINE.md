# Baseline oficial: `audio_text` v1

El baseline oficial para comparación es el conjunto congelado de predicciones de
`data/results_Ka_okSSytes.json`:

- modo: `audio_text`;
- ventana lingüística extendida + prosodia;
- decisión: `argmax` directo sobre Softmax;
- resultado adjudicado: 12/60 coincidencias exactas;
- entrenamiento: supervisión heurística débil (`label_source=auto`).

## Limitación de reproducibilidad

El checkpoint exacto que produjo esas predicciones no está disponible en este
workspace. El artefacto de `models_stale/ckpt_audio_text.pt` es anterior e
incompatible (`ling=10`, `pros=0`) y no debe promoverse como reemplazo del v1
contextual (`ling=12`, `pros=9`). Hasta recuperar el checkpoint exacto, el
baseline oficial es un artefacto de evaluación congelado, no un modelo
desplegable reproducible.

El experimento `audio_text_v2` fue rechazado y se conserva únicamente con fines
de auditoría bajo `results/rejected/`.
