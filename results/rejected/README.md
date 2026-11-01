# Experimentos rechazados

`audio_text_v2.pt` incorporó `duration_ratio` y una política post-Softmax con
umbrales manuales. Obtuvo 8/60 frente a 12/60 del baseline y no rescató ninguno
de los ocho falsos positivos `prolongation` cuyo juicio era `fluent`.

No debe utilizarse como modelo oficial ni desplegarse. Los reportes históricos
`audio_text_v2_comparison.*` permanecen en `results/` para trazabilidad.
