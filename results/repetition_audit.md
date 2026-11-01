# Auditoria de repeticiones: cuantas son costura del troceado

- Referencia (archivo entero): `C:\zArnes-tesis\mrcd\data\features` -> **92** eventos
- Troceada (8 s / solape 2 s): `C:\zArnes-tesis\mrcd\data\v4\features` -> **228** eventos
- Limites de region en 8, 14, 20, ... s (paso 6 s)

> Cruzar un limite es un INDICIO, no una prueba: un hablante tambien puede
> repetir una palabra justo ahi. Lo que lo vuelve informativo es comparar la
> tasa entre eventos nuevos y eventos que ya existian sin costuras.

| grupo | eventos | cruzan costura | tasa |
|---|---|---|---|
| ya estaban en la referencia | 58 | 4 | 6.9 % |
| **nuevos del troceado** | 170 | **80** | **47.1 %** |

## Lectura

Los eventos que ya existian sin trocear cruzan un limite el 6.9 % de las veces. Esa es la tasa base esperable por azar, porque los limites caen en posiciones arbitrarias del habla.

Si los eventos nuevos fueran repeticiones genuinas recuperadas, deberian cruzar limites a esa misma tasa. Cruzan al 47.1 %.

**Exceso estimado: ~68 de 170 eventos nuevos son atribuibles a costuras**, no al hablante. El troceado esta generando repeticiones falsas, coherente con la caida del acuerdo del juez en esta clase (85 % -> 67 %).

## Ejemplos de eventos nuevos que cruzan un limite

| grabacion | t | limite | tokens |
|---|---|---|---|
| ak4Oxthf_fc_s00 | 36.90–38.54 | 38 s | `la, la... la` |
| ak4Oxthf_fc_s01 | 31.80–32.32 | 32 s | `le... leído` |
| BAQYUo3JmkM_s02 | 109.98–110.26 | 110 s | `de... de` |
| BwAtqPAnHu4_s00 | 25.78–26.32 | 26 s | `expo... exposición` |
| C-8zZCmScYw_s00 | 109.54–110.04 | 110 s | `este... este,` |
| C-8zZCmScYw_s00 | 187.90–188.10 | 188 s | `de... de` |
| C-8zZCmScYw_s00 | 271.86–277.98 | 272 s | `sí, sí, sí, sí, sí, sí, sí, sí, sí, sí, ` |
| C-8zZCmScYw_s01 | 49.18–50.84 | 50 s | `eh... eh,` |
| C-8zZCmScYw_s01 | 67.86–68.04 | 68 s | `eh... eh,` |
| DNrrpE-sYnU_s02 | 7.80–8.40 | 8 s | `pa... patrocinado,` |
| DNrrpE-sYnU_s02 | 43.24–44.46 | 44 s | `Sí. sido` |
| DNrrpE-sYnU_s03 | 30.60–33.62 | 32 s | `desde desde` |
| DNrrpE-sYnU_s03 | 43.86–44.24 | 44 s | `de... declara` |
| DNrrpE-sYnU_s04 | 67.80–68.20 | 68 s | `pre... pregunta` |
| DNrrpE-sYnU_s05 | 7.56–8.48 | 8 s | `en... encuentra.` |

