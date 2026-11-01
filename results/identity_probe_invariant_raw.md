# R4 — Sonda de identidad de charla (invariant / raw)

> Sonda lineal por rama sobre GroupKFold por segmento (réplica de la partición P2).
> Mide información de charla linealmente accesible en cada rama. Resultado
> descriptivo pre-especificado: se reporta sin prueba de hipótesis ni corrección.

- Charlas: 17 · segmentos: 86 · ventanas: 18347
- Excluidas por tener un solo segmento: PsJ2IAVt7Is
- Azar (exactitud balanceada): 0.0588
- IC 95 %: bootstrap de clúster por segmento (B=300)

| Rama | Canales | Exactitud balanceada | IC 95 % | × azar |
|---|---|---|---|---|
| kinesic | 12 | 0.5412 | [0.5234, 0.6012] | 9.2× |
| linguistic | 12 | 0.1542 | [0.1371, 0.1824] | 2.6× |
| acoustic | 5 | 0.4527 | [0.4249, 0.4948] | 7.7× |

Lectura: una rama muy por encima del azar identifica la charla por sí sola. Cualquier protocolo que deje material de la misma charla en entrenamiento y prueba le entrega esa señal al clasificador de disfluencias.
