# R4 — Sonda de identidad de charla (invariant / speaker)

> Sonda lineal por rama sobre GroupKFold por segmento (réplica de la partición P2).
> Mide información de charla linealmente accesible en cada rama. Resultado
> descriptivo pre-especificado: se reporta sin prueba de hipótesis ni corrección.

- Charlas: 17 · segmentos: 86 · ventanas: 18347
- Excluidas por tener un solo segmento: PsJ2IAVt7Is
- Azar (exactitud balanceada): 0.0588
- IC 95 %: bootstrap de clúster por segmento (B=300)

| Rama | Canales | Exactitud balanceada | IC 95 % | × azar |
|---|---|---|---|---|
| kinesic | 12 | 0.5411 | [0.5230, 0.6015] | 9.2× |
| linguistic | 12 | 0.1538 | [0.1368, 0.1816] | 2.6× |
| acoustic | 5 | 0.4525 | [0.4248, 0.4947] | 7.7× |

Lectura: una rama muy por encima del azar identifica la charla por sí sola. Cualquier protocolo que deje material de la misma charla en entrenamiento y prueba le entrega esa señal al clasificador de disfluencias.
