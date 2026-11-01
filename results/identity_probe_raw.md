# R1 — Sonda de identidad de charla (raw)

> Sonda lineal por rama sobre GroupKFold por segmento (réplica de la partición P2).
> Mide información de charla linealmente accesible en cada rama. Resultado
> descriptivo pre-especificado: se reporta sin prueba de hipótesis ni corrección.

- Charlas: 17 · segmentos: 86 · ventanas: 18347
- Excluidas por tener un solo segmento: PsJ2IAVt7Is
- Azar (exactitud balanceada): 0.0588
- IC 95 %: bootstrap de clúster por segmento (B=300)

| Rama | Rasgos | Exactitud balanceada | IC 95 % | × azar |
|---|---|---|---|---|
| kinesic | 24 | 0.7511 | [0.7271, 0.8339] | 12.8× |
| prosodic | 9 | 0.1739 | [0.1558, 0.1887] | 3.0× |
| linguistic | 24 | 0.2137 | [0.1905, 0.2517] | 3.6× |
| acoustic | 10 | 0.6417 | [0.5867, 0.7169] | 10.9× |

Lectura: una rama muy por encima del azar identifica la charla por sí sola. Cualquier protocolo que deje material de la misma charla en entrenamiento y prueba le entrega esa señal al clasificador de disfluencias.
