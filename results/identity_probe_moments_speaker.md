# R1 — Sonda de identidad de charla (moments / speaker)

> Sonda lineal por rama sobre GroupKFold por segmento (réplica de la partición P2).
> Mide información de charla linealmente accesible en cada rama. Resultado
> descriptivo pre-especificado: se reporta sin prueba de hipótesis ni corrección.

- Charlas: 17 · segmentos: 86 · ventanas: 18347
- Excluidas por tener un solo segmento: PsJ2IAVt7Is
- Azar (exactitud balanceada): 0.0588
- IC 95 %: bootstrap de clúster por segmento (B=300)

| Rama | Canales | Exactitud balanceada | IC 95 % | × azar |
|---|---|---|---|---|
| kinesic | 12 | 0.3833 | [0.3508, 0.4367] | 6.5× |
| prosodic | 9 | 0.1739 | [0.1558, 0.1887] | 3.0× |
| linguistic | 12 | 0.6640 | [0.6192, 0.7085] | 11.3× |
| acoustic | 5 | 0.2128 | [0.1718, 0.2421] | 3.6× |

Lectura: una rama muy por encima del azar identifica la charla por sí sola. Cualquier protocolo que deje material de la misma charla en entrenamiento y prueba le entrega esa señal al clasificador de disfluencias.
