# R1 — Sonda de identidad de charla (speaker)

> Sonda lineal por rama sobre GroupKFold por segmento (réplica de la partición P2).
> Mide información de charla linealmente accesible en cada rama. Resultado
> descriptivo pre-especificado: se reporta sin prueba de hipótesis ni corrección.

- Charlas: 17 · segmentos: 86 · ventanas: 18347
- Excluidas por tener un solo segmento: PsJ2IAVt7Is
- Azar (exactitud balanceada): 0.0588
- IC 95 %: bootstrap de clúster por segmento (B=300)

| Rama | Rasgos | Exactitud balanceada | IC 95 % | × azar |
|---|---|---|---|---|
| kinesic | 24 | 0.0588 | [0.0000, 0.0667] | 1.0× |
| prosodic | 9 | 0.0588 | [0.0000, 0.0667] | 1.0× |
| linguistic | 24 | 0.0588 | [0.0000, 0.0667] | 1.0× |
| acoustic | 10 | 0.0588 | [0.0279, 0.0667] | 1.0× |

Lectura: una rama muy por encima del azar identifica la charla por sí sola. Cualquier protocolo que deje material de la misma charla en entrenamiento y prueba le entrega esa señal al clasificador de disfluencias.
