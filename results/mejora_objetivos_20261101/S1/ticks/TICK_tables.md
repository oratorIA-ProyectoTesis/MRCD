| config | tick s | units | ticks | latency median s | p95 s | max s | RTF_tick pooled | ticks over budget | max lag s | all units keep up (lag <= tick) | bg CPU % start/end |
|---|---|---|---|---|---|---|---|---|---|---|---|
| p0_cpu (cpu/int8) | 0.5 | 3 | 72 | 2.907 | 3.762 | 5.726 | 5.717 | 100.0% | 65.76 | False | 12.2/5.7 |
| p0_cpu (cpu/int8) | 2.0 | 3 | 21 | 2.577 | 4.509 | 4.618 | 1.373 | 100.0% | 9.26 | False | 15.2/19.9 |
| p0_cuda (cuda/float16) | 0.5 | 11 | 634 | 0.333 | 0.466 | 1.274 | 0.688 | 3.2% | 1.27 | False | 12.6/15.4 |
| p0_cuda (cuda/float16) | 2.0 | 11 | 169 | 0.340 | 0.424 | 1.271 | 0.170 | 0.0% | 1.27 | True | 22.5/18.2 |

Paired view: GPU restricted to the units used by the CPU baseline at the same tick (per-unit figures; pooled RTF = unit-tick-weighted mean).

| config | tick s | units | RTF_tick pooled | worst unit p95 s | max lag s |
|---|---|---|---|---|---|
| p0_cpu | 0.5 | 3 | 5.717 | 3.794 | 65.76 |
| p0_cpu | 2.0 | 3 | 1.373 | 4.118 | 9.26 |
| p0_cuda | 0.5 | 3 | 0.565 | 0.349 | 0.35 |
| p0_cuda | 2.0 | 3 | 0.161 | 0.419 | 0.43 |
