| config | device/compute | load s | first pass RTF (cold) | warm RTF median/file [min-max] | warm pooled RTF per round | RSS peak MB | GPU MB (load / peak, delta) | bg CPU % (start/end) | rounds discarded (load >15 %) |
|---|---|---|---|---|---|---|---|---|---|
| p0_cpu | cpu/int8 | 1.39 | 0.320 | 0.311 [0.270-0.741] | 0.392 | 921 | n/a (CPU) | 9.1/15.4 | [2] |
| p0_cuda | cuda/float16 | 2.27 | 0.040 | 0.040 [0.033-0.057] | 0.041, 0.041, 0.039, 0.043 | 921 | 743 / 889 | 12.9/5.0 | - |
| p1_cpu | cpu/int8 | 1.76 | 0.424 | 0.476 [0.340-0.731] | 0.456, 0.527 | 920 | n/a (CPU) | 24.1/9.6 | - |
| p1_cuda | cuda/float16 | 1.62 | 0.043 | 0.041 [0.036-0.050] | 0.042, 0.040, 0.043, 0.044 | 942 | 742 / 823 | 23.3/15.9 | - |

| pair | units | identical | pooled desacuerdo | median/unit | max/unit | loop-flag units (cpu, gpu) |
|---|---|---|---|---|---|---|
| p0_cpu_vs_p0_cuda | 13 | 4 | 0.0698 | 0.0149 | 0.6941 | (1, 0) |
| p1_cpu_vs_p1_cuda | 13 | 5 | 0.0278 | 0.0219 | 0.1304 | (0, 0) |
