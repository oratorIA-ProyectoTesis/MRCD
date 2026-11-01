# Final evaluation (frozen selections; fold 5 = reserved test)

frozen_selection.json sha256 `b1db7613a21fd2e9fc4e7ce284c0bdcb4d4bdb3c0ac6034b1836e48f26a33574`
Untrained comparator per class (chosen on DEV): filler_word: rule_asr_lexical (dev F1 0.67), prolongation: always_present (dev F1 0.32), repetition: rule_asr_lexical (dev F1 0.50), revision: rule_asr_lexical (dev F1 0.07)

Ea adopted: **A_noise_tau0.5**; Eb adopted classes (zero features): filler_word:V0 rule, prolongation:thr/max@0.02, repetition:V0 rule, revision:V0 rule


#### dihana / filler_word  (units: 1223, test partition only)

| version | partition | n | pos | TP/FP/FN/TN | P | R | F1 |
|---|---|---|---|---|---|---|---|
| V0 | test | 1223 | 145 | 64/44/81/1034 | 0.59 | 0.44 | 0.51 [0.39-0.59] |
| Va | test | 1223 | 145 | 67/50/78/1028 | 0.57 | 0.46 | 0.51 [0.41-0.59] |
| Vb | test | 1223 | 145 | 64/44/81/1034 | 0.59 | 0.44 | 0.51 [0.39-0.59] |
| Vab | test | 1223 | 145 | 64/44/81/1034 | 0.59 | 0.44 | 0.51 [0.41-0.58] |
| Vc | test | 1223 | 145 | 90/77/55/1001 | 0.54 | 0.62 | 0.58 [0.46-0.66] |
| Vac | test | 1223 | 145 | 83/71/62/1007 | 0.54 | 0.57 | 0.56 [0.45-0.63] |
| always_present | test | 1223 | 145 | 145/1078/0/0 | 0.12 | 1.00 | 0.21 [0.15-0.28] |
| rule_existing_autolabeler | test | 1223 | 145 | 65/22/80/1056 | 0.75 | 0.45 | 0.56 [0.45-0.64] |
| rule_asr_lexical | test | 1223 | 145 | 61/5/84/1073 | 0.92 | 0.42 | 0.58 [0.47-0.66] |

| version | dF1 vs V0 [95% CI by speaker] | verdict vs V0 | dF1 vs rule_asr_lexical [95% CI] | verdict vs untrained | McNemar vs V0 (v right/V0 wrong : V0 right/v wrong, p) |
|---|---|---|---|---|---|
| Va | +0.006 [-0.014, +0.031] | not demonstrated (CI includes 0) | -0.067 [-0.130, -0.011] | WORSENING (CI < 0) | 3:6 p=0.508 |
| Vb | +0.000 [+0.000, +0.000] | not demonstrated (CI includes 0) | -0.072 [-0.149, -0.007] | WORSENING (CI < 0) | 0:0 p=1.000 |
| Vab | +0.000 [-0.031, +0.035] | not demonstrated (CI includes 0) | -0.072 [-0.132, -0.019] | WORSENING (CI < 0) | 8:8 p=1.000 |
| Vc | +0.071 [+0.011, +0.129] | improvement demonstrated (CI > 0) | -0.001 [-0.072, +0.069] | not demonstrated (CI includes 0) | 47:54 p=0.551 |
| Vac | +0.049 [-0.009, +0.110] | not demonstrated (CI includes 0) | -0.023 [-0.096, +0.054] | not demonstrated (CI includes 0) | 34:42 p=0.422 |

#### dihana / prolongation  (units: 1223, test partition only)

| version | partition | n | pos | TP/FP/FN/TN | P | R | F1 |
|---|---|---|---|---|---|---|---|
| V0 | test | 1223 | 211 | 95/46/116/966 | 0.67 | 0.45 | 0.54 [0.44-0.62] |
| Va | test | 1223 | 211 | 95/46/116/966 | 0.67 | 0.45 | 0.54 [0.44-0.62] |
| Vb | test | 1223 | 211 | 157/206/54/806 | 0.43 | 0.74 | 0.55 [0.48-0.61] |
| Vab | test | 1223 | 211 | 158/222/53/790 | 0.42 | 0.75 | 0.53 [0.47-0.60] |
| Vc | test | 1223 | 211 | 157/130/54/882 | 0.55 | 0.74 | 0.63 [0.56-0.69] |
| Vac | test | 1223 | 211 | 139/92/72/920 | 0.60 | 0.66 | 0.63 [0.56-0.69] |
| always_present | test | 1223 | 211 | 211/1012/0/0 | 0.17 | 1.00 | 0.29 [0.25-0.34] |
| rule_existing_autolabeler | test | 1223 | 211 | 26/6/185/1006 | 0.81 | 0.12 | 0.21 [0.14-0.29] |

| version | dF1 vs V0 [95% CI by speaker] | verdict vs V0 | dF1 vs always_present [95% CI] | verdict vs untrained | McNemar vs V0 (v right/V0 wrong : V0 right/v wrong, p) |
|---|---|---|---|---|---|
| Va | +0.000 [+0.000, +0.000] | not demonstrated (CI includes 0) | +0.245 [+0.142, +0.337] | improvement demonstrated (CI > 0) | 0:0 p=1.000 |
| Vb | +0.007 [-0.081, +0.097] | not demonstrated (CI includes 0) | +0.253 [+0.209, +0.295] | improvement demonstrated (CI > 0) | 62:160 p=0.000 |
| Vab | -0.005 [-0.095, +0.085] | not demonstrated (CI includes 0) | +0.240 [+0.196, +0.282] | improvement demonstrated (CI > 0) | 63:176 p=0.000 |
| Vc | +0.091 [+0.009, +0.179] | improvement demonstrated (CI > 0) | +0.336 [+0.282, +0.388] | improvement demonstrated (CI > 0) | 68:90 p=0.094 |
| Vac | +0.089 [+0.015, +0.167] | improvement demonstrated (CI > 0) | +0.335 [+0.261, +0.396] | improvement demonstrated (CI > 0) | 52:54 p=0.923 |

#### dihana / repetition  (units: 1223, test partition only)

| version | partition | n | pos | TP/FP/FN/TN | P | R | F1 |
|---|---|---|---|---|---|---|---|
| V0 | test | 1223 | 60 | 17/20/43/1143 | 0.46 | 0.28 | 0.35 [0.21-0.48] |
| Va | test | 1223 | 60 | 18/22/42/1141 | 0.45 | 0.30 | 0.36 [0.22-0.48] |
| Vb | test | 1223 | 60 | 17/20/43/1143 | 0.46 | 0.28 | 0.35 [0.21-0.48] |
| Vab | test | 1223 | 60 | 18/22/42/1141 | 0.45 | 0.30 | 0.36 [0.22-0.48] |
| Vc | test | 1223 | 60 | 14/14/46/1149 | 0.50 | 0.23 | 0.32 [0.18-0.43] |
| Vac | test | 1223 | 60 | 24/56/36/1107 | 0.30 | 0.40 | 0.34 [0.22-0.44] |
| always_present | test | 1223 | 60 | 60/1163/0/0 | 0.05 | 1.00 | 0.09 [0.06-0.12] |
| rule_existing_autolabeler | test | 1223 | 60 | 19/17/41/1146 | 0.53 | 0.32 | 0.40 [0.26-0.52] |
| rule_asr_lexical | test | 1223 | 60 | 22/18/38/1145 | 0.55 | 0.37 | 0.44 [0.28-0.57] |

| version | dF1 vs V0 [95% CI by speaker] | verdict vs V0 | dF1 vs rule_asr_lexical [95% CI] | verdict vs untrained | McNemar vs V0 (v right/V0 wrong : V0 right/v wrong, p) |
|---|---|---|---|---|---|
| Va | +0.009 [-0.017, +0.053] | not demonstrated (CI includes 0) | -0.080 [-0.141, -0.025] | WORSENING (CI < 0) | 1:2 p=1.000 |
| Vb | +0.000 [+0.000, +0.000] | not demonstrated (CI includes 0) | -0.089 [-0.157, -0.025] | WORSENING (CI < 0) | 0:0 p=1.000 |
| Vab | +0.009 [-0.017, +0.053] | not demonstrated (CI includes 0) | -0.080 [-0.141, -0.025] | WORSENING (CI < 0) | 1:2 p=1.000 |
| Vc | -0.032 [-0.112, +0.043] | not demonstrated (CI includes 0) | -0.122 [-0.200, -0.053] | WORSENING (CI < 0) | 8:5 p=0.581 |
| Vac | -0.008 [-0.088, +0.077] | not demonstrated (CI includes 0) | -0.097 [-0.181, -0.009] | WORSENING (CI < 0) | 8:37 p=0.000 |

#### dihana / revision  (units: 1223, test partition only)

| version | partition | n | pos | TP/FP/FN/TN | P | R | F1 |
|---|---|---|---|---|---|---|---|
| V0 | test | 1223 | 36 | 2/5/34/1182 | 0.29 | 0.06 | 0.09 [0.00-0.20] |
| Va | test | 1223 | 36 | 2/5/34/1182 | 0.29 | 0.06 | 0.09 [0.00-0.20] |
| Vb | test | 1223 | 36 | 2/5/34/1182 | 0.29 | 0.06 | 0.09 [0.00-0.20] |
| Vab | test | 1223 | 36 | 2/5/34/1182 | 0.29 | 0.06 | 0.09 [0.00-0.20] |
| Vc | test | 1223 | 36 | 2/5/34/1182 | 0.29 | 0.06 | 0.09 [0.00-0.20] |
| Vac | test | 1223 | 36 | 2/5/34/1182 | 0.29 | 0.06 | 0.09 [0.00-0.20] |
| always_present | test | 1223 | 36 | 36/1187/0/0 | 0.03 | 1.00 | 0.06 [0.04-0.08] |
| rule_existing_autolabeler | test | 1223 | 36 | 1/1/35/1186 | 0.50 | 0.03 | 0.05 [0.00-0.17] |
| rule_asr_lexical | test | 1223 | 36 | 2/1/34/1186 | 0.67 | 0.06 | 0.10 [0.00-0.21] |

| version | dF1 vs V0 [95% CI by speaker] | verdict vs V0 | dF1 vs rule_asr_lexical [95% CI] | verdict vs untrained | McNemar vs V0 (v right/V0 wrong : V0 right/v wrong, p) |
|---|---|---|---|---|---|
| Va | +0.000 [+0.000, +0.000] | not demonstrated (CI includes 0) | -0.010 [-0.020, +0.000] | not demonstrated (CI includes 0) | 0:0 p=1.000 |
| Vb | +0.000 [+0.000, +0.000] | not demonstrated (CI includes 0) | -0.010 [-0.020, +0.000] | not demonstrated (CI includes 0) | 0:0 p=1.000 |
| Vab | +0.000 [+0.000, +0.000] | not demonstrated (CI includes 0) | -0.010 [-0.020, +0.000] | not demonstrated (CI includes 0) | 0:0 p=1.000 |
| Vc | +0.000 [+0.000, +0.000] | not demonstrated (CI includes 0) | -0.010 [-0.020, +0.000] | not demonstrated (CI includes 0) | 0:0 p=1.000 |
| Vac | +0.000 [+0.000, +0.000] | not demonstrated (CI includes 0) | -0.010 [-0.020, +0.000] | not demonstrated (CI includes 0) | 0:0 p=1.000 |

#### DIHANA fold 5: reference class present (rows) x predicted class (columns), counts of UNITS (multi-label: a unit can sit in several rows/cols)


`V0`

| reference \ predicted | filler_word | prolongation | repetition | revision |
|---|---|---|---|---|
| f (n=145) | 64 | 47 | 14 | 3 |
| a (n=211) | 42 | 95 | 21 | 4 |
| r (n=60) | 12 | 23 | 17 | 2 |
| rev (n=36) | 11 | 15 | 7 | 2 |
| none (n=899) | 27 | 29 | 7 | 1 |

`Va`

| reference \ predicted | filler_word | prolongation | repetition | revision |
|---|---|---|---|---|
| f (n=145) | 67 | 47 | 14 | 3 |
| a (n=211) | 43 | 95 | 21 | 4 |
| r (n=60) | 12 | 23 | 18 | 2 |
| rev (n=36) | 11 | 15 | 7 | 2 |
| none (n=899) | 32 | 29 | 9 | 1 |

`Vb`

| reference \ predicted | filler_word | prolongation | repetition | revision |
|---|---|---|---|---|
| f (n=145) | 64 | 87 | 14 | 3 |
| a (n=211) | 42 | 157 | 21 | 4 |
| r (n=60) | 12 | 36 | 17 | 2 |
| rev (n=36) | 11 | 29 | 7 | 2 |
| none (n=899) | 27 | 154 | 7 | 1 |

`Vab`

| reference \ predicted | filler_word | prolongation | repetition | revision |
|---|---|---|---|---|
| f (n=145) | 64 | 87 | 14 | 3 |
| a (n=211) | 38 | 158 | 21 | 4 |
| r (n=60) | 10 | 36 | 18 | 2 |
| rev (n=36) | 11 | 30 | 7 | 2 |
| none (n=899) | 29 | 169 | 9 | 1 |

`Vc`

| reference \ predicted | filler_word | prolongation | repetition | revision |
|---|---|---|---|---|
| f (n=145) | 90 | 90 | 10 | 3 |
| a (n=211) | 89 | 157 | 18 | 4 |
| r (n=60) | 28 | 46 | 14 | 2 |
| rev (n=36) | 21 | 34 | 7 | 2 |
| none (n=899) | 24 | 76 | 5 | 1 |

`Vac`

| reference \ predicted | filler_word | prolongation | repetition | revision |
|---|---|---|---|---|
| f (n=145) | 83 | 76 | 34 | 3 |
| a (n=211) | 80 | 139 | 49 | 4 |
| r (n=60) | 29 | 42 | 24 | 2 |
| rev (n=36) | 20 | 31 | 13 | 2 |
| none (n=899) | 26 | 52 | 13 | 1 |

#### All declared Ea candidates on fold 5 (V0 decision rule), paired vs V0

| candidate | f1 filler (dF1 [CI]) | f1 prolongation | f1 repetition |
|---|---|---|---|
| A_zero_tau0.5 | 0.506 (+0.000 [+0.000,+0.000]) | 0.540 (+0.000 [+0.000,+0.000]) | 0.351 (+0.000 [+0.000,+0.000]) |
| A_zero_tau1.0 | 0.506 (+0.000 [+0.000,+0.000]) | 0.540 (+0.000 [+0.000,+0.000]) | 0.337 (-0.014 [-0.044,+0.000]) |
| A_zero_tau1.5 | 0.506 (+0.000 [+0.000,+0.000]) | 0.540 (+0.000 [+0.000,+0.000]) | 0.304 (-0.046 [-0.121,+0.006]) |
| A_noise_tauNone | 0.467 (-0.039 [-0.071,-0.004]) | 0.540 (+0.000 [+0.000,+0.000]) | 0.360 (+0.009 [-0.017,+0.053]) |
| A_noise_tau0.5 | 0.511 (+0.006 [-0.014,+0.031]) | 0.540 (+0.000 [+0.000,+0.000]) | 0.360 (+0.009 [-0.017,+0.053]) |
| A_noise_tau1.0 | 0.510 (+0.004 [-0.012,+0.026]) | 0.540 (+0.000 [+0.000,+0.000]) | 0.330 (-0.021 [-0.059,+0.000]) |
| A_noise_tau1.5 | 0.504 (-0.002 [-0.014,+0.015]) | 0.540 (+0.000 [+0.000,+0.000]) | 0.298 (-0.053 [-0.129,+0.003]) |

#### epic / filler_word  (units: 136, test partition only)

| version | partition | n | pos | TP/FP/FN/TN | P | R | F1 |
|---|---|---|---|---|---|---|---|
| V0 | test | 136 | 68 | 44/10/24/58 | 0.81 | 0.65 | 0.72 [0.63-0.79] |
| Va | test | 136 | 68 | 44/10/24/58 | 0.81 | 0.65 | 0.72 [0.63-0.79] |
| Vb | test | 136 | 68 | 44/10/24/58 | 0.81 | 0.65 | 0.72 [0.63-0.79] |
| Vab | test | 136 | 68 | 41/9/27/59 | 0.82 | 0.60 | 0.69 [0.59-0.77] |
| Vc | test | 136 | 68 | 61/30/7/38 | 0.67 | 0.90 | 0.77 [0.71-0.82] |
| Vac | test | 136 | 68 | 59/34/9/34 | 0.63 | 0.87 | 0.73 [0.68-0.78] |
| always_present | test | 136 | 68 | 68/68/0/0 | 0.50 | 1.00 | 0.67 [0.66-0.68] |
| rule_existing_autolabeler | test | 136 | 68 | 30/4/38/64 | 0.88 | 0.44 | 0.59 [0.46-0.69] |
| rule_asr_lexical | test | 136 | 68 | 36/1/32/67 | 0.97 | 0.53 | 0.69 [0.58-0.78] |

| version | dF1 vs V0 [95% CI by speaker] | verdict vs V0 | dF1 vs rule_asr_lexical [95% CI] | verdict vs untrained | McNemar vs V0 (v right/V0 wrong : V0 right/v wrong, p) |
|---|---|---|---|---|---|
| Va | +0.000 [+0.000, +0.000] | not demonstrated (CI includes 0) | +0.036 [-0.029, +0.101] | not demonstrated (CI includes 0) | 0:0 p=1.000 |
| Vb | +0.000 [+0.000, +0.000] | not demonstrated (CI includes 0) | +0.036 [-0.029, +0.101] | not demonstrated (CI includes 0) | 0:0 p=1.000 |
| Vab | -0.026 [-0.061, +0.000] | not demonstrated (CI includes 0) | +0.009 [-0.066, +0.087] | not demonstrated (CI includes 0) | 1:3 p=0.625 |
| Vc | +0.046 [-0.019, +0.133] | not demonstrated (CI includes 0) | +0.082 [-0.011, +0.198] | not demonstrated (CI includes 0) | 19:22 p=0.755 |
| Vac | +0.012 [-0.051, +0.092] | not demonstrated (CI includes 0) | +0.047 [-0.046, +0.154] | not demonstrated (CI includes 0) | 15:24 p=0.200 |

#### ameresco / prolongation  (units: 98, test partition only)

| version | partition | n | pos | TP/FP/FN/TN | P | R | F1 |
|---|---|---|---|---|---|---|---|
| V0 | test | 98 | 37 | 7/8/30/53 | 0.47 | 0.19 | 0.27 |
| Va | test | 98 | 37 | 7/8/30/53 | 0.47 | 0.19 | 0.27 |
| Vb | test | 98 | 37 | 26/34/11/27 | 0.43 | 0.70 | 0.54 |
| Vab | test | 98 | 37 | 26/34/11/27 | 0.43 | 0.70 | 0.54 |
| Vc | test | 98 | 37 | 32/46/5/15 | 0.41 | 0.86 | 0.56 |
| Vac | test | 98 | 37 | 32/38/5/23 | 0.46 | 0.86 | 0.60 |
| always_present | test | 98 | 37 | 37/61/0/0 | 0.38 | 1.00 | 0.55 |
| rule_existing_autolabeler | test | 98 | 37 | 1/2/36/59 | 0.33 | 0.03 | 0.05 |

| version | dF1 vs V0 [95% CI by speaker] | verdict vs V0 | dF1 vs always_present [95% CI] | verdict vs untrained | McNemar vs V0 (v right/V0 wrong : V0 right/v wrong, p) |
|---|---|---|---|---|---|
| Va | point +0.000 (no CI: 4 conversations) | McNemar only | point -0.279 | McNemar | 0:0 p=1.000 |
| Vb | point +0.267 (no CI: 4 conversations) | McNemar only | point -0.012 | McNemar | 19:26 p=0.371 |
| Vab | point +0.267 (no CI: 4 conversations) | McNemar only | point -0.012 | McNemar | 19:26 p=0.371 |
| Vc | point +0.287 (no CI: 4 conversations) | McNemar only | point +0.008 | McNemar | 27:40 p=0.142 |
| Vac | point +0.329 (no CI: 4 conversations) | McNemar only | point +0.050 | McNemar | 25:30 p=0.590 |