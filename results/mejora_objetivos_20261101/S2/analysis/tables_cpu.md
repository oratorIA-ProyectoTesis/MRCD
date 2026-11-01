# S2 tables - device cpu (EXPLORATORY; control C0)

## U_W: loops and WER vs human reference (loop clip = excess >= 20 tokens beyond the reference)

| branch | loop clips all (AMERESCO / DIHANA) | lenient flag clips | excess tokens | WER all [95% CI] | WER AMERESCO (4 rec.) | WER DIHANA |
|---|---|---|---|---|---|---|
| C0 | 11/135 (11/75, 0/60) | 12 | 1065 | 0.583 [0.296, 0.734] | 0.794 | 0.154 |
| P1 | 9/135 (9/75, 0/60) | 11 | 898 | 0.531 [0.241, 0.695] | 0.720 | 0.146 |
| B | 0/135 (0/75, 0/60) | 1 | 13 | 0.385 [0.213, 0.469] | 0.499 | 0.154 |
| D_B | 2/135 (2/75, 0/60) | 3 | 173 | 0.400 [0.200, 0.490] | 0.528 | 0.140 |

## U_S: survival of literal repetition (214 positives) and rule false positives (679 sampled negatives)

| branch | survival k/n [95% CI] | rule FP k/n [95% CI] | WER vs ref A on negatives | empty outputs | mean words/unit | RTF (all units) | looped chunks / re-decoded / unresolved |
|---|---|---|---|---|---|---|---|
| C0 | 105/214 = 0.491 [0.426, 0.550] | 11/679 = 0.0162 [0.0065, 0.0279] | 0.153 | 2 | 11.03 | nan | -/-/- |
| P1 | 144/214 = 0.673 [0.613, 0.730] | 14/679 = 0.0206 [0.0095, 0.0336] | 0.130 | 3 | 10.87 | nan | -/-/- |
| B | 103/214 = 0.481 [0.417, 0.541] | 10/679 = 0.0147 [0.0056, 0.0259] | 0.133 | 2 | 10.71 | 0.481 | 25/25/3 |
| D_B | 143/214 = 0.668 [0.609, 0.723] | 12/679 = 0.0177 [0.0076, 0.0293] | 0.127 | 3 | 10.70 | 0.454 | 27/27/4 |

## Paired differences (branch - reference), group bootstrap 5000x, Holm over a fixed family of 16 (rows vs C0 only)

| comparison | metric | delta [95% CI] | groups | p | Holm p |
|---|---|---|---|---|---|
| P1_vs_C0 | loop_rate | -0.0148 [-0.0492, +0.0000] | 43 | 0.7367 | (outside family) |
| P1_vs_C0 | wer | -0.0523 [-0.0983, -0.0116] | 43 | 0.0116 | (outside family) |
| P1_vs_C0 | survival | +0.1822 [+0.1337, +0.2328] | 84 | 0.0004 | (outside family) |
| P1_vs_C0 | rule_fp | +0.0044 [-0.0015, +0.0113] | 122 | 0.2480 | (outside family) |
| B_vs_C0 | loop_rate | -0.0815 [-0.1542, -0.0115] | 43 | 0.0300 | 0.4199 |
| B_vs_C0 | wer | -0.1981 [-0.2749, -0.0588] | 43 | 0.0300 | 0.4199 |
| B_vs_C0 | survival | -0.0093 [-0.0293, +0.0000] | 84 | 0.7355 | 1.0000 |
| B_vs_C0 | rule_fp | -0.0015 [-0.0049, +0.0000] | 122 | 0.7323 | 1.0000 |
| B_vs_P1 | loop_rate | -0.0667 [-0.1099, -0.0115] | 43 | 0.0300 | (outside family) |
| B_vs_P1 | wer | -0.1458 [-0.2330, -0.0218] | 43 | 0.0228 | (outside family) |
| B_vs_P1 | survival | -0.1916 [-0.2468, -0.1382] | 84 | 0.0004 | (outside family) |
| B_vs_P1 | rule_fp | -0.0059 [-0.0135, +0.0011] | 122 | 0.1380 | (outside family) |
| D_B_vs_C0 | loop_rate | -0.0667 [-0.1475, -0.0104] | 43 | 0.0300 | 0.4199 |
| D_B_vs_C0 | wer | -0.1834 [-0.2708, -0.0725] | 43 | 0.0016 | 0.0240 |
| D_B_vs_C0 | survival | +0.1776 [+0.1292, +0.2281] | 84 | 0.0004 | 0.0064 |
| D_B_vs_C0 | rule_fp | +0.0015 [-0.0031, +0.0068] | 122 | 0.7910 | 1.0000 |
| D_B_vs_P1 | loop_rate | -0.0519 [-0.1020, -0.0104] | 43 | 0.0300 | (outside family) |
| D_B_vs_P1 | wer | -0.1311 [-0.2266, -0.0365] | 43 | 0.0088 | (outside family) |
| D_B_vs_P1 | survival | -0.0047 [-0.0147, +0.0000] | 84 | 0.7355 | (outside family) |
| D_B_vs_P1 | rule_fp | -0.0029 [-0.0076, +0.0000] | 122 | 0.2795 | (outside family) |

## Decision rule (protocol 4.1-4.2)

| branch | control loop clips | branch loop clips | reduces loops | WER ok | survival ok | rule FP ok | qualifies |
|---|---|---|---|---|---|---|---|
| B | 11 | 0 | True | True | True | True | True |
| D_B | 11 | 2 | True | True | True | True | True |
