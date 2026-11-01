# S2 tables - device cuda (EXPLORATORY; control C0)

## U_W: loops and WER vs human reference (loop clip = excess >= 20 tokens beyond the reference)

| branch | loop clips all (AMERESCO / DIHANA) | lenient flag clips | excess tokens | WER all [95% CI] | WER AMERESCO (4 rec.) | WER DIHANA |
|---|---|---|---|---|---|---|
| C0 | 8/135 (7/75, 1/60) | 11 | 953 | 0.546 [0.269, 0.717] | 0.707 | 0.217 |
| P1 | 11/135 (11/75, 0/60) | 16 | 886 | 0.520 [0.259, 0.656] | 0.705 | 0.143 |
| A | 7/135 (6/75, 1/60) | 9 | 863 | 0.524 [0.255, 0.679] | 0.676 | 0.217 |
| B | 4/135 (4/75, 0/60) | 5 | 294 | 0.420 [0.211, 0.528] | 0.551 | 0.154 |
| C | 12/135 (12/75, 0/60) | 14 | 1260 | 0.553 [0.303, 0.758] | 0.755 | 0.142 |
| D_B | 1/135 (1/75, 0/60) | 3 | 92 | 0.384 [0.199, 0.471] | 0.504 | 0.140 |

## U_S: survival of literal repetition (214 positives) and rule false positives (679 sampled negatives)

| branch | survival k/n [95% CI] | rule FP k/n [95% CI] | WER vs ref A on negatives | empty outputs | mean words/unit | RTF (all units) | looped chunks / re-decoded / unresolved |
|---|---|---|---|---|---|---|---|
| C0 | 107/214 = 0.500 [0.431, 0.561] | 12/679 = 0.0177 [0.0078, 0.0296] | 0.135 | 2 | 11.02 | 0.059 | 0/0/0 |
| P1 | 148/214 = 0.692 [0.626, 0.751] | 14/679 = 0.0206 [0.0096, 0.0330] | 0.127 | 3 | 10.95 | 0.061 | 0/0/0 |
| A | 107/214 = 0.500 [0.431, 0.561] | 12/679 = 0.0177 [0.0078, 0.0296] | 0.137 | 2 | 10.85 | 0.066 | 0/0/0 |
| B | 105/214 = 0.491 [0.425, 0.550] | 11/679 = 0.0162 [0.0066, 0.0278] | 0.131 | 2 | 10.61 | 0.061 | 23/23/5 |
| C | 100/214 = 0.467 [0.396, 0.533] | 10/679 = 0.0147 [0.0056, 0.0260] | 0.127 | 2 | 10.91 | 0.047 | 0/0/0 |
| D_B | 147/214 = 0.687 [0.624, 0.744] | 14/679 = 0.0206 [0.0096, 0.0330] | 0.126 | 3 | 10.65 | 0.051 | 27/27/3 |

## Paired differences (branch - reference), group bootstrap 5000x, Holm over a fixed family of 16 (rows vs C0 only)

| comparison | metric | delta [95% CI] | groups | p | Holm p |
|---|---|---|---|---|---|
| P1_vs_C0 | loop_rate | +0.0222 [-0.0316, +0.0564] | 43 | 0.4851 | (outside family) |
| P1_vs_C0 | wer | -0.0257 [-0.1325, +0.0660] | 43 | 0.5835 | (outside family) |
| P1_vs_C0 | survival | +0.1916 [+0.1371, +0.2477] | 84 | 0.0004 | (outside family) |
| P1_vs_C0 | rule_fp | +0.0029 [-0.0053, +0.0111] | 122 | 0.6051 | (outside family) |
| A_vs_C0 | loop_rate | -0.0074 [-0.0246, +0.0000] | 43 | 0.7367 | 1.0000 |
| A_vs_C0 | wer | -0.0212 [-0.0423, +0.0003] | 43 | 0.0928 | 1.0000 |
| A_vs_C0 | survival | +0.0000 [+0.0000, +0.0000] | 84 | 1.0000 | 1.0000 |
| A_vs_C0 | rule_fp | +0.0000 [+0.0000, +0.0000] | 122 | 1.0000 | 1.0000 |
| A_vs_P1 | loop_rate | -0.0296 [-0.0588, +0.0161] | 43 | 0.2803 | (outside family) |
| A_vs_P1 | wer | +0.0044 [-0.0969, +0.1119] | 43 | 0.9374 | (outside family) |
| A_vs_P1 | survival | -0.1916 [-0.2477, -0.1371] | 84 | 0.0004 | (outside family) |
| A_vs_P1 | rule_fp | -0.0029 [-0.0111, +0.0053] | 122 | 0.6051 | (outside family) |
| B_vs_C0 | loop_rate | -0.0296 [-0.0647, +0.0000] | 43 | 0.1020 | 1.0000 |
| B_vs_C0 | wer | -0.1256 [-0.2057, -0.0481] | 43 | 0.0072 | 0.1008 |
| B_vs_C0 | survival | -0.0093 [-0.0293, +0.0000] | 84 | 0.7355 | 1.0000 |
| B_vs_C0 | rule_fp | -0.0015 [-0.0049, +0.0000] | 122 | 0.7323 | 1.0000 |
| B_vs_P1 | loop_rate | -0.0519 [-0.0825, -0.0114] | 43 | 0.0300 | (outside family) |
| B_vs_P1 | wer | -0.0999 [-0.1708, -0.0201] | 43 | 0.0280 | (outside family) |
| B_vs_P1 | survival | -0.2009 [-0.2611, -0.1422] | 84 | 0.0004 | (outside family) |
| B_vs_P1 | rule_fp | -0.0044 [-0.0119, +0.0030] | 122 | 0.3407 | (outside family) |
| C_vs_C0 | loop_rate | +0.0296 [-0.0159, +0.0686] | 43 | 0.2767 | 1.0000 |
| C_vs_C0 | wer | +0.0073 [-0.1232, +0.1568] | 43 | 0.9006 | 1.0000 |
| C_vs_C0 | survival | -0.0327 [-0.0730, +0.0050] | 84 | 0.1112 | 1.0000 |
| C_vs_C0 | rule_fp | -0.0029 [-0.0076, +0.0000] | 122 | 0.2583 | 1.0000 |
| C_vs_P1 | loop_rate | +0.0074 [-0.0469, +0.0625] | 43 | 0.8522 | (outside family) |
| C_vs_P1 | wer | +0.0330 [-0.0724, +0.1630] | 43 | 0.6271 | (outside family) |
| C_vs_P1 | survival | -0.2243 [-0.2980, -0.1542] | 84 | 0.0004 | (outside family) |
| C_vs_P1 | rule_fp | -0.0059 [-0.0130, +0.0000] | 122 | 0.1336 | (outside family) |
| D_B_vs_C0 | loop_rate | -0.0519 [-0.1190, -0.0072] | 43 | 0.0336 | 0.4367 |
| D_B_vs_C0 | wer | -0.1614 [-0.2655, -0.0598] | 43 | 0.0008 | 0.0120 |
| D_B_vs_C0 | survival | +0.1869 [+0.1329, +0.2438] | 84 | 0.0004 | 0.0064 |
| D_B_vs_C0 | rule_fp | +0.0029 [-0.0053, +0.0111] | 122 | 0.6051 | 1.0000 |
| D_B_vs_P1 | loop_rate | -0.0741 [-0.1176, -0.0115] | 43 | 0.0300 | (outside family) |
| D_B_vs_P1 | wer | -0.1358 [-0.2081, -0.0554] | 43 | 0.0088 | (outside family) |
| D_B_vs_P1 | survival | -0.0047 [-0.0147, +0.0000] | 84 | 0.7355 | (outside family) |
| D_B_vs_P1 | rule_fp | +0.0000 [+0.0000, +0.0000] | 122 | 1.0000 | (outside family) |

## Decision rule (protocol 4.1-4.2)

| branch | control loop clips | branch loop clips | reduces loops | WER ok | survival ok | rule FP ok | qualifies |
|---|---|---|---|---|---|---|---|
| A | 8 | 7 | False | True | True | True | False |
| B | 8 | 4 | True | True | True | True | True |
| C | 8 | 12 | False | False | False | True | False |
| D_B | 8 | 1 | True | True | True | True | True |
