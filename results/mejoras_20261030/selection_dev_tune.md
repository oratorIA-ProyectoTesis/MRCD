# Selection on DIHANA dev (3816 turns, folds 1-3) and tune (1240 turns, fold 4). Fold 5 / EPIC / AMERESCO not loaded.

## Ea short-turn handling (V0 decision rule held fixed)

V0 macro-F1(f,a,r): dev 0.479, tune 0.515

| candidate | dev macroF1 | dev dMacroF1 [95% CI] | tune macroF1 | tune dMacroF1 | passes criterion |
|---|---|---|---|---|---|
| A_zero_tau0.5 | 0.479 | +0.000 [+0.000, +0.000] | 0.515 | +0.000 | False |
| A_zero_tau1.0 | 0.479 | +0.000 [+0.000, +0.000] | 0.515 | +0.000 | False |
| A_zero_tau1.5 | 0.471 | -0.008 [-0.015, -0.002] | 0.517 | +0.002 | False |
| A_noise_tauNone | 0.480 | +0.001 [-0.010, +0.013] | 0.502 | -0.013 | False |
| A_noise_tau0.5 | 0.493 | +0.014 [+0.005, +0.025] | 0.517 | +0.002 | True |
| A_noise_tau1.0 | 0.486 | +0.007 [+0.001, +0.013] | 0.516 | +0.001 | True |
| A_noise_tau1.5 | 0.475 | -0.003 [-0.012, +0.004] | 0.514 | -0.001 | False |

Adopted: **A_noise_tau0.5**


## Eb threshold rule, features = V0 zero-pad

| class | dev pos | agg | thr | dev F1 V0 rule | dev OOF F1 thr rule | tune F1 V0 rule | tune F1 thr rule | adopted |
|---|---|---|---|---|---|---|---|---|
| filler_word | 606 | top2 | 0.08 | 0.572 | 0.584 | 0.549 | 0.526 | False |
| prolongation | 718 | max | 0.02 | 0.445 | 0.584 | 0.496 | 0.606 | True |
| repetition | 221 | max | 0.76 | 0.420 | 0.424 | 0.500 | 0.480 | False |
| revision | 113 | top2 | 0.02 | 0.091 | 0.147 | 0.172 | 0.149 | False |

## Eb threshold rule, features = A_noise_tau0.5

| class | dev pos | agg | thr | dev F1 V0 rule | dev OOF F1 thr rule | tune F1 V0 rule | tune F1 thr rule | adopted |
|---|---|---|---|---|---|---|---|---|
| filler_word | 604 | max | 0.54 | 0.603 | 0.607 | 0.569 | 0.571 | True |
| prolongation | 715 | max | 0.02 | 0.446 | 0.583 | 0.496 | 0.607 | True |
| repetition | 221 | max | 0.74 | 0.432 | 0.434 | 0.486 | 0.466 | False |
| revision | 113 | top2 | 0.02 | 0.091 | 0.140 | 0.172 | 0.147 | False |