
#### DIHANA(turn,audio+ASR) / filler_word

| comparator | partition | n | pos | TP/FP/FN/TN | P | R | F1 [95% CI, bootstrap by group] |
|---|---|---|---|---|---|---|---|
| always_present | dev | 1253 | 218 | 218/1035/0/0 | 0.17 | 1.00 | 0.30 [0.25-0.34] |
| always_absent | dev | 1253 | 218 | 0/0/218/1035 | n/a | 0.00 | 0.00 [0.00-0.00] |
| rule_asr_lexical | dev | 1253 | 218 | 113/6/105/1029 | 0.95 | 0.52 | 0.67 [0.61-0.72] |
| rule_existing_autolabeler | dev | 1253 | 218 | 124/51/94/984 | 0.71 | 0.57 | 0.63 [0.57-0.69] |
| mrcd_checkpoint | dev | 1253 | 218 | 101/44/117/991 | 0.70 | 0.46 | 0.56 [0.49-0.61] |
| always_present | tune | 1240 | 168 | 168/1072/0/0 | 0.14 | 1.00 | 0.24 [0.19-0.29] |
| always_absent | tune | 1240 | 168 | 0/0/168/1072 | n/a | 0.00 | 0.00 [0.00-0.00] |
| rule_asr_lexical | tune | 1240 | 168 | 82/6/86/1066 | 0.93 | 0.49 | 0.64 [0.57-0.71] |
| rule_existing_autolabeler | tune | 1240 | 168 | 91/41/77/1031 | 0.69 | 0.54 | 0.61 [0.53-0.68] |
| mrcd_checkpoint | tune | 1240 | 168 | 87/53/81/1019 | 0.62 | 0.52 | 0.56 [0.48-0.63] |
| always_present | test | 1223 | 145 | 145/1078/0/0 | 0.12 | 1.00 | 0.21 [0.15-0.28] |
| always_absent | test | 1223 | 145 | 0/0/145/1078 | n/a | 0.00 | 0.00 [0.00-0.00] |
| rule_asr_lexical | test | 1223 | 145 | 65/7/80/1071 | 0.90 | 0.45 | 0.60 [0.49-0.68] |
| rule_existing_autolabeler | test | 1223 | 145 | 69/25/76/1053 | 0.73 | 0.48 | 0.58 [0.46-0.67] |
| mrcd_checkpoint | test | 1223 | 145 | 68/46/77/1032 | 0.60 | 0.47 | 0.53 [0.40-0.62] |
| always_present | all | 3716 | 531 | 531/3185/0/0 | 0.14 | 1.00 | 0.25 [0.22-0.28] |
| always_absent | all | 3716 | 531 | 0/0/531/3185 | n/a | 0.00 | 0.00 [0.00-0.00] |
| rule_asr_lexical | all | 3716 | 531 | 260/19/271/3166 | 0.93 | 0.49 | 0.64 [0.60-0.68] |
| rule_existing_autolabeler | all | 3716 | 531 | 284/117/247/3068 | 0.71 | 0.53 | 0.61 [0.56-0.65] |
| mrcd_checkpoint | all | 3716 | 531 | 256/143/275/3042 | 0.64 | 0.48 | 0.55 [0.50-0.59] |

#### DIHANA(turn,audio+ASR) / prolongation

| comparator | partition | n | pos | TP/FP/FN/TN | P | R | F1 [95% CI, bootstrap by group] |
|---|---|---|---|---|---|---|---|
| always_present | dev | 1253 | 224 | 224/1029/0/0 | 0.18 | 1.00 | 0.30 [0.26-0.34] |
| always_absent | dev | 1253 | 224 | 0/0/224/1029 | n/a | 0.00 | 0.00 [0.00-0.00] |
| rule_existing_autolabeler | dev | 1253 | 224 | 26/7/198/1022 | 0.79 | 0.12 | 0.20 [0.13-0.28] |
| mrcd_checkpoint | dev | 1253 | 224 | 76/45/148/984 | 0.63 | 0.34 | 0.44 [0.38-0.51] |
| always_present | tune | 1240 | 249 | 249/991/0/0 | 0.20 | 1.00 | 0.33 [0.28-0.39] |
| always_absent | tune | 1240 | 249 | 0/0/249/991 | n/a | 0.00 | 0.00 [0.00-0.00] |
| rule_existing_autolabeler | tune | 1240 | 249 | 35/5/214/986 | 0.88 | 0.14 | 0.24 [0.17-0.31] |
| mrcd_checkpoint | tune | 1240 | 249 | 95/40/154/951 | 0.70 | 0.38 | 0.49 [0.43-0.56] |
| always_present | test | 1223 | 211 | 211/1012/0/0 | 0.17 | 1.00 | 0.29 [0.25-0.34] |
| always_absent | test | 1223 | 211 | 0/0/211/1012 | n/a | 0.00 | 0.00 [0.00-0.00] |
| rule_existing_autolabeler | test | 1223 | 211 | 28/6/183/1006 | 0.82 | 0.13 | 0.23 [0.15-0.31] |
| mrcd_checkpoint | test | 1223 | 211 | 92/45/119/967 | 0.67 | 0.44 | 0.53 [0.43-0.62] |
| always_present | all | 3716 | 684 | 684/3032/0/0 | 0.18 | 1.00 | 0.31 [0.28-0.34] |
| always_absent | all | 3716 | 684 | 0/0/684/3032 | n/a | 0.00 | 0.00 [0.00-0.00] |
| rule_existing_autolabeler | all | 3716 | 684 | 89/18/595/3014 | 0.83 | 0.13 | 0.23 [0.18-0.27] |
| mrcd_checkpoint | all | 3716 | 684 | 263/130/421/2902 | 0.67 | 0.38 | 0.49 [0.44-0.53] |

#### DIHANA(turn,audio+ASR) / repetition

| comparator | partition | n | pos | TP/FP/FN/TN | P | R | F1 [95% CI, bootstrap by group] |
|---|---|---|---|---|---|---|---|
| always_present | dev | 1253 | 70 | 70/1183/0/0 | 0.06 | 1.00 | 0.11 [0.08-0.14] |
| always_absent | dev | 1253 | 70 | 0/0/70/1183 | n/a | 0.00 | 0.00 [0.00-0.00] |
| rule_asr_lexical | dev | 1253 | 70 | 35/23/35/1160 | 0.60 | 0.50 | 0.55 [0.45-0.64] |
| rule_existing_autolabeler | dev | 1253 | 70 | 31/21/39/1162 | 0.60 | 0.44 | 0.51 [0.41-0.60] |
| mrcd_checkpoint | dev | 1253 | 70 | 28/24/42/1159 | 0.54 | 0.40 | 0.46 [0.35-0.56] |
| always_present | tune | 1240 | 60 | 60/1180/0/0 | 0.05 | 1.00 | 0.09 [0.07-0.12] |
| always_absent | tune | 1240 | 60 | 0/0/60/1180 | n/a | 0.00 | 0.00 [0.00-0.00] |
| rule_asr_lexical | tune | 1240 | 60 | 29/25/31/1155 | 0.54 | 0.48 | 0.51 [0.39-0.60] |
| rule_existing_autolabeler | tune | 1240 | 60 | 24/20/36/1160 | 0.55 | 0.40 | 0.46 [0.34-0.56] |
| mrcd_checkpoint | tune | 1240 | 60 | 28/21/32/1159 | 0.57 | 0.47 | 0.51 [0.41-0.60] |
| always_present | test | 1223 | 60 | 60/1163/0/0 | 0.05 | 1.00 | 0.09 [0.06-0.12] |
| always_absent | test | 1223 | 60 | 0/0/60/1163 | n/a | 0.00 | 0.00 [0.00-0.00] |
| rule_asr_lexical | test | 1223 | 60 | 21/14/39/1149 | 0.60 | 0.35 | 0.44 [0.28-0.58] |
| rule_existing_autolabeler | test | 1223 | 60 | 18/13/42/1150 | 0.58 | 0.30 | 0.40 [0.24-0.53] |
| mrcd_checkpoint | test | 1223 | 60 | 16/18/44/1145 | 0.47 | 0.27 | 0.34 [0.19-0.47] |
| always_present | all | 3716 | 190 | 190/3526/0/0 | 0.05 | 1.00 | 0.10 [0.08-0.11] |
| always_absent | all | 3716 | 190 | 0/0/190/3526 | n/a | 0.00 | 0.00 [0.00-0.00] |
| rule_asr_lexical | all | 3716 | 190 | 85/62/105/3464 | 0.58 | 0.45 | 0.50 [0.43-0.57] |
| rule_existing_autolabeler | all | 3716 | 190 | 73/54/117/3472 | 0.57 | 0.38 | 0.46 [0.39-0.52] |
| mrcd_checkpoint | all | 3716 | 190 | 72/63/118/3463 | 0.53 | 0.38 | 0.44 [0.37-0.50] |

#### DIHANA(turn,audio+ASR) / revision

| comparator | partition | n | pos | TP/FP/FN/TN | P | R | F1 [95% CI, bootstrap by group] |
|---|---|---|---|---|---|---|---|
| always_present | dev | 1253 | 37 | 37/1216/0/0 | 0.03 | 1.00 | 0.06 [0.04-0.08] |
| always_absent | dev | 1253 | 37 | 0/0/37/1216 | n/a | 0.00 | 0.00 [0.00-0.00] |
| rule_asr_lexical | dev | 1253 | 37 | 2/0/35/1216 | 1.00 | 0.05 | 0.10 [0.00-0.25] |
| rule_existing_autolabeler | dev | 1253 | 37 | 0/3/37/1213 | 0.00 | 0.00 | 0.00 [0.00-0.00] |
| mrcd_checkpoint | dev | 1253 | 37 | 2/5/35/1211 | 0.29 | 0.05 | 0.09 [0.00-0.22] |
| always_present | tune | 1240 | 48 | 48/1192/0/0 | 0.04 | 1.00 | 0.07 [0.05-0.10] |
| always_absent | tune | 1240 | 48 | 0/0/48/1192 | n/a | 0.00 | 0.00 [0.00-0.00] |
| rule_asr_lexical | tune | 1240 | 48 | 3/0/45/1192 | 1.00 | 0.06 | 0.12 [0.00-0.23] |
| rule_existing_autolabeler | tune | 1240 | 48 | 5/2/43/1190 | 0.71 | 0.10 | 0.18 [0.05-0.29] |
| mrcd_checkpoint | tune | 1240 | 48 | 5/4/43/1188 | 0.56 | 0.10 | 0.18 [0.05-0.28] |
| always_present | test | 1223 | 36 | 36/1187/0/0 | 0.03 | 1.00 | 0.06 [0.04-0.08] |
| always_absent | test | 1223 | 36 | 0/0/36/1187 | n/a | 0.00 | 0.00 [0.00-0.00] |
| rule_asr_lexical | test | 1223 | 36 | 2/1/34/1186 | 0.67 | 0.06 | 0.10 [0.00-0.21] |
| rule_existing_autolabeler | test | 1223 | 36 | 1/1/35/1186 | 0.50 | 0.03 | 0.05 [0.00-0.17] |
| mrcd_checkpoint | test | 1223 | 36 | 3/4/33/1183 | 0.43 | 0.08 | 0.14 [0.00-0.27] |
| always_present | all | 3716 | 121 | 121/3595/0/0 | 0.03 | 1.00 | 0.06 [0.05-0.08] |
| always_absent | all | 3716 | 121 | 0/0/121/3595 | n/a | 0.00 | 0.00 [0.00-0.00] |
| rule_asr_lexical | all | 3716 | 121 | 7/1/114/3594 | 0.88 | 0.06 | 0.11 [0.04-0.18] |
| rule_existing_autolabeler | all | 3716 | 121 | 6/6/115/3589 | 0.50 | 0.05 | 0.09 [0.03-0.15] |
| mrcd_checkpoint | all | 3716 | 121 | 10/13/111/3582 | 0.43 | 0.08 | 0.14 [0.07-0.21] |

#### DIHANA(turn,audio+ASR) turns>=3s / filler_word

| comparator | partition | n | pos | TP/FP/FN/TN | P | R | F1 [95% CI, bootstrap by group] |
|---|---|---|---|---|---|---|---|
| always_present | dev | 470 | 151 | 151/319/0/0 | 0.32 | 1.00 | 0.49 [0.42-0.55] |
| always_absent | dev | 470 | 151 | 0/0/151/319 | n/a | 0.00 | 0.00 [0.00-0.00] |
| rule_asr_lexical | dev | 470 | 151 | 72/1/79/318 | 0.99 | 0.48 | 0.64 [0.56-0.71] |
| rule_existing_autolabeler | dev | 470 | 151 | 83/21/68/298 | 0.80 | 0.55 | 0.65 [0.58-0.71] |
| mrcd_checkpoint | dev | 470 | 151 | 80/33/71/286 | 0.71 | 0.53 | 0.61 [0.53-0.67] |
| always_present | tune | 479 | 125 | 125/354/0/0 | 0.26 | 1.00 | 0.41 [0.33-0.49] |
| always_absent | tune | 479 | 125 | 0/0/125/354 | n/a | 0.00 | 0.00 [0.00-0.00] |
| rule_asr_lexical | tune | 479 | 125 | 57/2/68/352 | 0.97 | 0.46 | 0.62 [0.53-0.71] |
| rule_existing_autolabeler | tune | 479 | 125 | 67/26/58/328 | 0.72 | 0.54 | 0.61 [0.52-0.70] |
| mrcd_checkpoint | tune | 479 | 125 | 72/45/53/309 | 0.62 | 0.58 | 0.60 [0.50-0.67] |
| always_present | test | 444 | 110 | 110/334/0/0 | 0.25 | 1.00 | 0.40 [0.30-0.49] |
| always_absent | test | 444 | 110 | 0/0/110/334 | n/a | 0.00 | 0.00 [0.00-0.00] |
| rule_asr_lexical | test | 444 | 110 | 48/1/62/333 | 0.98 | 0.44 | 0.60 [0.48-0.69] |
| rule_existing_autolabeler | test | 444 | 110 | 51/12/59/322 | 0.81 | 0.46 | 0.59 [0.46-0.68] |
| mrcd_checkpoint | test | 444 | 110 | 56/41/54/293 | 0.58 | 0.51 | 0.54 [0.41-0.64] |
| always_present | all | 1393 | 386 | 386/1007/0/0 | 0.28 | 1.00 | 0.43 [0.39-0.48] |
| always_absent | all | 1393 | 386 | 0/0/386/1007 | n/a | 0.00 | 0.00 [0.00-0.00] |
| rule_asr_lexical | all | 1393 | 386 | 177/4/209/1003 | 0.98 | 0.46 | 0.62 [0.57-0.67] |
| rule_existing_autolabeler | all | 1393 | 386 | 201/59/185/948 | 0.77 | 0.52 | 0.62 [0.57-0.67] |
| mrcd_checkpoint | all | 1393 | 386 | 208/119/178/888 | 0.64 | 0.54 | 0.58 [0.53-0.63] |

#### DIHANA(turn,audio+ASR) turns>=3s / prolongation

| comparator | partition | n | pos | TP/FP/FN/TN | P | R | F1 [95% CI, bootstrap by group] |
|---|---|---|---|---|---|---|---|
| always_present | dev | 470 | 191 | 191/279/0/0 | 0.41 | 1.00 | 0.58 [0.52-0.63] |
| always_absent | dev | 470 | 191 | 0/0/191/279 | n/a | 0.00 | 0.00 [0.00-0.00] |
| rule_existing_autolabeler | dev | 470 | 191 | 25/5/166/274 | 0.83 | 0.13 | 0.23 [0.15-0.30] |
| mrcd_checkpoint | dev | 470 | 191 | 76/45/115/234 | 0.63 | 0.40 | 0.49 [0.42-0.55] |
| always_present | tune | 479 | 209 | 209/270/0/0 | 0.44 | 1.00 | 0.61 [0.54-0.68] |
| always_absent | tune | 479 | 209 | 0/0/209/270 | n/a | 0.00 | 0.00 [0.00-0.00] |
| rule_existing_autolabeler | tune | 479 | 209 | 35/4/174/266 | 0.90 | 0.17 | 0.28 [0.21-0.35] |
| mrcd_checkpoint | tune | 479 | 209 | 95/40/114/230 | 0.70 | 0.45 | 0.55 [0.48-0.62] |
| always_present | test | 444 | 183 | 183/261/0/0 | 0.41 | 1.00 | 0.58 [0.51-0.65] |
| always_absent | test | 444 | 183 | 0/0/183/261 | n/a | 0.00 | 0.00 [0.00-0.00] |
| rule_existing_autolabeler | test | 444 | 183 | 28/6/155/255 | 0.82 | 0.15 | 0.26 [0.17-0.34] |
| mrcd_checkpoint | test | 444 | 183 | 92/45/91/216 | 0.67 | 0.50 | 0.57 [0.48-0.66] |
| always_present | all | 1393 | 583 | 583/810/0/0 | 0.42 | 1.00 | 0.59 [0.55-0.63] |
| always_absent | all | 1393 | 583 | 0/0/583/810 | n/a | 0.00 | 0.00 [0.00-0.00] |
| rule_existing_autolabeler | all | 1393 | 583 | 88/15/495/795 | 0.85 | 0.15 | 0.26 [0.21-0.30] |
| mrcd_checkpoint | all | 1393 | 583 | 263/130/320/680 | 0.67 | 0.45 | 0.54 [0.49-0.58] |

#### DIHANA(turn,audio+ASR) turns>=3s / repetition

| comparator | partition | n | pos | TP/FP/FN/TN | P | R | F1 [95% CI, bootstrap by group] |
|---|---|---|---|---|---|---|---|
| always_present | dev | 470 | 58 | 58/412/0/0 | 0.12 | 1.00 | 0.22 [0.17-0.28] |
| always_absent | dev | 470 | 58 | 0/0/58/412 | n/a | 0.00 | 0.00 [0.00-0.00] |
| rule_asr_lexical | dev | 470 | 58 | 27/13/31/399 | 0.68 | 0.47 | 0.55 [0.43-0.66] |
| rule_existing_autolabeler | dev | 470 | 58 | 23/11/35/401 | 0.68 | 0.40 | 0.50 [0.37-0.62] |
| mrcd_checkpoint | dev | 470 | 58 | 22/16/36/396 | 0.58 | 0.38 | 0.46 [0.33-0.57] |
| always_present | tune | 479 | 52 | 52/427/0/0 | 0.11 | 1.00 | 0.20 [0.14-0.26] |
| always_absent | tune | 479 | 52 | 0/0/52/427 | n/a | 0.00 | 0.00 [0.00-0.00] |
| rule_asr_lexical | tune | 479 | 52 | 24/12/28/415 | 0.67 | 0.46 | 0.55 [0.43-0.63] |
| rule_existing_autolabeler | tune | 479 | 52 | 21/9/31/418 | 0.70 | 0.40 | 0.51 [0.39-0.60] |
| mrcd_checkpoint | tune | 479 | 52 | 24/13/28/414 | 0.65 | 0.46 | 0.54 [0.44-0.62] |
| always_present | test | 444 | 49 | 49/395/0/0 | 0.11 | 1.00 | 0.20 [0.14-0.26] |
| always_absent | test | 444 | 49 | 0/0/49/395 | n/a | 0.00 | 0.00 [0.00-0.00] |
| rule_asr_lexical | test | 444 | 49 | 16/9/33/386 | 0.64 | 0.33 | 0.43 [0.26-0.58] |
| rule_existing_autolabeler | test | 444 | 49 | 13/8/36/387 | 0.62 | 0.27 | 0.37 [0.22-0.49] |
| mrcd_checkpoint | test | 444 | 49 | 12/14/37/381 | 0.46 | 0.24 | 0.32 [0.18-0.44] |
| always_present | all | 1393 | 159 | 159/1234/0/0 | 0.11 | 1.00 | 0.20 [0.17-0.24] |
| always_absent | all | 1393 | 159 | 0/0/159/1234 | n/a | 0.00 | 0.00 [0.00-0.00] |
| rule_asr_lexical | all | 1393 | 159 | 67/34/92/1200 | 0.66 | 0.42 | 0.52 [0.44-0.58] |
| rule_existing_autolabeler | all | 1393 | 159 | 57/28/102/1206 | 0.67 | 0.36 | 0.47 [0.39-0.54] |
| mrcd_checkpoint | all | 1393 | 159 | 58/43/101/1191 | 0.57 | 0.36 | 0.45 [0.38-0.51] |

#### DIHANA(turn,audio+ASR) turns>=3s / revision

| comparator | partition | n | pos | TP/FP/FN/TN | P | R | F1 [95% CI, bootstrap by group] |
|---|---|---|---|---|---|---|---|
| always_present | dev | 470 | 36 | 36/434/0/0 | 0.08 | 1.00 | 0.14 [0.09-0.19] |
| always_absent | dev | 470 | 36 | 0/0/36/434 | n/a | 0.00 | 0.00 [0.00-0.00] |
| rule_asr_lexical | dev | 470 | 36 | 2/0/34/434 | 1.00 | 0.06 | 0.11 [0.00-0.26] |
| rule_existing_autolabeler | dev | 470 | 36 | 0/3/36/431 | 0.00 | 0.00 | 0.00 [0.00-0.00] |
| mrcd_checkpoint | dev | 470 | 36 | 2/5/34/429 | 0.29 | 0.06 | 0.09 [0.00-0.23] |
| always_present | tune | 479 | 44 | 44/435/0/0 | 0.09 | 1.00 | 0.17 [0.11-0.23] |
| always_absent | tune | 479 | 44 | 0/0/44/435 | n/a | 0.00 | 0.00 [0.00-0.00] |
| rule_asr_lexical | tune | 479 | 44 | 3/0/41/435 | 1.00 | 0.07 | 0.13 [0.00-0.25] |
| rule_existing_autolabeler | tune | 479 | 44 | 5/2/39/433 | 0.71 | 0.11 | 0.20 [0.06-0.31] |
| mrcd_checkpoint | tune | 479 | 44 | 5/4/39/431 | 0.56 | 0.11 | 0.19 [0.06-0.30] |
| always_present | test | 444 | 35 | 35/409/0/0 | 0.08 | 1.00 | 0.15 [0.09-0.20] |
| always_absent | test | 444 | 35 | 0/0/35/409 | n/a | 0.00 | 0.00 [0.00-0.00] |
| rule_asr_lexical | test | 444 | 35 | 2/0/33/409 | 1.00 | 0.06 | 0.11 [0.00-0.22] |
| rule_existing_autolabeler | test | 444 | 35 | 1/1/34/408 | 0.50 | 0.03 | 0.05 [0.00-0.17] |
| mrcd_checkpoint | test | 444 | 35 | 3/2/32/407 | 0.60 | 0.09 | 0.15 [0.00-0.29] |
| always_present | all | 1393 | 115 | 115/1278/0/0 | 0.08 | 1.00 | 0.15 [0.12-0.18] |
| always_absent | all | 1393 | 115 | 0/0/115/1278 | n/a | 0.00 | 0.00 [0.00-0.00] |
| rule_asr_lexical | all | 1393 | 115 | 7/0/108/1278 | 1.00 | 0.06 | 0.11 [0.04-0.19] |
| rule_existing_autolabeler | all | 1393 | 115 | 6/6/109/1272 | 0.50 | 0.05 | 0.09 [0.03-0.16] |
| mrcd_checkpoint | all | 1393 | 115 | 10/11/105/1267 | 0.48 | 0.09 | 0.15 [0.08-0.22] |

#### DIHANA T-task: human transcript marks stripped / repetition

| comparator | partition | n | pos | TP/FP/FN/TN | P | R | F1 [95% CI, bootstrap by group] |
|---|---|---|---|---|---|---|---|
| rule_adjacent_repetition_on_transcript | dev | 1253 | 70 | 54/6/16/1177 | 0.90 | 0.77 | 0.83 [0.75-0.90] |
| rule_adjacent_repetition_on_transcript | tune | 1240 | 60 | 43/10/17/1170 | 0.81 | 0.72 | 0.76 [0.63-0.86] |
| rule_adjacent_repetition_on_transcript | test | 1223 | 60 | 43/7/17/1156 | 0.86 | 0.72 | 0.78 [0.68-0.86] |
| rule_adjacent_repetition_on_transcript | all | 3716 | 190 | 140/23/50/3503 | 0.86 | 0.74 | 0.79 [0.74-0.84] |

#### AMERESCO(segment 3-8 s, audio+ASR) / prolongation

| comparator | partition | n | pos | TP/FP/FN/TN | P | R | F1 [95% CI, bootstrap by group] |
|---|---|---|---|---|---|---|---|
| always_present | pilot32 | 32 | 16 | 16/16/0/0 | 0.50 | 1.00 | 0.67 [0.50-0.83] UNSTABLE(<30 pos) |
| always_absent | pilot32 | 32 | 16 | 0/0/16/16 | n/a | 0.00 | 0.00 [0.00-0.00] UNSTABLE(<30 pos) |
| rule_existing_autolabeler | pilot32 | 32 | 16 | 0/0/16/16 | n/a | 0.00 | 0.00 [0.00-0.00] UNSTABLE(<30 pos) |
| mrcd_checkpoint | pilot32 | 32 | 16 | 5/2/11/14 | 0.71 | 0.31 | 0.43 [0.22-0.60] UNSTABLE(<30 pos) |
| always_present | new66 | 66 | 21 | 21/45/0/0 | 0.32 | 1.00 | 0.48 [0.00-0.73] UNSTABLE(<30 pos) |
| always_absent | new66 | 66 | 21 | 0/0/21/45 | n/a | 0.00 | 0.00 [0.00-0.00] UNSTABLE(<30 pos) |
| rule_existing_autolabeler | new66 | 66 | 21 | 2/2/19/43 | 0.50 | 0.10 | 0.16 [0.00-0.36] UNSTABLE(<30 pos) |
| mrcd_checkpoint | new66 | 66 | 21 | 3/6/18/39 | 0.33 | 0.14 | 0.20 [0.00-0.22] UNSTABLE(<30 pos) |
| always_present | all | 98 | 37 | 37/61/0/0 | 0.38 | 1.00 | 0.55 [0.20-0.77] |
| always_absent | all | 98 | 37 | 0/0/37/61 | n/a | 0.00 | 0.00 [0.00-0.00] |
| rule_existing_autolabeler | all | 98 | 37 | 2/2/35/59 | 0.50 | 0.05 | 0.10 [0.00-0.23] |
| mrcd_checkpoint | all | 98 | 37 | 8/8/29/53 | 0.50 | 0.22 | 0.30 [0.22-0.43] |

#### EPIC v2 (sentence; 136 Codex set = DEVELOPMENT, not test) / filler_word

| comparator | partition | n | pos | TP/FP/FN/TN | P | R | F1 [95% CI, bootstrap by group] |
|---|---|---|---|---|---|---|---|
| always_present | dev(all 136) | 136 | 68 | 68/68/0/0 | 0.50 | 1.00 | 0.67 [0.66-0.68] |
| always_absent | dev(all 136) | 136 | 68 | 0/0/68/68 | n/a | 0.00 | 0.00 [0.00-0.00] |
| rule_asr_lexical | dev(all 136) | 136 | 68 | 34/1/34/67 | 0.97 | 0.50 | 0.66 [0.54-0.76] |
| rule_existing_autolabeler | dev(all 136) | 136 | 68 | 29/4/39/64 | 0.88 | 0.43 | 0.57 [0.45-0.67] |
| mrcd_checkpoint | dev(all 136) | 136 | 68 | 45/8/23/60 | 0.85 | 0.66 | 0.74 [0.64-0.82] |
| always_present | all | 136 | 68 | 68/68/0/0 | 0.50 | 1.00 | 0.67 [0.66-0.68] |
| always_absent | all | 136 | 68 | 0/0/68/68 | n/a | 0.00 | 0.00 [0.00-0.00] |
| rule_asr_lexical | all | 136 | 68 | 34/1/34/67 | 0.97 | 0.50 | 0.66 [0.54-0.76] |
| rule_existing_autolabeler | all | 136 | 68 | 29/4/39/64 | 0.88 | 0.43 | 0.57 [0.45-0.67] |
| mrcd_checkpoint | all | 136 | 68 | 45/8/23/60 | 0.85 | 0.66 | 0.74 [0.64-0.82] |

### Text-only tasks (T)


#### ES-Port / T:repetition U false-start (joint, utterance)

| comparator | partition | n | pos | TP/FP/FN/TN | P | R | F1 [95% CI, bootstrap by group] |
|---|---|---|---|---|---|---|---|
| always_absent | dev | 57283 | 6076 | 0/0/6076/51207 | n/a | 0.00 | 0.00 [0.00-0.00] |
| always_present | dev | 57283 | 6076 | 6076/51207/0/0 | 0.11 | 1.00 | 0.19 [0.18-0.20] |
| rule_adjacent_repetition | dev | 57283 | 6076 | 3145/2078/2931/49129 | 0.60 | 0.52 | 0.56 [0.54-0.57] |
| always_absent | tune | 21271 | 2258 | 0/0/2258/19013 | n/a | 0.00 | 0.00 [0.00-0.00] |
| always_present | tune | 21271 | 2258 | 2258/19013/0/0 | 0.11 | 1.00 | 0.19 [0.18-0.20] |
| rule_adjacent_repetition | tune | 21271 | 2258 | 1160/912/1098/18101 | 0.56 | 0.51 | 0.54 [0.51-0.56] |
| always_absent | test | 20744 | 2242 | 0/0/2242/18502 | n/a | 0.00 | 0.00 [0.00-0.00] |
| always_present | test | 20744 | 2242 | 2242/18502/0/0 | 0.11 | 1.00 | 0.20 [0.18-0.21] |
| rule_adjacent_repetition | test | 20744 | 2242 | 1119/798/1123/17704 | 0.58 | 0.50 | 0.54 [0.51-0.56] |
| always_absent | all | 99298 | 10576 | 0/0/10576/88722 | n/a | 0.00 | 0.00 [0.00-0.00] |
| always_present | all | 99298 | 10576 | 10576/88722/0/0 | 0.11 | 1.00 | 0.19 [0.19-0.20] |
| rule_adjacent_repetition | all | 99298 | 10576 | 5424/3788/5152/84934 | 0.59 | 0.51 | 0.55 [0.54-0.56] |

#### disfluentES / T:filler(sentence)

| comparator | partition | n | pos | TP/FP/FN/TN | P | R | F1 [95% CI, bootstrap by group] |
|---|---|---|---|---|---|---|---|
| always_absent | dev | 2567 | 46 | 0/0/46/2521 | n/a | 0.00 | 0.00 [0.00-0.00] |
| always_present | dev | 2567 | 46 | 46/2521/0/0 | 0.02 | 1.00 | 0.04 [0.03-0.05] |
| rule_lexicon_plus_bare_e_OPTIMISTIC | dev | 2567 | 46 | 40/25/6/2496 | 0.62 | 0.87 | 0.72 [0.62-0.82] |
| rule_vocalic_lexicon_apriori | dev | 2567 | 46 | 0/1/46/2520 | 0.00 | 0.00 | 0.00 [0.00-0.00] |
| always_absent | tune | 1047 | 25 | 0/0/25/1022 | n/a | 0.00 | 0.00 [0.00-0.00] UNSTABLE(<30 pos) |
| always_present | tune | 1047 | 25 | 25/1022/0/0 | 0.02 | 1.00 | 0.05 [0.03-0.07] UNSTABLE(<30 pos) |
| rule_lexicon_plus_bare_e_OPTIMISTIC | tune | 1047 | 25 | 23/12/2/1010 | 0.66 | 0.92 | 0.77 [0.60-0.89] UNSTABLE(<30 pos) |
| rule_vocalic_lexicon_apriori | tune | 1047 | 25 | 0/2/25/1020 | 0.00 | 0.00 | 0.00 [0.00-0.00] UNSTABLE(<30 pos) |
| always_absent | test | 892 | 15 | 0/0/15/877 | n/a | 0.00 | 0.00 [0.00-0.00] UNSTABLE(<30 pos) |
| always_present | test | 892 | 15 | 15/877/0/0 | 0.02 | 1.00 | 0.03 [0.02-0.05] UNSTABLE(<30 pos) |
| rule_lexicon_plus_bare_e_OPTIMISTIC | test | 892 | 15 | 15/7/0/870 | 0.68 | 1.00 | 0.81 [0.62-0.94] UNSTABLE(<30 pos) |
| rule_vocalic_lexicon_apriori | test | 892 | 15 | 0/1/15/876 | 0.00 | 0.00 | 0.00 [0.00-0.00] UNSTABLE(<30 pos) |
| always_absent | all | 4506 | 86 | 0/0/86/4420 | n/a | 0.00 | 0.00 [0.00-0.00] |
| always_present | all | 4506 | 86 | 86/4420/0/0 | 0.02 | 1.00 | 0.04 [0.03-0.05] |
| rule_lexicon_plus_bare_e_OPTIMISTIC | all | 4506 | 86 | 78/44/8/4376 | 0.64 | 0.91 | 0.75 [0.68-0.82] |
| rule_vocalic_lexicon_apriori | all | 4506 | 86 | 0/4/86/4416 | 0.00 | 0.00 | 0.00 [0.00-0.00] |

#### disfluentES / T:repetition(sentence)

| comparator | partition | n | pos | TP/FP/FN/TN | P | R | F1 [95% CI, bootstrap by group] |
|---|---|---|---|---|---|---|---|
| always_absent | dev | 2567 | 364 | 0/0/364/2203 | n/a | 0.00 | 0.00 [0.00-0.00] |
| always_present | dev | 2567 | 364 | 364/2203/0/0 | 0.14 | 1.00 | 0.25 [0.22-0.27] |
| rule_adjacent_repetition | dev | 2567 | 364 | 201/16/163/2187 | 0.93 | 0.55 | 0.69 [0.65-0.73] |
| always_absent | tune | 1047 | 144 | 0/0/144/903 | n/a | 0.00 | 0.00 [0.00-0.00] |
| always_present | tune | 1047 | 144 | 144/903/0/0 | 0.14 | 1.00 | 0.24 [0.20-0.28] |
| rule_adjacent_repetition | tune | 1047 | 144 | 79/6/65/897 | 0.93 | 0.55 | 0.69 [0.62-0.75] |
| always_absent | test | 892 | 125 | 0/0/125/767 | n/a | 0.00 | 0.00 [0.00-0.00] |
| always_present | test | 892 | 125 | 125/767/0/0 | 0.14 | 1.00 | 0.25 [0.20-0.28] |
| rule_adjacent_repetition | test | 892 | 125 | 74/5/51/762 | 0.94 | 0.59 | 0.73 [0.65-0.79] |
| always_absent | all | 4506 | 633 | 0/0/633/3873 | n/a | 0.00 | 0.00 [0.00-0.00] |
| always_present | all | 4506 | 633 | 633/3873/0/0 | 0.14 | 1.00 | 0.25 [0.23-0.27] |
| rule_adjacent_repetition | all | 4506 | 633 | 354/27/279/3846 | 0.93 | 0.56 | 0.70 [0.66-0.73] |

### AMERESCO per conversation (pool of 98; pos = <alargamiento/> segments)

| conversation | comparator | n | pos | TP/FP/FN/TN | P | R | F1 |
|---|---|---|---|---|---|---|---|
| BAQ_001_03_16 | always_present | 11 | 2 | 2/9/0/0 | 0.18 | 1.00 | 0.31 |
| BAQ_001_03_16 | always_absent | 11 | 2 | 0/0/2/9 | n/a | 0.00 | 0.00 |
| BAQ_001_03_16 | mrcd_checkpoint | 11 | 2 | 0/0/2/9 | n/a | 0.00 | 0.00 |
| BAQ_001_03_16 | rule_existing_autolabeler | 11 | 2 | 0/0/2/9 | n/a | 0.00 | 0.00 |
| BUE_001_03_19 | always_present | 41 | 19 | 19/22/0/0 | 0.46 | 1.00 | 0.63 |
| BUE_001_03_19 | always_absent | 41 | 19 | 0/0/19/22 | n/a | 0.00 | 0.00 |
| BUE_001_03_19 | mrcd_checkpoint | 41 | 19 | 3/3/16/19 | 0.50 | 0.16 | 0.24 |
| BUE_001_03_19 | rule_existing_autolabeler | 41 | 19 | 0/1/19/21 | 0.00 | 0.00 | 0.00 |
| HAV_001_02_12 | always_present | 20 | 14 | 14/6/0/0 | 0.70 | 1.00 | 0.82 |
| HAV_001_02_12 | always_absent | 20 | 14 | 0/0/14/6 | n/a | 0.00 | 0.00 |
| HAV_001_02_12 | mrcd_checkpoint | 20 | 14 | 3/1/11/5 | 0.75 | 0.21 | 0.33 |
| HAV_001_02_12 | rule_existing_autolabeler | 20 | 14 | 2/1/12/5 | 0.67 | 0.14 | 0.24 |
| MEX_001_02_15 | always_present | 26 | 2 | 2/24/0/0 | 0.08 | 1.00 | 0.14 |
| MEX_001_02_15 | always_absent | 26 | 2 | 0/0/2/24 | n/a | 0.00 | 0.00 |
| MEX_001_02_15 | mrcd_checkpoint | 26 | 2 | 2/4/0/20 | 0.33 | 1.00 | 0.50 |
| MEX_001_02_15 | rule_existing_autolabeler | 26 | 2 | 0/0/2/24 | n/a | 0.00 | 0.00 |