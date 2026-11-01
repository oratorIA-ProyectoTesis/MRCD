# V0 on frozen ASR caches (OMP=4)

Cell: TP/FP/FN/TN. F1 CI = 95% bootstrap by speaker/actor. Rows with <30 positives are UNSTABLE.


#### dihana / filler_word

| comparator | partition | n | pos | TP/FP/FN/TN | P | R | F1 |
|---|---|---|---|---|---|---|---|
| always_present | dev | 3816 | 606 | 606/3210/0/0 | 0.16 | 1.00 | 0.27 [0.24-0.31] |
| always_absent | dev | 3816 | 606 | 0/0/606/3210 | n/a | 0.00 | 0.00 [0.00-0.00] |
| rule_existing_autolabeler | dev | 3816 | 606 | 347/164/259/3046 | 0.68 | 0.57 | 0.62 [0.57-0.67] |
| rule_asr_lexical | dev | 3816 | 606 | 321/29/285/3181 | 0.92 | 0.53 | 0.67 [0.62-0.71] |
| mrcd_checkpoint_V0 | dev | 3816 | 606 | 302/148/304/3062 | 0.67 | 0.50 | 0.57 [0.52-0.62] |
| always_present | tune | 1240 | 168 | 168/1072/0/0 | 0.14 | 1.00 | 0.24 [0.19-0.29] |
| always_absent | tune | 1240 | 168 | 0/0/168/1072 | n/a | 0.00 | 0.00 [0.00-0.00] |
| rule_existing_autolabeler | tune | 1240 | 168 | 89/42/79/1030 | 0.68 | 0.53 | 0.60 [0.51-0.67] |
| rule_asr_lexical | tune | 1240 | 168 | 78/6/90/1066 | 0.93 | 0.46 | 0.62 [0.54-0.70] |
| mrcd_checkpoint_V0 | tune | 1240 | 168 | 84/54/84/1018 | 0.61 | 0.50 | 0.55 [0.46-0.63] |
| always_present | test | 1223 | 145 | 145/1078/0/0 | 0.12 | 1.00 | 0.21 [0.15-0.28] |
| always_absent | test | 1223 | 145 | 0/0/145/1078 | n/a | 0.00 | 0.00 [0.00-0.00] |
| rule_existing_autolabeler | test | 1223 | 145 | 65/22/80/1056 | 0.75 | 0.45 | 0.56 [0.45-0.64] |
| rule_asr_lexical | test | 1223 | 145 | 61/5/84/1073 | 0.92 | 0.42 | 0.58 [0.47-0.66] |
| mrcd_checkpoint_V0 | test | 1223 | 145 | 64/44/81/1034 | 0.59 | 0.44 | 0.51 [0.39-0.59] |
| always_present | all | 6279 | 919 | 919/5360/0/0 | 0.15 | 1.00 | 0.26 [0.22-0.29] |
| always_absent | all | 6279 | 919 | 0/0/919/5360 | n/a | 0.00 | 0.00 [0.00-0.00] |
| rule_existing_autolabeler | all | 6279 | 919 | 501/228/418/5132 | 0.69 | 0.55 | 0.61 [0.57-0.65] |
| rule_asr_lexical | all | 6279 | 919 | 460/40/459/5320 | 0.92 | 0.50 | 0.65 [0.61-0.68] |
| mrcd_checkpoint_V0 | all | 6279 | 919 | 450/246/469/5114 | 0.65 | 0.49 | 0.56 [0.52-0.59] |

#### dihana / prolongation

| comparator | partition | n | pos | TP/FP/FN/TN | P | R | F1 |
|---|---|---|---|---|---|---|---|
| always_present | dev | 3816 | 718 | 718/3098/0/0 | 0.19 | 1.00 | 0.32 [0.28-0.35] |
| always_absent | dev | 3816 | 718 | 0/0/718/3098 | n/a | 0.00 | 0.00 [0.00-0.00] |
| rule_existing_autolabeler | dev | 3816 | 718 | 85/17/633/3081 | 0.83 | 0.12 | 0.21 [0.17-0.24] |
| mrcd_checkpoint_V0 | dev | 3816 | 718 | 246/141/472/2957 | 0.64 | 0.34 | 0.45 [0.40-0.49] |
| always_present | tune | 1240 | 249 | 249/991/0/0 | 0.20 | 1.00 | 0.33 [0.28-0.39] |
| always_absent | tune | 1240 | 249 | 0/0/249/991 | n/a | 0.00 | 0.00 [0.00-0.00] |
| rule_existing_autolabeler | tune | 1240 | 249 | 36/5/213/986 | 0.88 | 0.14 | 0.25 [0.18-0.31] |
| mrcd_checkpoint_V0 | tune | 1240 | 249 | 95/39/154/952 | 0.71 | 0.38 | 0.50 [0.42-0.56] |
| always_present | test | 1223 | 211 | 211/1012/0/0 | 0.17 | 1.00 | 0.29 [0.25-0.34] |
| always_absent | test | 1223 | 211 | 0/0/211/1012 | n/a | 0.00 | 0.00 [0.00-0.00] |
| rule_existing_autolabeler | test | 1223 | 211 | 26/6/185/1006 | 0.81 | 0.12 | 0.21 [0.14-0.29] |
| mrcd_checkpoint_V0 | test | 1223 | 211 | 95/46/116/966 | 0.67 | 0.45 | 0.54 [0.44-0.62] |
| always_present | all | 6279 | 1178 | 1178/5101/0/0 | 0.19 | 1.00 | 0.32 [0.29-0.34] |
| always_absent | all | 6279 | 1178 | 0/0/1178/5101 | n/a | 0.00 | 0.00 [0.00-0.00] |
| rule_existing_autolabeler | all | 6279 | 1178 | 147/28/1031/5073 | 0.84 | 0.12 | 0.22 [0.19-0.25] |
| mrcd_checkpoint_V0 | all | 6279 | 1178 | 436/226/742/4875 | 0.66 | 0.37 | 0.47 [0.44-0.51] |

#### dihana / repetition

| comparator | partition | n | pos | TP/FP/FN/TN | P | R | F1 |
|---|---|---|---|---|---|---|---|
| always_present | dev | 3816 | 221 | 221/3595/0/0 | 0.06 | 1.00 | 0.11 [0.09-0.13] |
| always_absent | dev | 3816 | 221 | 0/0/221/3595 | n/a | 0.00 | 0.00 [0.00-0.00] |
| rule_existing_autolabeler | dev | 3816 | 221 | 79/51/142/3544 | 0.61 | 0.36 | 0.45 [0.38-0.51] |
| rule_asr_lexical | dev | 3816 | 221 | 92/54/129/3541 | 0.63 | 0.42 | 0.50 [0.43-0.56] |
| mrcd_checkpoint_V0 | dev | 3816 | 221 | 73/54/148/3541 | 0.57 | 0.33 | 0.42 [0.35-0.48] |
| always_present | tune | 1240 | 60 | 60/1180/0/0 | 0.05 | 1.00 | 0.09 [0.07-0.12] |
| always_absent | tune | 1240 | 60 | 0/0/60/1180 | n/a | 0.00 | 0.00 [0.00-0.00] |
| rule_existing_autolabeler | tune | 1240 | 60 | 23/19/37/1161 | 0.55 | 0.38 | 0.45 [0.32-0.55] |
| rule_asr_lexical | tune | 1240 | 60 | 27/23/33/1157 | 0.54 | 0.45 | 0.49 [0.37-0.58] |
| mrcd_checkpoint_V0 | tune | 1240 | 60 | 27/21/33/1159 | 0.56 | 0.45 | 0.50 [0.39-0.59] |
| always_present | test | 1223 | 60 | 60/1163/0/0 | 0.05 | 1.00 | 0.09 [0.06-0.12] |
| always_absent | test | 1223 | 60 | 0/0/60/1163 | n/a | 0.00 | 0.00 [0.00-0.00] |
| rule_existing_autolabeler | test | 1223 | 60 | 19/17/41/1146 | 0.53 | 0.32 | 0.40 [0.26-0.52] |
| rule_asr_lexical | test | 1223 | 60 | 22/18/38/1145 | 0.55 | 0.37 | 0.44 [0.28-0.57] |
| mrcd_checkpoint_V0 | test | 1223 | 60 | 17/20/43/1143 | 0.46 | 0.28 | 0.35 [0.21-0.48] |
| always_present | all | 6279 | 341 | 341/5938/0/0 | 0.05 | 1.00 | 0.10 [0.09-0.12] |
| always_absent | all | 6279 | 341 | 0/0/341/5938 | n/a | 0.00 | 0.00 [0.00-0.00] |
| rule_existing_autolabeler | all | 6279 | 341 | 121/87/220/5851 | 0.58 | 0.35 | 0.44 [0.38-0.49] |
| rule_asr_lexical | all | 6279 | 341 | 141/95/200/5843 | 0.60 | 0.41 | 0.49 [0.43-0.54] |
| mrcd_checkpoint_V0 | all | 6279 | 341 | 117/95/224/5843 | 0.55 | 0.34 | 0.42 [0.37-0.47] |

#### dihana / revision

| comparator | partition | n | pos | TP/FP/FN/TN | P | R | F1 |
|---|---|---|---|---|---|---|---|
| always_present | dev | 3816 | 113 | 113/3703/0/0 | 0.03 | 1.00 | 0.06 [0.05-0.07] |
| always_absent | dev | 3816 | 113 | 0/0/113/3703 | n/a | 0.00 | 0.00 [0.00-0.00] |
| rule_existing_autolabeler | dev | 3816 | 113 | 2/8/111/3695 | 0.20 | 0.02 | 0.03 [0.00-0.08] |
| rule_asr_lexical | dev | 3816 | 113 | 4/2/109/3701 | 0.67 | 0.04 | 0.07 [0.01-0.14] |
| mrcd_checkpoint_V0 | dev | 3816 | 113 | 6/13/107/3690 | 0.32 | 0.05 | 0.09 [0.03-0.16] |
| always_present | tune | 1240 | 48 | 48/1192/0/0 | 0.04 | 1.00 | 0.07 [0.05-0.10] |
| always_absent | tune | 1240 | 48 | 0/0/48/1192 | n/a | 0.00 | 0.00 [0.00-0.00] |
| rule_existing_autolabeler | tune | 1240 | 48 | 5/2/43/1190 | 0.71 | 0.10 | 0.18 [0.05-0.29] |
| rule_asr_lexical | tune | 1240 | 48 | 3/0/45/1192 | 1.00 | 0.06 | 0.12 [0.00-0.23] |
| mrcd_checkpoint_V0 | tune | 1240 | 48 | 5/5/43/1187 | 0.50 | 0.10 | 0.17 [0.05-0.28] |
| always_present | test | 1223 | 36 | 36/1187/0/0 | 0.03 | 1.00 | 0.06 [0.04-0.08] |
| always_absent | test | 1223 | 36 | 0/0/36/1187 | n/a | 0.00 | 0.00 [0.00-0.00] |
| rule_existing_autolabeler | test | 1223 | 36 | 1/1/35/1186 | 0.50 | 0.03 | 0.05 [0.00-0.17] |
| rule_asr_lexical | test | 1223 | 36 | 2/1/34/1186 | 0.67 | 0.06 | 0.10 [0.00-0.21] |
| mrcd_checkpoint_V0 | test | 1223 | 36 | 2/5/34/1182 | 0.29 | 0.06 | 0.09 [0.00-0.20] |
| always_present | all | 6279 | 197 | 197/6082/0/0 | 0.03 | 1.00 | 0.06 [0.05-0.07] |
| always_absent | all | 6279 | 197 | 0/0/197/6082 | n/a | 0.00 | 0.00 [0.00-0.00] |
| rule_existing_autolabeler | all | 6279 | 197 | 8/11/189/6071 | 0.42 | 0.04 | 0.07 [0.03-0.12] |
| rule_asr_lexical | all | 6279 | 197 | 9/3/188/6079 | 0.75 | 0.05 | 0.09 [0.04-0.14] |
| mrcd_checkpoint_V0 | all | 6279 | 197 | 13/23/184/6059 | 0.36 | 0.07 | 0.11 [0.06-0.17] |

#### epic / filler_word

| comparator | partition | n | pos | TP/FP/FN/TN | P | R | F1 |
|---|---|---|---|---|---|---|---|
| always_present | dev(all 136) | 136 | 68 | 68/68/0/0 | 0.50 | 1.00 | 0.67 [0.66-0.68] |
| always_absent | dev(all 136) | 136 | 68 | 0/0/68/68 | n/a | 0.00 | 0.00 [0.00-0.00] |
| rule_existing_autolabeler | dev(all 136) | 136 | 68 | 30/4/38/64 | 0.88 | 0.44 | 0.59 [0.46-0.69] |
| rule_asr_lexical | dev(all 136) | 136 | 68 | 36/1/32/67 | 0.97 | 0.53 | 0.69 [0.58-0.78] |
| mrcd_checkpoint_V0 | dev(all 136) | 136 | 68 | 44/10/24/58 | 0.81 | 0.65 | 0.72 [0.63-0.79] |
| always_present | all | 136 | 68 | 68/68/0/0 | 0.50 | 1.00 | 0.67 [0.66-0.68] |
| always_absent | all | 136 | 68 | 0/0/68/68 | n/a | 0.00 | 0.00 [0.00-0.00] |
| rule_existing_autolabeler | all | 136 | 68 | 30/4/38/64 | 0.88 | 0.44 | 0.59 [0.46-0.69] |
| rule_asr_lexical | all | 136 | 68 | 36/1/32/67 | 0.97 | 0.53 | 0.69 [0.58-0.78] |
| mrcd_checkpoint_V0 | all | 136 | 68 | 44/10/24/58 | 0.81 | 0.65 | 0.72 [0.63-0.79] |

#### ameresco / prolongation

| comparator | partition | n | pos | TP/FP/FN/TN | P | R | F1 |
|---|---|---|---|---|---|---|---|
| always_present | pilot32 | 32 | 16 | 16/16/0/0 | 0.50 | 1.00 | 0.67 [0.50-0.80] UNSTABLE(<30 pos) |
| always_absent | pilot32 | 32 | 16 | 0/0/16/16 | n/a | 0.00 | 0.00 [0.00-0.00] UNSTABLE(<30 pos) |
| rule_existing_autolabeler | pilot32 | 32 | 16 | 0/0/16/16 | n/a | 0.00 | 0.00 [0.00-0.00] UNSTABLE(<30 pos) |
| mrcd_checkpoint_V0 | pilot32 | 32 | 16 | 5/3/11/13 | 0.62 | 0.31 | 0.42 [0.22-0.53] UNSTABLE(<30 pos) |
| always_present | new66 | 66 | 21 | 21/45/0/0 | 0.32 | 1.00 | 0.48 [0.00-0.73] UNSTABLE(<30 pos) |
| always_absent | new66 | 66 | 21 | 0/0/21/45 | n/a | 0.00 | 0.00 [0.00-0.00] UNSTABLE(<30 pos) |
| rule_existing_autolabeler | new66 | 66 | 21 | 1/2/20/43 | 0.33 | 0.05 | 0.08 [0.00-0.20] UNSTABLE(<30 pos) |
| mrcd_checkpoint_V0 | new66 | 66 | 21 | 2/5/19/40 | 0.29 | 0.10 | 0.14 [0.00-0.22] UNSTABLE(<30 pos) |
| always_present | all | 98 | 37 | 37/61/0/0 | 0.38 | 1.00 | 0.55 [0.20-0.75] |
| always_absent | all | 98 | 37 | 0/0/37/61 | n/a | 0.00 | 0.00 [0.00-0.00] |
| rule_existing_autolabeler | all | 98 | 37 | 1/2/36/59 | 0.33 | 0.03 | 0.05 [0.00-0.12] |
| mrcd_checkpoint_V0 | all | 98 | 37 | 7/8/30/53 | 0.47 | 0.19 | 0.27 [0.19-0.37] |