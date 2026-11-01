# MRCD baselines on existing human references (2026-10-30)

New files only. No new human annotation, no automatic judge as reference, original labels kept. Scripts: `scripts/baselines/`. Every run folder has a `config.json` (seed 13, versions, sha256 of inputs, checkpoint `87f95108...b669`, Whisper `model.bin`, and all `mrcd/*.py`). Metrics JSON: `metrics_audio.json`, `text/text_metrics.json`. Per-unit predictions: `dihana/predictions_per_turn.csv`, `ameresco/predictions_per_segment.csv`, `epic/predictions_per_sentence.csv`, `text/disfluentES_predictions.csv`. Error analysis: `error_analysis.json`.

## 0. Read this first

- **Tasks are separate and never pooled.** (S) presence per turn/segment/sentence from audio + the engine's OWN ASR; (T) text-only rules on a human transcript with marks removed. **No temporal-detection (D) metric exists**: no human event boundaries on disk.
- **Inference path used for the MRCD checkpoint:** one acoustic extraction per file + windows built per batch with the CURRENT `mrcd/` (`acoustic_windows`, `prosody_windows`, `linguistic_windows`, 3 s window, 0.5 s stride, terminal window), as in `run_audio_pilot.py` / `verify_current_vs_pilot.py`. NOT `Engine.analyze` (its parity is measured elsewhere). Prediction = any window whose argmax equals the class (pilot rule). Checkpoint is a `label_source=auto` smoke-test checkpoint (VERIFICACION.md): nothing here validates it as a trained model.
- **Zero tuning.** All rules are fixed a priori; thresholds and checkpoint untouched. DIHANA partition (declared before inference): folds 1-3 = development, fold 4 = tuning/selection (used for nothing), fold 5 = reserved test. EPIC 136 = development (already inspected), never a test. Text corpora: sha256(group)%5 buckets (0-2 dev, 3 tune, 4 test), group = speaker (disfluentES) / dialogue (ES-Port).
- **Comparators (same units, same partitions):** always_present, always_absent, `mrcd_checkpoint`, `rule_existing_autolabeler` (mrcd/annotation/auto_labeler detectors, default parameters; NOTE the checkpoint was weakly supervised by this same labeler), `rule_asr_lexical` (my fixed rules on the engine's ASR: vocalic filler lexicon `eh/ehm/em/mm/ah...`; adjacent exact repetition of 1-4 tokens; edit-term lexicon `digo|perdon|quiero decir|mejor dicho` for revision; none for prolongation because Whisper text carries no lengthening). The marked human transcript is NEVER given to the engine.
- **Absence = not marked, not a verified negative**, except where the corpus documents exhaustive marking (DIHANA manual, EPIC README for filled pauses, disfluentES labelled records). AMERESCO negative coverage is unverified.
- **Mapping**: DIHANA `f`->filler, `a`->prolongation (observable lengthening, not proven disfluent), `r:repeticion`->repetition, `r:sustitucion|insercion|borrado`->revision (partial). `b`, `l:cortada`, `d:*`, `p` NOT mapped. No block / rhetorical_pause / neutral_pause evaluation (no reference).
- **Bootstrap CIs** (2000 resamples) resample speakers (DIHANA, EPIC, disfluentES), dialogues (ES-Port). Rows with <30 positives are flagged UNSTABLE. AMERESCO has 4 conversations: a bootstrap over 4 groups is NOT meaningful, read the per-conversation table instead.

## 1. DIHANA audio decoding (blocker resolved)

Objective checks (`dihana/codec_verification.json`, `scripts/baselines/dihana_decode.py verify`): the XML header says "linear 16 bit" but the `.ul` files are **mu-law, 8 kHz, 1 byte/sample**. Evidence: (a) 16.4% of all bytes are 0x7E/0x7F/0xFE/0xFF (mu-law silence) vs 1.5% in the A-law silence zone; (b) total duration 5.40 h vs README "5.5 hours" (linear-16 would give 2.5 h); (c) speech rate median 2.68 words/s (linear-16 hypothesis: 5.4, implausible); (d) Whisper-small WER vs the marks-stripped transcript on 24 dev turns: mu-law 0.16, A-law 0.45, linear-16 LE 2.05, linear-16 BE 1.47, mu-law read as 16 kHz 0.64. Adapter: mu-law decode (G.711 in numpy) + `resample_poly(2,1)` to 16 kHz PCM16 into `dihana/wav16k/` (6279 WAVs, ul/wav sha256 in `wav16k_manifest.json`); originals untouched.

## 2. Executed scope and deviations

| item                                 | planned        | executed                                                                                                                                 |
| ------------------------------------ | -------------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| DIHANA test (fold 5) / tune (fold 4) | all turns      | 1223 / 1240 turns, all 45+45 speakers                                                                                                    |
| DIHANA dev (folds 1-3)               | all 3816 turns | **1253 turns (33%, seeded random order)**; stopped for time. Dev numbers are a sample                                                    |
| AMERESCO                             | pool 3-8 s     | all 98 (the 32 pilot segments + 66 new); recipe reproduces the 32 pilot WAVs byte-identically                                            |
| EPIC                                 | 136 dev        | 136                                                                                                                                      |
| Short turns                          | n/a            | 62% of DIHANA turns are <3 s: zero-padded at the end to 3 s so one window exists (secondary tables "turns>=3s" give the unpadded subset) |

**ASR is not reproduced from the pilot.** Re-running Whisper (CPU int8, same wavs, same model.bin) gives text different from the pilot's cached words in 6/6 clips checked (`asr_determinism.json`), while the current environment is self-consistent (repeat and threads 0/1/4: identical 6/6). Consequence: 3 of 32 AMERESCO model decisions flipped vs the pilot (29/32 same; rule detector 32/32 same). The diff-0 verification holds only with cached ASR words; ASR determinism across environments is NOT established. Cause not isolated.

## 3. Results by task / corpus / class / comparator / partition

Cell format: TP/FP/FN/TN. F1 in brackets = 95% CI.

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
| MEX_001_02_15 | rule_existing_autolabeler | 26 | 2 | 0/0/2/24 | n/a | 0.00 | 0.00 |## 4. Reading of the results (reserved test = DIHANA fold 5 unless stated)

- **filler_word (S, 145 pos, 45 speakers)**: MRCD F1 0.53 [0.40-0.62] (P 0.60, R 0.47); fixed ASR lexical rule F1 0.60 [0.49-0.68] (P 0.90, R 0.45); existing labeler 0.58; always-present 0.21. The checkpoint does not beat a one-line lexical rule on ASR; its only difference is higher false positives (46 vs 7). Tune/dev show the same ordering (0.56 vs 0.64-0.67). Median per-speaker recall (MRCD) 0.40; 20 of 45 speakers have >=3 positives.
- **prolongation (S, 211 pos)**: MRCD F1 0.53 [0.43-0.62] vs existing labeler 0.23 and always-present 0.29. The only class where the checkpoint clearly exceeds the fixed comparators on DIHANA (recall 0.44 vs 0.13). Consistent in tune (0.49) and dev (0.44). Label = lengthening mark of the DIHANA manual, not necessarily disfluent.
- **repetition (S, 60 pos)**: MRCD 0.34 [0.19-0.47], existing 0.40, ASR lexical 0.44 [0.28-0.58]. Intervals overlap: no ordering supported. The separate T-task rule on the human transcript reaches 0.78 [0.68-0.86], i.e. the information is in the words; the audio route loses it (section 5).
- **revision (S, 36 pos)**: every comparator <=0.14, all CIs reach 0.00. UNSTABLE; only conclusion: no comparator detects it.
- **AMERESCO prolongation (98 segments, 37 pos, 4 conversations)**: MRCD P 0.50 R 0.22 F1 0.30 (8/8/29/53); existing rule F1 0.10; always-present F1 0.55 (prevalence 38%) **beats the checkpoint**. Pilot-32 subset F1 0.43 vs the 66 new segments 0.20: the pilot figure was optimistic (balanced 16/16 sample). Per conversation (recall): BUE 3/19, HAV 3/14, BAQ 0/2, MEX 2/2 (with 4 FP). Positives sit in BUE+HAV (33/37): conversation-level generalisation is not measurable with 4 groups.
- **EPIC filler (136 dev, 68 pos, 29 speakers, development only)**: MRCD P 0.85 R 0.66 F1 0.74 [0.64-0.82]; ASR lexical 0.66 [0.54-0.76] (P 0.97, R 0.50); always-present 0.67; existing 0.57. Parliament speech, long sentences: the checkpoint is above rule baselines here, but this set was used in earlier development and is balanced 50/50 by construction.
- **Text-only (T)**: disfluentES repetition, adjacent-repetition rule P 0.93 R 0.56 F1 0.70 (633 pos / 4506 sentences, 329 speakers; test bucket 0.73 [0.65-0.79]); FILL with the a-priori lexicon F1 0.00 (children write the pause as `e`), with the lexicon plus bare `e` (informed by inspecting that same corpus: optimistic, not independent) 0.75 (86 pos). ES-Port `< >` (repetition U false start, joint) rule P 0.59 R 0.51 F1 0.55 over 99,298 utterances / 1170 dialogues (always-present 0.19). ES-Port filler not evaluated (removing `<% >` removes the token; keeping the word would make the rule a lookup of the label).

## 5. Error analysis (FP/FN; `error_analysis.json` has seeded examples per class; no labels created)

Computed over all scored DIHANA turns (3716) for the checkpoint.

- **ASR erases or normalises the phenomenon (repetition, filler).** Repetition: 74% of positive turns show an adjacent repetition in the human transcript, but only 54% of those keep it in the engine's ASR; 62 of 118 FNs are exactly that (e.g. ref "[quote withheld]" -> ASR "el horario"; "Sí sí" -> "Sí, sí." is kept but the label still missed, window argmax `neutral_pause`). Filler: only 49% of filler-positive turns contain a vocalic filler token in the ASR (0.6% of negatives do): Whisper drops many `eh`, capping every ASR-based comparator near R 0.5 (consistent with `asr_chunk_study.md`).
- **3 s window / padding.** 62% of turns are <3 s and padded; filler FNs are 35% padded turns vs 27% of positives, so padding explains little; candidate cause (not tested): the "any window" OR rule over many 3 s windows raises FP on long turns (0% of prolongation FPs and 17% of filler FPs are padded turns, vs 73% / 68% of negatives).
- **Phenomenon vs corpus definition.** Prolongation FPs co-occur with silent pause marks `p` in 72% (vs 16% of negatives) and `r:repeticion` 19% (vs 3%); filler FPs co-occur with `d:rellenar` 24% (vs 2%), i.e. discourse-marker fillers (`mira`, `bueno`) that DIHANA tags `d`, not `f`, and with lengthening `a` 43% (vs 14%): the checkpoint confuses neighbouring phenomena that the corpus separates. Revision FPs/FNs mix `r:repeticion` (15% vs 4.5%) and `b` (15% vs 1%): the 4-way boundary repetition/revision/abandonment is not recoverable from audio+ASR here. Revision FNs: the ASR often removes the reparandum entirely ("[quote withheld]" -> "[quote withheld]").
- **Corpus/domain.** 8 kHz telephone upsampled, Wizard-of-Oz train-information turns, peninsular Spanish, very short turns, versus the YouTube material the checkpoint was weakly supervised on.
- **AMERESCO**: 29 FNs; several FNs are segments whose ASR collapses into `no, no, no...` loops (Whisper hallucination in overlapping/noisy speech), so the linguistic branch is fed garbage. FPs include expressive lengthening not marked by transcribers (negative coverage unverified, a "FP" may be an unmarked true lengthening).
- **EPIC**: FNs are mostly ASR deleting `-ehm-` (50% of positives keep an ASR filler token); negatives almost all contain silent-pause notation `(.)`/`(..)` (66/68), a candidate cause of the FPs (not tested).

## 6. What can NOT be concluded

- Nothing about block, rhetorical_pause, neutral_pause (no reference), or about seven-class accuracy, event boundaries (no D task), video gain, or Engine.analyze behaviour (other inference path).
- Not that the checkpoint is "good" or "bad" in general: it is a weakly supervised smoke-test checkpoint evaluated zero-shot on a telephone domain; no ablation of audio vs text branches was run here.
- DIHANA dev rows are a 33% sample; revision (36 test pos) and AMERESCO (37 pos, 4 conversations) are unstable; EPIC numbers are development, not generalisation; AMERESCO "positive" = transcriber-marked lengthening, "negative" = unmarked.
- ASR text differs from the pilot's cached words: any claim built on the pilot's exact per-clip outputs is not reproducible bit-for-bit end to end.
- Text results (T) use human transcripts of different populations (children reading, tech-support calls) and cannot be compared with audio (S) numbers; the optimistic FILL rule is not independent evidence.
- `rule_existing_autolabeler` is the teacher of the weak labels: its agreement with the checkpoint is not independent evidence.
