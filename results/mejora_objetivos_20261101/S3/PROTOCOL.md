# S3 protocol: text route on P1, rule+model fusion, calibration with abstention, prolongation (EXPLORATORY)

Frozen BEFORE any S3 number is computed (hash and UTC time in `PROTOCOL.sha256.json`). Deviations go to dated annexes. English.

## 0. Status
Everything is EXPLORATORY: DIHANA only (telephone, peninsular Spanish), folds 1-4, speaker-grouped nested CV. Fold 5 (historical) is never loaded. No claim is confirmatory; no score is called a calibrated probability unless a data-separated check shows it, and then only within this pool.

## 1. Data and partition
Pool = 5056 user turns of folds 1-4 (`mrcd.asr_recovery.common.load_pool`, counts asserted). Outer partition = parent partition (`group_folds(speakers, 5, 13)`); inner K=3, seed 14. Everything learned (standardisation, C, thresholds, calibrators, abstention bands, signal statistics) comes from the TRAIN rows of the outer fold only.
V0 and ASR P0 text are the frozen caches. Vc is trained on folds 1-3 (in-sample there) and tuned on fold 4: every Vc comparison uses fold-4 units only and is labelled "Vc-favourable, descriptive". Vc-based fusion is nested-CV inside fold 4 (speaker groups).

## 2. E1 - repetition text route on P1 (task 1)
Units: frame of `asr_recovery_20261031/subset.json` (346 positives of the union, 679 sampled negatives, weights w) restricted to label-defined `rep_literal`. Same ASR for every text model: P1 re-decode (CPU int8) for P1 text, P0 re-decode (control) for P0 text. CUDA profiles are NOT used.
Alternatives (max 4): A1 = ctx_LR retrained on P1 text (weighted LR, C in {0.03,0.3,3} by weighted inner-OOF AP, threshold by weighted inner-OOF F1); A2 = same trained on P0 and P1 texts of the train units (augmentation), tested on P1 text.
Comparators on identical units: rule(P1 text), rule(P0 text, descriptive), V0 (stored, P0-ASR audio features), Vc (fold 4 only), ctx_LR(P0 model on P0 text), ctx_LR(P0 model on P1 text).
Family E1 (Holm 4): A1 minus {rule(P1), V0, ctx_LR(P0 text), ctx_LR(P0 model on P1 text)}, paired weighted-F1 delta, speaker bootstrap 5000, seed 13. Everything else descriptive. A1 is the pre-declared route to integrate (opt-in, experimental) regardless of the verdict.

## 3. E2 - rule + model fusion (task 2)
Repetition (frame, P1 text): F-OR = rule(P1) OR (A1 score >= t_or), t_or by weighted F1 on train inner-OOF A1 scores (grid 0.02..0.98 plus "never", i.e. rule alone); F-STACK = weighted L2 LR on [rule flag, logit(A1 inner-OOF score)], C by inner AP, threshold by inner F1.
Filler (5056 turns, P0 ASR): rule_asr_lexical (filler), V0 filler unit score = max window probability. F-OR = rule OR (V0 score >= t); F-STACK = LR on [rule, logit V0 score]; comparator V0thr = V0 score >= t learned on train; V0 argmax and rule as references. Vc filler fusion: fold 4 only, nested inside fold 4, descriptive.
Family E2 (Holm 4): rep F-OR minus rule(P1); rep F-OR minus A1; filler F-OR minus rule; filler F-OR minus V0thr. F-STACK and Vc fusion descriptive.

## 4. E3 - calibration and three-state decision (task 3)
Score sources (OOF): V0 unit score for filler_word, prolongation, repetition(rep_literal, 4989 units); A1 score (frame, weighted); filler F-STACK score. Calibrators (2 alternatives): Platt on logit; isotonic (increasing, out_of_bounds clip). Cross-fitting by speaker groups: calibrator fitted on the outer-train scores (inner-OOF scores for learned models, raw scores for V0), applied to outer-test.
Three states: present if p_cal >= hi, absent if p_cal < lo, else uncertain. Band (lo, hi) chosen in the outer train on inner cross-fitted calibrated scores: maximise coverage subject to accepted error <= 0.75 x error of the no-abstention classifier at the F1 threshold (error = 0-1 error; grid of 5% quantile steps); if infeasible, no band (declared). Final service band = same rule on all pooled OOF scores.
Reports: Brier and ECE with 10 equal-MASS (quantile) bins, raw vs Platt vs isotonic, speaker-bootstrap CI of differences; coverage; error among accepted; accepted precision / NPV; share of positives left uncertain; all-cases F1 with uncertain counted as absent; no-abstention F1 for reference.
Family E3 (Holm 10): Brier(calibrated) minus Brier(raw), 5 sources x 2 methods. ECE and band results descriptive. Wording rule: "calibrated" is used only for a source/method whose ECE difference CI excludes 0 in the improving direction; otherwise "score, not demonstrated calibrated".

## 5. E4 - prolongation (task 4)
Label `lengthening_present` (observable lengthening mark; NOT a disfluency claim). Features: `mrcd.asr_recovery.signals` (word-duration z, gaps, trailing silence) fitted on train rows. Pause-associated FP: FP turns carrying a DIHANA silent-pause mark `p`.
Alternatives: P-a = stack [V0 prolongation score, signal features] on folds 1-4; P-b = stack [Vc score, signal features] inside fold 4. Comparators: V0 argmax, V0thr, Vc frozen decision (fold 4).
Family E4 (Holm 4): P-a minus V0thr (F1 delta); P-a minus V0thr (pause-associated FP count delta); P-b minus Vc (F1); P-b minus Vc (pause-FP count). Bootstrap over speakers.

## 6. Statistics and output
Per class: support, TP/FP/FN, P, R, F1, speakers (total / with positives), 95% speaker-bootstrap CI (2000, seed 13), paired deltas. Weighted metrics for the frame. Negative results are reported.

## 7. Service scope (task 5, engineering, not an experiment)
Default scope = filler_word, prolongation, repetition. block, rhetorical_pause, neutral_pause, revision only with an explicit option, marked `out_of_scope_unvalidated`, never counted in totals/rates by default.
