# O-campaign protocol: best obtainable result for filler_word, prolongation and repetition (EXPLORATORY)

Frozen BEFORE any number of this campaign is computed (hash, UTC date, git head, code hashes: `PROTOCOL.sha256.json`, written by `scripts/o_campaign/freeze_protocol.py`). Deviations go to dated annexes; this file is never edited after freezing. English.

## 0. Status and rules

- EXPLORATORY. All data were already seen in earlier phases. No new annotation; only existing labels: DIHANA marks `f` (filler), `a` (lengthening) and literal repetition `r` (sub-type A of the audit; indeterminate turns are excluded, never negatives); AMERESCO lengthening (external, zero-shot); EPIC FPAUSE (external, zero-shot). disfluentES / ES-Port are text-only and not used.
- Absence of a mark is a negative only under the DIHANA convention already adopted by the importer; a block that cannot be computed makes the configuration unavailable, never a zero.
- DIHANA fold 5 is never loaded by fitting, selection, thresholds, calibration or comparison. A single separate "historical, contaminated" report is produced at the very end (section 9).
- The labels are corpus marks, not double human annotation; "prolongation" means observable lengthening, with no claim that it is disfluent.
- No score is called a calibrated probability.

## 1. Data and partition

- Pool: DIHANA folds 1-4, 5056 user turns, 180 speakers (counts asserted by `mrcd.asr_recovery.common.check_pool_counts`).
  - `filler_word`: 5056 turns (774 positive). `prolongation`: 5056 turns (967 positive). `repetition`: the 1019 label-defined turns of the frozen ASR-recovery frame (214 positives); negatives carry inverse-sampling weights `w` and every repetition metric is weighted.
- Outer: 5 folds by speaker, `mrcd.context_eval.partition.group_folds(seed 13)` = the partition of S3, so out-of-fold predictions of earlier comparators pair unit by unit. Inner: 3 speaker-grouped folds inside each outer train (`inner_splits`).
- ALL selection happens in the inner loop of the outer-train rows: ASR profile, feature blocks, model family, hyper-parameters, fusion with the lexical rule, class threshold, OR-threshold, calibration method and abstention band. The reported figure comes only from the outer test folds. Duration normalisers of the `sig`/`lenx` blocks are fitted on the rows used to fit (inner-train inside the inner loop, outer-train for the final refit). This is unit-tested (flipping outer-test labels or destroying outer-test inputs does not change any selection).

## 2. Search space (explicit and finite; the code is `mrcd/optimize/space.py`, its hash is frozen)

| class        | ASR profiles                 | feature block sets | models | fusion        | configurations |
| ------------ | ---------------------------- | ------------------ | ------ | ------------- | -------------- |
| filler_word  | p0                           | 7                  | 7      | none, or_rule | 98             |
| prolongation | p0                           | 7                  | 7      | none          | 49             |
| repetition   | p0, p1, p0_noloop, p1_noloop | 7                  | 7      | none, or_rule | 308            |

- Profiles are limited to the transcripts that exist in cache: P0 for the whole 5056-turn pool, and p0/p1/p0_noloop/p1_noloop for the 1025-turn repetition frame. No new ASR is run for filler / prolongation; if one is added later (same ASR for every compared branch) it is an annex. The ASR profile is therefore searched only for repetition. `acoustic` (DTW self-similarity, computed on P0 words) is allowed only with p0.
- Blocks: `rule` (lexical filler rule / `rule_rep` of the profile text), `v0` (per-label max and mean of the window probabilities + number of windows), `emb` (max and mean of the pooled 128-d head embeddings), `pause` (internal pause, edge silence, speech fraction, frames), `sig` (timing/confidence signals of `asr_recovery.signals`), `lenx` (per-word duration z-scores, top-3, counts, last word), `ctxfill` (filler candidates with +-2 word context: gaps, duration, neighbours, punctuation, probabilities), `reptxt` (`textfeat` repetition features), `loop` (repeat-run, compression, n-gram counts), `acoustic`. The block sets per class are the table in `space.py`.
- Models: L2 logistic regression C in {0.03, 0.3, 3}; histogram gradient boosting (depth {2,3} x iterations {100,200}, lr 0.05, L2 1.0, min leaf 20). Stacking with the rule and with V0 is expressed through the `rule` and `v0` blocks; `or_rule` is the learned OR fusion of S3. Vc (head trained on folds 1-3, tuned on fold 4) is NOT a feature (in-sample on folds 1-3); it is only a fold-4 comparator.
- Calibration is chosen after the model (section 5), not searched.

## 3. Search procedure and stop rule

- Per outer fold and class: seeded random permutation of the space (`seed 13 + 1000*(fold+1)`), truncated to a budget of **60 configurations**, evaluated in that order. **Stop rule**: budget exhausted, OR 20 consecutive configurations without improving the best inner objective by more than 0.002 (patience). The best inner objective (earliest wins ties) selects the configuration.
- Inner objective: weighted F1 of the pooled inner out-of-fold scores at the threshold chosen on the same scores (grid 0.02..0.98); for `or_rule`, F1 of `rule OR score >= t_or` with `t_or` chosen on the same scores (never-threshold allowed). It is optimistic, equally for every configuration.
- Final per fold: refit on the whole outer train, score the outer test, apply the inner-chosen thresholds. Repetition uses weights in fitting, thresholds and F1.
- Final pipeline (for packaging and external validation): the same search on all folds 1-4 rows (seed 13 + 99).

## 4. Metrics and comparators

- Target metric per class: F1 over all cases (weighted for repetition); secondary: coverage/risk with abstention (section 5). Reported per class: support, TP/FP/FN, P, R, F1 with 95% speaker-bootstrap CI (2000), winning configuration per fold and its stability (profile, block set, model, fusion frequency).
- Fixed comparators (decisions, outer partition identical to S3): filler: lexical rule on ASR, V0 argmax, V0 thresholded (nested), S3-best `F_STACK`, Vc (fold 4 only, Vc-favourable, descriptive). Prolongation: V0 argmax, V0 thresholded, S3 `P_a_stack_V0_signals`, Vc (fold 4 only). Repetition: rule on P0 and on P1, V0, ctx_LR on P1 text (A1), S3 A2, Vc (fold 4 only).
- **Current product default** = V0 argmax decision of the engine (S3 README: defaults V0, P0 unchanged), for every class.

## 5. Calibration and abstention (post-model, inner loop)

- Method in {none, platt, isotonic} with the lowest cross-fitted inner Brier (ties: none first). Band `[lo, hi]` by `calibration.choose_band` on cross-fitted inner scores (max coverage subject to accepted error <= 0.75 x baseline). Reported on outer test: coverage, error among accepted, positives left uncertain, F1 with uncertain = absent, speaker-bootstrap CI of coverage and risk. Uncertain is an abstention, never a negative.

## 6. Comparison family and Holm

- Per class: paired weighted-F1 difference of the optimised pipeline minus each comparator on identical units, speaker bootstrap (5000, seed 13), Holm over the comparators of that class (family sizes: filler 4 [Vc is separate, descriptive], prolongation 3, repetition 5). The promotion family (section 7) is separate: 3 tests (one per class vs the default), Holm over 3.
- Verdict "supported after Holm" only if adjusted p < 0.05 and the 95% CI excludes 0.

## 7. Promotion rule (frozen)

The optimised pipeline becomes the product default ONLY IF, on the outer folds, (a) it beats the current default (V0 argmax) in at least one class with the CI of the paired delta above 0 after Holm (3-test family), AND (b) in no class is it worse (the CI of the paired delta includes or exceeds 0; a class that cannot be evaluated blocks promotion). Otherwise it stays opt-in. Packaging marks `default: false` either way; adoption by the service is the orchestrator's decision. The strict comparison against the best fixed comparator is reported but is not part of the rule.

## 8. Learning curve (data ceiling vs model ceiling)

For each outer fold the winning configuration is retrained on 25/50/75% of the outer-train speakers (deterministic sha order; threshold re-chosen by inner CV) and scored on the same outer test; pooled F1 per fraction with speaker-bootstrap CI, and the paired delta 100% minus 75% (CI). Reading rule (declared now): if the 75% -> 100% gain has a CI including 0 and |gain| < 0.01, the ceiling is "not data-limited with this model/features"; if the curve is still rising (CI above 0), "data-limited". Inner-objective F1 minus outer F1 is reported as the selection optimism.

## 9. External and historical reports

- External zero-shot: the final pipeline (folds 1-4) on EPIC (filler_word, FPAUSE) and AMERESCO (prolongation); both corpora were inspected in earlier phases (declared); domain shift (non-telephone, different language mix) is expected; the engine's earlier numbers are given alongside when available. Not part of selection.
- DIHANA fold 5: one run at the very end of the campaign, final pipeline, filler_word and prolongation only (the repetition-literal label of fold 5 is not derived by the audited importer), labelled "historical, contaminated" (evaluated at least twice before). Never used for any decision.

## 10. Reproducibility

`OMP_NUM_THREADS=4` pinned before numpy; seed 13; `config.json` per run with versions, hashes of inputs (pool, caches, acoustic features), of the protocol and of the code; feature blocks cached in memory per run; per-fold results written atomically and resumable. Heavy compute starts only after `S8_incremental/README_S8.md` exists (the script refuses to run otherwise). CPU only unless a GPU ASR step is added by annex with one ASR for all compared branches.
