# EXPERIMENTOS: contextual repetition / revision (EXPLORATORY, DIHANA folds 1-4)

Protocol frozen first: `PROTOCOL.md` (sha256 `e5c9b5a037b5ca871f9db7f0d78770feb3ea4897d471221f6b80a6cf8a12b920`, frozen 2026-10-31T01:48:14Z, `PROTOCOL.sha256.json`); deviations: `PROTOCOL_ANNEX_20261031.md`. Full numeric tables: `TABLES.md`. Raw: `run_main/` (`results.json`, `predictions.csv` with unit / speaker / task / condition / model / score / decision / y, `config.json`, post-hoc JSONs).

**Status of every number here: EXPLORATORY.** No independent test partition exists (DIME / ESLORA have no labels; DIHANA fold 5 is historical; EPIC / AMERESCO have no repetition or revision reference). One corpus, telephone speech, peninsular Spanish, train-information domain, unknown inter-annotator agreement. Nothing is confirmed.

## 1. Setup (as executed)
* Pool: 5056 DIHANA user turns of folds 1-4, 180 speakers, 720 dialogues. Fold 5 is never loaded (loader assertion, test). Nested speaker-grouped CV: outer K=5, inner K=3; no speaker and no dialogue crosses a fold (checked at run time, recorded in `config.json`, tested). C, threshold, standardisation and Platt calibrator are learned only in each outer train.
* Reference re-derived from the XML and parity-tested against the audit (sub-type counts on folds 1-3 identical: A 207, B 41, C 17, D 1; R1 4, R2 48, R3 1, R4 64). `rep_literal`: 214 positive turns / 84 speakers / 4989 units (67 turns whose only repetition marks are B / C / D are excluded as indeterminate: 5056 - 4989). `rep_full`: 281 / 114 / 5056. `revision`: 161 / 91 / 5056. All three meet the gate (>= 30 positives, >= 15 positive speakers).
* Conditions: A = marks-stripped reference transcript; B = frozen ASR (`asr_cache`, OMP=4). V0 / Vc exist only in B (no timed words for A; no alignment, no invented times: N/A).
* Models: `rule` (1-4 token immediate repetition; revision: edit-term lexicon or cut fragment), `V0`, `Vc` (fold 4 only is fair, and even then Vc-favourable because fold 4 tuned it), `ctx_LR` (the one contextual variant: candidates by +-k words with gaps, fragments, fillers, punctuation, position, evidence before and after; L2 logistic regression, C in {0.03, 0.3, 3}).

## 2. Pre-declared family (Holm over 4; paired F1 difference ctx_LR minus rule, identical units, speaker bootstrap 5000)
| task | cond | dF1 | 95% CI | p_Holm | verdict |
|---|---|---|---|---|---|
| rep_literal | A | +0.046 | [0.016, 0.082] | 0.0072 | supported after Holm |
| rep_literal | B | -0.004 | [-0.036, 0.028] | 0.81 | not demonstrated |
| revision | A | +0.137 | [0.048, 0.223] | 0.0072 | supported after Holm |
| revision | B | +0.140 | [0.073, 0.207] | 0.0016 | supported after Holm |

Read these with the caveats of sections 4 and 6: A / repetition is a circular diagnostic (the label is derived from the same marked text); the revision comparator is a rigid lexicon; and the post-hoc check shows turn length alone reaches revision F1 0.23-0.26. "Supported after Holm" means only that this exploratory contrast survives the correction on this pool.

## 3. Main results (F1, 95% speaker-bootstrap CI; full confusion in TABLES.md)
| task | cond | rule | V0 | Vc (fold 4 only, Vc-favourable) | ctx_LR |
|---|---|---|---|---|---|
| rep_literal (214 pos) | A | 0.923 [0.89, 0.95] | N/A | N/A | 0.970 [0.95, 0.99] |
| rep_literal | B | 0.541 [0.48, 0.59] | 0.487 [0.42, 0.54] | 0.500 [0.33, 0.62] (rule 0.500, V0 0.529, ctx 0.452 on the same 1223 units) | 0.537 [0.48, 0.59] |
| rep_full (281 pos, approximate) | B | 0.499 [0.44, 0.55] | 0.439 [0.38, 0.49] | 0.475 (rule 0.491, V0 0.500, ctx 0.421; 1240 units) | 0.486 [0.42, 0.54] |
| revision (161 pos) | A | 0.208 [0.13, 0.28] | N/A | N/A | 0.345 [0.28, 0.40] |
| revision | B | 0.146 [0.09, 0.21] | 0.116 [0.06, 0.17] | = V0 (no revision head) | 0.286 [0.23, 0.34] |

* Repetition, B: nothing separates the rule, V0, Vc and ctx_LR beyond noise except that the rule beats V0 (paired +0.054 [0.011, 0.097], descriptive). Every paired fold-4 difference against Vc has a CI containing 0 (45 speakers, 23 with positives). The negative result stands: with the same frozen ASR text, the contextual text model does not beat a one-line rule, and the audio-based V0 / Vc do not beat it either.
* Historical, fold 5, copied as is (not used for anything): earlier-study repetition mark V0 0.351 (60 pos), Vc 0.318; revision V0 0.093 (36 pos). Same order of magnitude as folds 1-4 (V0 0.439 on `rep_full`, 0.116 on revision).
* ctx_LR revision: precision 0.21-0.26, recall 0.44-0.51 (336 / 314 flagged turns for 161 positives): a screening score, not a detector.

## 4. Error decomposition (A vs B) and why things happen
Positives destroyed by the frozen ASR (human span not recoverable): `rep_literal` 109 / 214 (51%), `rep_full` 176 / 281 (63%), `revision` 108 / 161 (67%). Rule recall on visible literal positives 104 / 105, on destroyed ones 4 / 109; V0: 87 / 105 and 6 / 109; ctx_LR 98 / 105 and 11 / 109.

F1 lost from A to B (separate fits of the same model type, order-dependent split declared in the protocol):
| task | model | F1 A | F1 B | loss | of which on units without destroyed positives | of which attributed to destroyed positives |
|---|---|---|---|---|---|---|
| rep_literal | rule | 0.923 | 0.541 | 0.382 | 0.140 | 0.242 (63%) |
| rep_literal | ctx_LR | 0.970 | 0.537 | 0.433 | 0.282 | 0.150 (35%) |
| revision | rule | 0.208 | 0.146 | 0.063 | -0.031 | 0.093 |
| revision | ctx_LR | 0.345 | 0.286 | 0.060 | 0.001 | 0.058 |

* **Why repetition does not beat simple rules.** (i) Half of the literal repeats are not in the ASR text (Whisper collapses them), a ceiling on any text rule (recall about 0.5) that no text model can lift (recall on destroyed positives 4-10%). (ii) On the visible half the rule already has recall 0.99; there is nothing left for context to add. (iii) What context can fix, FPs, works only when the text is clean: in A it cuts FPs from 32 to 6 (punctuation between copies, polarity words "no, no" / "sí, sí" which DIHANA does not mark); in B the ASR re-punctuates and the FP count does not fall (77 vs 83). Post-hoc causes of the 77 rule FPs in B: 25 repeated polarity word, 6 ASR loop, 46 other; 27 of them also have an unmarked adjacent duplicate in the human transcript (label-convention ambiguity, not necessarily a detector error). (iv) V0 / Vc are below the rule: the audio branch has no literal-copy comparison and the window argmax competes among eight classes.
* **Why revision is not detected.** 94 of the 161 positive turns are silent substitutions / insertions / deletions with no lexical trace (R4, 58%); only 7 carry an explicit edit signal; cut-word restarts (R2, 61 turns) almost never survive the ASR (1 / 48 marks in the audit). The edit-lexicon rule detects 4 / 7 explicit signals and 14 / 94 silent ones in A. ctx_LR reaches recall 0.44 (B) mostly through turn length and repeat structure: the length / punctuation-only variant scores F1 0.257 (A) and 0.232 (B) vs 0.345 and 0.286, so the genuinely contextual excess is about +0.09 (A) and +0.05 (B), untested. FP / FN examples (selected by hash order, no labels created) are in `run_main/results.json` (`error_examples`): FNs are mostly silent edits and ASR-erased fragments ("[quote withheld]" becomes "las cuatro"); ctx_LR FPs are long turns containing repeats.

## 5. Calibration (all three tasks meet the support gate; ctx_LR only)
Cross-fitted Platt scaling (calibrator fitted on the inner-OOF scores of each outer train), pooled OOF, 10 equal-width bins. Brier raw vs calibrated differs by < 0.0002 with CIs containing 0 (revision B: 0.00008 [0.00001, 0.00015], negligible); ECE differences have CIs containing 0 everywhere. ECE is dominated by the first bin (about 94% of units have score < 0.1). Raw Brier for `rep_literal` B is 0.0271 vs 0.0411 for a constant at the prevalence. Reliability (B): `rep_literal` mid-score bins are over-confident (score 0.14 / observed 0.07, 0.25 / 0.15, 0.46 / 0.38) and roughly acceptable above 0.6; `revision` upper bins have n <= 5 and are not interpretable (score 0.64 / observed 0.20). Conclusion: Platt calibration gave no demonstrated gain and **no claim of calibrated probability is made**; scores are rankings. Abstention was not implemented (nothing to promote from one partition).

## 6. Limitations
Exploratory single corpus; folds 1-4 reused for model fitting, selection and evaluation through nested CV only; official folds 1-3 trained Vc (so Vc is only fair on fold 4, which also tuned it); `rep_literal` excludes 67 indeterminate turns; A-condition repetition is circular; ASR condition B text-only (word times, confidences and audio are not used by ctx_LR; V0 / Vc use audio but not literal-copy comparison); C was at the grid edge (0.03) in most revision folds, so the regularisation grid may be too narrow (not extended: no open search); the survival heuristic is the audit's, not validated by listening; no inter-annotator agreement; Platt calibration assumes inner-OOF and full-train scores are comparable; DIHANA fillers and pauses are not in the A text but are in the ASR text.

## 7. Functioning check on utt74 (route verification, NOT a metric)
V0 on `utt74.wav` (3.339 s, recipe of `run_mrcd.py`, two windows 0.00 and 0.34 s): emits nothing (all windows argmax `fluent`, revision probability 0.000, max non-fluent prolongation 0.29). The frozen-recipe ASR returned "[quote withheld]": the reparandum "un" of the reference event (revision 0.538-1.813 s, "[quote withheld]") is not in the ASR text, i.e. the same destruction measured above. The rule detectors also emitted nothing. n = 1, a segment of dialogue d12; this proves the path runs, nothing about detection.

## 8. Status of the seven classes
| class | status | basis |
|---|---|---|
| filler_word | reference compatible; exploratory | DIHANA `f` (turn level), EPIC; earlier study: Vc only ties a one-line ASR rule |
| prolongation | partial approximation; exploratory | DIHANA `a` = lengthening mark, AMERESCO `alargamiento`; not necessarily a disfluent prolongation |
| repetition | reference compatible for literal repeats (sub-type A, 78% of marks), partial approximation for the full mark; exploratory (this experiment) | section 3-4 |
| revision | partial approximation (grouped mark, 58% silent edits); exploratory; utt74 is one exact example (n=1) | section 3-4 |
| block | no reference | no annotated blocks anywhere; silence != block |
| rhetorical_pause | no reference | duration != rhetorical |
| neutral_pause | no reference | absence of a mark != neutral pause |
Confirmed: none. Classes without reference are not averaged as F1 = 0. Only previously measured sensitivities exist for the three pause / block classes (`properties_noref.json`, `MEJORAS.md` sec. 3: padding flips rhetorical / neutral emission in 42-46% of short turns, emission follows duration and pause length); this is not validity and truncation != revision. No defect was fixed (needs edits outside the allowed scope, Annex item 3).

## 9. Reproduction (repo root, `.venv\Scripts\python.exe`; OMP=4 is set inside every script before numpy)
```
python scripts/rep_rev/freeze_protocol.py          # already done; refuses to overwrite
python -I scripts/rep_rev/run_experiment.py         # about 80 s the first time (builds _cache/pool.pkl), writes run_main/
python -I scripts/rep_rev/posthoc_diagnostics.py    # annex item 2a
python -I scripts/rep_rev/posthoc_error_causes.py   # annex item 2b
python -I scripts/rep_rev/historical_fold5.py       # fold 5 re-tabulated as is
python -I scripts/rep_rev/utt74_check.py            # route check (loads Whisper)
python -I scripts/rep_rev/render_tables.py          # TABLES.md
OMP_NUM_THREADS=4 python -m pytest tests/test_rep_rev_*.py -q
```
Hashes: `run_main/config.json` (inputs, caches, head checkpoint, code), `config_auxiliary.json` (base checkpoint `87f95108...b669`, Whisper `model.bin`, scripts). Package: `mrcd/context_eval/` (partition, metrics, reference, survival, textfeat, model, pool).


---
# ADDENDUM 2026-10-30 (after the independent round-3 review, `revision3/REVISION3.md` sections 2-5). Nothing above was edited; where this addendum is narrower, THIS addendum wins.

Reviewer: independent in round 3 (a different agent). The corrections below were written by the same writer who produced the experiments (self-review), not by a new independent pass. The reviewer's figures below were re-computed by the reviewer's own scripts (`scripts/review3/`), not by this writer, except item 5, which this writer computed (`scripts/rep_rev/calibration_quantile.py`).

1. **The revision result is a screening result, and most of its gain over the rule is turn length.** "Supported after Holm" for revision A / B means only: a length-aware learned operating point with a tuned threshold beats a rigid edit-lexicon rule. It is not evidence that context matters. Reviewer figures (same nested protocol, condition B): length / punctuation / filler-only model F1 0.232; `log_len` alone 0.225; the length-only model beats the rule by +0.087 [0.015, 0.155] (post hoc, p .021); the full ctx_LR beats the length-only model by +0.053 [0.017, 0.091] (post hoc, unadjusted, p .004, NOT part of the Holm family, not confirmatory); full minus `log_len` alone +0.060 [0.019, 0.101]. Section 4's "genuinely contextual excess about +0.05, untested" is therefore "nominally positive, post hoc, unadjusted".
2. **Condition A is circular and depends on the transcribers' comma convention.** The "marks-stripped" transcript keeps the transcribers' punctuation, and punctuation tracks the annotation decision. Adjacent identical-word turns (4989 `rep_literal` units): marked positives (y=1): 195 with a plain duplicate, 19 with none; unmarked (y=0): 4 plain duplicate, 19 comma-separated (`no, no`), 4752 none. Comma-separated copies are never marked (0 / 19). So A's rule 0.923 -> ctx_LR 0.970 is largely learning the DIHANA punctuation convention. Do not quote A as detector performance. The feature design was also done after the folds 1-3 sub-type audit (nested CV cannot protect against design-time knowledge): possible small optimism, already labelled exploratory.
3. **"The ASR destroys 51% / 63% / 67%" holds only for this transcript:** the frozen faster-whisper small int8 with 8 s chunks, not "the ASR" and not the app's Speechmatics (untested). Of the 109 destroyed `rep_literal` positives, 102 are collapsed repetitions (45: span >= 80% present but no adjacent duplicate; 57: one copy present), 7 not recognised, 0 empty. "Visible" is nearly "an adjacent duplicate exists", so rule recall on visible positives (104 / 105) is close to tautological. The 67% for revision uses a heuristic never validated by listening: call it a heuristic.
4. **Untested hypotheses.** In section 4, reason (iv) ("the audio branch has no literal-copy comparison and the window argmax competes among eight classes") and the claim that "the ASR re-punctuates" are NOT proven; reasons (i)-(iii) are supported by counts. The revision upper-bin reliability (n <= 5) is not interpretable.
5. **Calibration: "no demonstrated gain", not "no gain".** The 10 equal-width bins put ~94% of units in the first bin, so that ECE is nearly blind to the mid-score over-confidence (0.14 vs 0.07, 0.25 vs 0.15), and n = 214 positives gives little power. Re-computed with equal-mass (quantile) bins (10 bins, same predictions, same speaker bootstrap; post hoc; `calibration_quantile.json`):

| task | cond | n / pos | ECE equal-width raw | ECE quantile raw | ECE quantile Platt | dECE_q raw - Platt [95% CI] |
|---|---|---|---|---|---|---|
| rep_full | A | 5056 / 281 | 0.0055 | 0.0021 | 0.0024 | -0.0002 [-0.0021, +0.0018] |
| rep_full | B | 5056 / 281 | 0.0081 | 0.0116 | 0.0106 | +0.0009 [-0.0024, +0.0031] |
| rep_literal | A | 4989 / 214 | 0.0023 | 0.0012 | 0.0007 | +0.0005 [-0.0010, +0.0012] |
| rep_literal | B | 4989 / 214 | 0.0075 | 0.0077 | 0.0061 | +0.0015 [-0.0014, +0.0034] |
| revision | A | 5056 / 161 | 0.0073 | 0.0052 | 0.0043 | +0.0009 [-0.0011, +0.0017] |
| revision | B | 5056 / 161 | 0.0050 | 0.0074 | 0.0080 | -0.0006 [-0.0021, +0.0005] |

   Every CI contains 0 again. Equal-mass bins also have a limit here: with prevalence 3-6%, nine of the ten bins hold scores near 0 and only the top decile covers the region where the over-confidence shows, so the test still has low power. Conclusion unchanged and now worded correctly: calibration gain is not demonstrated; no calibrated-probability claim is made.
6. **Fold 5.** Section 1's "Fold 5 is never loaded" is literally false: `turns.csv` (all folds) is read and `v0_vc()` loads every `dihana_zero_shard*.pkl` (which includes fold-5 units) into memory before the pool ids are indexed. The correct statement is: **fold 5 rows are never indexed and never used** (the pool is built from folds 1-4, `assert_no_historical` guards it, no code path uses those rows; `historical_fold5.py` only re-tabulates the old CSV). Also: the freeze order PROTOCOL < run < TABLES < EXPERIMENTOS is evidenced by filesystem mtimes only (`results/` is git-ignored and the code commits postdate the run); `config.json` records the code sha256 at the end of the run, there is no code snapshot at freeze time.
7. Minor: `rep_literal` drops 67 B / C / D-only turns (disclosed); Vc was trained on the full repetition mark and is scored here on `rep_literal`, so its precision is structurally handicapped (the label "Vc-favourable" refers to tuning, not to target mismatch); inner-OOF thresholds come from 2/3-data models and are applied to a full-train model (disclosed only for Platt); revision C = 0.03 (grid edge) in all 5 outer folds.
