# O-campaign: best obtainable result for filler_word, prolongation and repetition (2026-11-01)

**Status: measured (EXPLORATORY).** Protocol frozen before measuring (PROTOCOL.md sha256 `6e9e2b60...`); four dated annexes (V0 comparator clarification, progress logging, S3 weight rounding tolerance, loader module shadowing) declare every post-freeze change, none touched selection logic. DIHANA folds 1-4, nested CV by speaker (5 outer, 3 inner), speaker bootstrap, Holm. Self-review by the authors only; nothing is confirmed. Machine shared with the user's MRCD service/app and other processes during the runs (timings not clean).

## 1. What exists

- Frozen protocol: `PROTOCOL.md`, `PROTOCOL.sha256.json` (sha256 `6e9e2b6032f5...`, frozen 2026-11-01T20:42:46Z, git head `124226c`, space sha256 `15d5ab2ed4ee...`, code hashes).
- Package `mrcd/optimize/`: `space.py` (finite space: filler 98, prolongation 49, repetition 308 configurations; budget 60 per outer fold; stop on budget or 20 configs without a gain > 0.002), `blocks.py` (10 feature blocks), `models.py` (LR, HGB with sample weights), `search.py` (nested search, calibration/band choice, learning curve), `evaluate.py` (weighted metrics, speaker bootstrap, paired deltas + Holm, coverage/risk CI, promotion rule), `product_pipeline.py` (versioned, sha256-verified artifact, `default: false`), `data.py` (real loaders; fold 5 only in `load_fold5_historical`).
- Scripts `scripts/o_campaign/`: `freeze_protocol.py`, `run_campaign.py` (stages `classes`, `package`), `run_fold5_historical.py`.
- Tests (26, synthetic): no speaker crossing outer/inner; selection (config, thresholds, calibration, band) unchanged when outer-test labels are flipped or outer-test inputs destroyed; stop rule (budget and patience); Holm/bootstrap/weights; promotion rule (superiority, non-inferiority, unevaluable class blocks promotion); record path equals matrix path; artifact hash tamper detection; fold-5 loader referenced only by the final script; protocol/space hash and gate.

## 2. Commands (from `<repo>`, OMP_NUM_THREADS=4 is pinned by the scripts)

```
.venv\Scripts\python.exe scripts\o_campaign\run_campaign.py classes     # nested search, tables, learning curve, final fit (resumable per fold)
.venv\Scripts\python.exe scripts\o_campaign\run_campaign.py package     # promotion rule, models\o_campaign_20261101\, EPIC / AMERESCO zero-shot
.venv\Scripts\python.exe scripts\o_campaign\run_fold5_historical.py     # LAST, once: historical, contaminated
```

A cheap dry run before the full one: `classes --classes prolongation --budget 3`. Outputs: `results_<class>.json`, `predictions_<class>.csv`, `promotion.json`, `external_zero_shot.json`, `fold5_historical_contaminated.json`, `runs/` (per-fold pickles).

## 3. Honest limits already known (before measuring)

- ASR profile selection (p0 / p1 / p0_noloop / p1_noloop) is possible only for repetition: the other transcripts exist only on the 1025-turn frame. Filler and prolongation are searched on P0 only; running P1 on the 5056 turns (same ASR for all compared branches) would be an annex.
- Vc is not a feature and only a fold-4, Vc-favourable comparator (trained on folds 1-3, tuned on fold 4).
- The inner objective is optimistic (threshold chosen on the same scores); the outer figure is the honest one, and the gap is reported.
- `run_campaign.py` and `data.py` were never executed on real data (gate), only compiled; expect a short debugging pass on the first dry run, which will be declared in an annex.
- Product claim: nothing is promoted; `default: false` in any packaged manifest until the frozen rule (PROTOCOL section 7) is met.

## 4. Results (outer folds only; weighted F1 for repetition; CI = 95% speaker bootstrap)
Pooled outer-fold F1 of the optimised pipeline vs fixed comparators; delta = optimised minus comparator, paired, Holm per class. V0 = `any_window_argmax` = the engine's event rule (the product default).
| class (n/pos) | optimised P / R / F1 [CI] TP/FP/FN | V0 argmax F1 | delta vs V0 [CI] | vs rule | vs S3 best |
|---|---|---|---|---|---|
| filler_word (5056/774) | 0.739 / 0.659 / **0.697** [0.658, 0.732] 510/180/264 | 0.567 | +0.130 [0.100, 0.161] | +0.036 [0.008, 0.064] (lexical rule 0.661; Holm 0.014) | +0.040 [0.013, 0.066] vs F_STACK 0.657 (Holm 0.013) |
| prolongation (5056/967) | 0.542 / 0.720 / **0.619** [0.582, 0.652] 696/587/271 | 0.458 | +0.160 [0.122, 0.198] | no rule exists | +0.027 [0.008, 0.046] vs P_a stack 0.592 (Holm 0.007); +0.037 vs V0thr 0.582 |
| repetition literal (1019/214, weighted) | 0.714 / 0.650 / **0.680** [0.608, 0.741] 139/28/75 | 0.516 | +0.165 [0.089, 0.241] | -0.002 [-0.034, 0.032] vs rule on P1 (0.682); +0.103 vs rule P0 | -0.019 [-0.049, 0.012] vs A1 ctx_LR P1 (0.699); -0.030 [-0.060, -0.003] vs A2 (0.711, Holm 0.095) |
Vc (fold 4 only, Vc-favourable, descriptive): filler 0.632 vs Vc 0.651 (delta -0.019 [-0.061, 0.023]); prolongation 0.634 vs 0.648 (-0.014 [-0.048, 0.019]); repetition 0.627 vs 0.563 (+0.064 [-0.057, 0.221]).

**Winning pipelines and stability (5 outer folds).** All 5 folds chose a different configuration in filler and repetition (modal config in 1/5 folds): the choice is NOT stable beyond the family. Filler: P0, `rule+v0(+ctxfill...)`, LR or HGB, OR-with-rule in 3/5. Prolongation: `v0+sig+lenx+pause(+emb)`, LR 3/HGB 2, 4/5 share the block set. Repetition: profile P1 in 4/5 (p1_noloop 1/5), `reptxt+rule+loop(+sig)`, LR 4/5. Final pipelines (folds 1-4): filler p0 `rule+v0+ctxfill+sig+pause` LR C=3 or_rule; prolongation p0 `v0+sig+lenx+pause+emb` LR C=0.03; repetition p1 `reptxt+rule+loop+sig` HGB depth 2, 100 iter. Every search stopped by patience (22-57 of 60 configs). Selection optimism (mean inner F1 - outer F1): filler 0.715 vs 0.697; prolongation 0.626 vs 0.619; repetition 0.718 vs 0.680.

**Coverage / risk (abstention band chosen in the inner loop; uncertain = abstain, never a negative).** filler coverage 0.938 [0.929, 0.948], risk 0.059 [0.050, 0.068], F1 with uncertain as absent 0.689; prolongation 0.903 [0.895, 0.911], risk 0.122 [0.109, 0.135], F1 0.578; repetition 0.964 [0.954, 0.973], risk 0.024 [0.018, 0.030], F1 0.488 (risk not informative: always-absent is already low). Calibration chosen was "none" in 13/15 folds. Scores are not calibrated probabilities.

**Learning curve** (same winning configuration retrained on 25/50/75/100% of the train speakers; pooled outer F1):
| class | 25% | 50% | 75% | 100% | 100%-75% [CI] | 75%-25% [CI] |
|---|---|---|---|---|---|---|
| filler_word | 0.683 | 0.695 | 0.711 | 0.697 | -0.014 [-0.025, -0.003] | +0.028 [0.007, 0.048] |
| prolongation | 0.588 | 0.604 | 0.613 | 0.619 | +0.006 [-0.004, 0.016] | +0.025 [0.009, 0.041] |
| repetition | 0.608 | 0.661 | 0.682 | 0.680 | -0.001 [-0.033, 0.025] | +0.074 [0.024, 0.130] |
By the frozen reading rule (75%->100% gain with CI including 0 and |gain| < 0.01): prolongation and repetition are on a plateau (not data-limited with this model/features); filler 100%-75% is slightly negative (also no gain). Early gains (25%->75%) are small, so more annotation of the same kind would likely add little; the ceiling is set by features/labels/ASR (e.g. fillers that are unwritten marks, lengthening marks that coincide with pauses), not by training size.

**External zero-shot (final pipeline trained on folds 1-4; both corpora already inspected earlier; domain shift).** EPIC filler (136/68): F1 0.727 [0.630, 0.809] vs V0 0.721 (delta +0.006 [-0.050, 0.068]), lexical rule 0.686: no demonstrated gain. AMERESCO prolongation (98/37): F1 0.534 [0.154, 0.759] (R 0.946, P 0.372, 35/59/2) vs V0 0.269 (delta +0.265 [-0.210, 0.497], CI over few recordings, not informative): high recall, many false positives.

**DIHANA fold 5 (historical, contaminated; reported once, no decision uses it).** filler (1223/145): F1 0.639 [0.548, 0.717] vs V0 0.506 (delta +0.133 [0.073, 0.205]); prolongation (1223/211): 0.613 [0.543, 0.676] vs V0 0.540 (+0.074 [-0.007, 0.156]). Repetition not evaluated on fold 5.

**Promotion decision (frozen rule, Holm over 3 classes vs V0 argmax): PROMOTION CRITERIA MET** (filler, prolongation and repetition all superior, Holm p 0.0012; no class worse). Packaged in `models/o_campaign_20261101/` (MANIFEST.json sha256-verified, `default: false`, `status: promoted-candidate`); adoption by the service is the orchestrator's call. Reservations the rule does not weigh: the comparator is the weak engine default; against the best fixed comparator the gain is modest (filler +0.036, prolongation +0.027) and for repetition the pipeline is NOT better than the one-line rule on P1 text (-0.002) or S3's ctx_LR (-0.019/-0.030), so for repetition prefer the S3 route `rep_ctx_p1`. Repetition's pipeline needs the P1 transcript. Prolongation and the 'ctxfill/sig' blocks need word timestamps; DTW `acoustic` was never selected.

## 5. Conclusion and suite
Best result obtainable with these data (exploratory, DIHANA folds 1-4, outer-fold): filler_word F1 0.697, prolongation 0.619, repetition literal 0.680 (weighted). Ceiling by data vs model: plateau after ~75% of the training speakers in all three classes (CI of the last step includes 0 or is negative), so the ceiling is of features/labels/ASR, not of data volume or model family (LR and HGB tie). No class is confirmed; external gains are not demonstrated (EPIC) or uninformative (AMERESCO).
Suite: see `pytest_full.log` (`OMP_NUM_THREADS=4 .venv\Scripts\python.exe -m pytest -q -rs`); result line quoted below.
