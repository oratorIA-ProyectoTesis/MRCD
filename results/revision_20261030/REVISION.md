# Independent adversarial review of results/mejoras_20261030 (2026-10-30)

Reviewer did not take part in the work. Read-only except this file and `pytest_full_rerun.txt`. Scratch scripts (leak check, independent reproduction, feature recomputation) live in the session scratchpad, not in the repo.

## Verdict

No leakage or temporal-order violation found. The headline numbers reproduce exactly. The weaknesses are in the strength of the claims, not in the data handling.

## 1. Leakage (checked against real manifests, not declared ones)

- REFUTED: fold 4/5 in training. `dev_units_frozen.json`: dev = 3816 units, folds {1:1350, 2:1294, 3:1172}, 135 speakers; tune = 1240 units, all fold 4, 45 speakers. Overlap dev/tune/fold 5 = 0 units, 0 speakers. `train_head.py` reads only `dev_units` for training and `tune_units` for selection; checkpoint meta says train_folds [1,2,3], selection_fold 4.
- REFUTED: repeated DIHANA speakers across folds. 225 speakers (B1xx), 900 dialogues; every speaker and dialogue sits in exactly one fold (pairwise intersections all 0).
- REFUTED: normalisation with fold 5/EPIC/AMERESCO. The feature norm comes from the base checkpoint; head mu/sd are fitted on dev windows only. `select_dev.py` filters splits {dev, tune}; `final_eval.py` reads only frozen JSON (it fits nothing).
- PLAUSIBLE (low): design exposure. The previous baseline study (same repo) already looked at EPIC/AMERESCO V0 results and error analysis, and its BASELINES.md calls EPIC "development, not generalisation". MEJORAS describes EPIC/AMERESCO as never used and "zero-shot". No numbers were used to select anything, but the choice of experiments was informed by those results.
- PLAUSIBLE (low): fold-5 "debug rows" were seen in scratch on the INVALID cache (acknowledged in MEJORAS section 0). Not auditable, and no evidence they influenced choices.

## 2. Temporal order

CONFIRMED OK. PREREGISTRATION.md 12:24; dev/tune ASR+features finished 14:24-14:33; V0 15:30; frozen_selection/dev_units 15:34:01-06; frozen_head_* 15:34:42-43 (checkpoints 15:34); final/* 15:35:25-28. All freezes precede the test evaluation by under 2 minutes, but in the correct order. Each `config.json` records improve_code_sha256; all match the current files (no drift). Minor: the configs of v0/selection/head/final record `OMP_NUM_THREADS=None` (those stages do not call ASR, so benign), while MEJORAS says every config records OMP.

## 3. Independent reproduction (fresh process, OMP=4 before numpy, own head and bootstrap code, no `mrcd.adaptation`/`evalkit`)

Fold 5 (1223 turns, 45 speakers), identical to MEJORAS/FINAL_TABLES:

- filler (145 pos): V0 TP/FP/FN/TN 64/44/81/1034, F1 0.506; Vc 90/77/55/1001, F1 0.577; dF1 +0.071, CI [+0.011,+0.129].
- prolongation (211 pos): V0 95/46/116/966, F1 0.540; Vc 157/130/54/882, F1 0.631; dF1 +0.091, CI [+0.009,+0.179].
- The paired bootstrap resamples SPEAKERS and uses the same index matrix for both versions (same units). Re-running with 3 other seeds x 10000 resamples gives lower bounds +0.011/+0.006-0.009: the prolongation bound is unstable around 0.01.
- Features recomputed from audio plus frozen ASR for 16 random fold-5 units in scratch: probs and pooled vectors differ by exactly 0. Cross-class counts (899 units without reference: prolongation 29 vs 76, filler 27 vs 24) also reproduce.
- Two-sided bootstrap p for the two "demonstrated" gains is about 0.02-0.03.

## 4. Claims

- CONFIRMED (medium): multiplicity. About 25 paired comparisons vs V0 (5 versions x 4 classes + EPIC) plus the same number vs the untrained comparator, all with one-sided-style "lower bound > 0" at 95%, no correction. The pre-registration names no primary endpoint. Holm/Bonferroni over 25 turns p of 0.02-0.03 into about 0.5-0.7: none of the three "demonstrated" gains survives. Lower bounds are +0.009, +0.011, +0.015.
- CONFIRMED (medium): "demonstrated improvement" is F1-only. Vc prolongation: precision 0.67 to 0.55, FP 46 to 130, McNemar vs V0 68:90 (p 0.094): more units wrong than V0. Vb prolongation has McNemar 62:160 (more errors) while F1 is flat. The F1 gain is recall bought with false positives.
- CONFIRMED (medium): comparators. Prolongation is compared with `always_present` (F1 0.29), weak; +0.336 is not evidence of usefulness. Filler Vc = ASR lexical rule (-0.001) so nothing beyond a one-line text rule; repetition: every version is worse than the ASR rule (disclosed).
- CONFIRMED (medium): optimism of tune and non-replication on test. Vb prolongation: dev OOF 0.58 vs V0 0.45, tune 0.61 vs 0.50, test 0.55 vs 0.54 (+0.007). Head thresholds at tune F1: filler 0.65, prolongation 0.65, repetition 0.48 vs test 0.58, 0.63, 0.32. V0 itself varies 0.45 (dev), 0.50 (tune), 0.54 (test) for prolongation: between-fold variation is about as large as the claimed gain. One test fold, 45 speakers.
- CONFIRMED (low): Ea "adopted" A_noise_tau0.5 passes on dev CI lower bound +0.005 (7 candidates, no correction) and tune +0.002; on test Va is indistinguishable from V0 (filler +0.006, others 0). "Adopted" should not be read as an improvement.
- CONFIRMED (low): prolongation threshold 0.02 is the grid floor (boundary solution), disclosed. Prolongation = DIHANA lengthening mark, not necessarily a disfluency.
- CONFIRMED (low): EPIC/AMERESCO gains are not significant (EPIC CI includes 0; AMERESCO McNemar 0.14-0.59, 4 conversations, Vc about equals always_present 0.55). MEJORAS states this correctly; ensure the abstract/thesis does not cite 0.27 to 0.56.
- REFUTED: ausencia como cero (n.d./NaN/"not available" used; revision reported as 0.09 [0.00-0.20] with support 36, not as a result of detection), mixing tasks/corpora (tables are per corpus; DIHANA turn vs AMERESCO segment vs EPIC sentence kept separate), and calling a probability calibrated (docstrings and meta say NOT calibrated).
- Doc inconsistency (trivial): MEJORAS says test_windowing has 7 tests and 25 new total; the file has 8 (6+8+6+5=25), so the per-file counts do not add up. "pipeline equals baseline (60/60, diff 0)" is not covered by any new test (script-based).

## 5. Tests

Literal result of the full run (this review): `232 passed, 6 warnings in 51.35s` (`results/revision_20261030/pytest_full_rerun.txt`), matches 232.
Quality of the 25 new tests: mostly real (hook equals head softmax, MIL recovers a signal in one window, group folds never split a speaker, save refuses overwrite and detects a tampered base hash, plan windows). Weak or tautological:

- `test_cv_threshold_oof_is_honest_on_noise` asserts only `in_sample >= oof - 0.05`; it does not check that OOF is near chance or that the fold-f threshold ignores fold f.
- `test_late_assignment_is_reported...`: the `effective=True` branch is true by assumption (the value is pre-set in the passed dict); it does not prove the numpy-order effect.
- No tests at all for `scripts/improve/evalkit.py` (paired bootstrap, McNemar, `paired_macro_delta`), where the headline statistics live. I verified them by independent reimplementation (they are correct), but nothing guards them.

## 6. Integrity of base and earlier results

- `models/ckpt_audio_text.pt` sha256 = `87f95108a765bfacbbc092ff1b7580cfc804fea84132a9813163d928d033b669` (matches), mtime 2026-09-24. New checkpoints: `..._zero.pt` 776c72f8...ab4a and `..._adopted.pt` 078876ea...1564 match MEJORAS. frozen_selection b1db7613..., dev_units 32931d81... recorded here.
- results/baselines_20261030: `metrics_audio.json` and `text_metrics.json` hashes equal those in its config.json; its 22 code files match their recorded sha256. Baseline scripts carry mtime 12:08-12:11 today (before the pre-registration, i.e. the earlier study's config rewrite), not modified by this study.
- ASR determinism audit (60 DIHANA, 12 EPIC, 12 AMERESCO bit-identical) present; I did not re-run ASR.

## Statements to soften

1. "Improvement demonstrated" (Vc filler, Vc/Vac prolongation): say "F1 gain with a CI lower bound near 0 (+0.009 to +0.015), uncorrected for about 25 comparisons, achieved by trading precision for recall; not robust".
2. "Demonstrated over always_present (+0.336)": weak comparator; do not present as evidence of detector quality.
3. "Adopted" for Ea and Eb(prolongation): the test shows no gain; say "passed the dev/tune gate, no effect on test".
4. "Zero-shot" EPIC/AMERESCO: the corpora were examined in the previous study; say "not used for fitting or selection here".
5. Any claim of generalisation from 1 test fold (45 speakers), telephone, peninsular, train-information domain.
6. Remove the per-file test-count mismatch (windowing 8, not 7).
