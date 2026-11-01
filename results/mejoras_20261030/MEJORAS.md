# MRCD improvements study (2026-10-30)

New files only (no commit, no existing code/checkpoint/result overwritten). Pre-registration: `PREREGISTRATION.md` (written before any candidate number on fold 5 / EPIC / AMERESCO). Frozen selections: `frozen_selection.json`, `frozen_head_{zero,adopted}.json`, `dev_units_frozen.json`. Every run folder has `config.json` (seed 13, versions, sha256 of code, checkpoint, Whisper `model.bin`, ASR cache manifests, OMP, threads).

## 0. Status against the pre-registration

| item                                                                               | status                                                                                                                                                                                                                                   |
| ---------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Step 1 frozen ASR caches (DIHANA 6279, EPIC 136, AMERESCO 98) at OMP=4 + manifests | DONE. AMERESCO reproduces the pilot: 32/32 identical text and bit-identical words/times/probs (`asr_cache/ameresco_vs_pilot.json`). Determinism audit in a fresh process: 84/84 units bit-identical (`asr_cache/determinism_audit.json`) |
| V0 on frozen caches (dev 3816 / tune 1240 / test 1223; EPIC; AMERESCO)             | DONE (`v0/V0_BASELINE.md`)                                                                                                                                                                                                               |
| Ea short-turn handling (selection on dev/tune)                                     | DONE; adopted `A_noise_tau0.5` (`selection_dev_tune.md`)                                                                                                                                                                                 |
| Eb threshold rule (dev CV by speaker)                                              | DONE; adopted only for prolongation                                                                                                                                                                                                      |
| Ec supervised head (train folds 1-3, select fold 4)                                | DONE, new checkpoints (hashes below)                                                                                                                                                                                                     |
| Final evaluation fold 5 / EPIC / AMERESCO                                          | DONE (`final/FINAL_TABLES.md`, `FINAL_metrics.json`)                                                                                                                                                                                     |
| Step 4 no-reference classes                                                        | DONE (`properties/properties_noref.json`)                                                                                                                                                                                                |
| Step 5 full suite                                                                  | DONE: 232 passed (207 + 25 new), `logs/pytest_full.txt`                                                                                                                                                                                  |
| Not done                                                                           | see section 6                                                                                                                                                                                                                            |

### The `*_INVALID_*` directories

- `asr_cache_INVALID_pin_after_numpy/` and `features_INVALID_from_invalid_asr_cache/` (+ `logs/invalid_first_attempt/`): first ASR run (~1.5k DIHANA turns, 98 AMERESCO, ~120 EPIC units, about 50 min). It set `OMP_NUM_THREADS=4` from Python AFTER `import numpy` (via `import mrcd`). `os.environ` showed 4 but the transcripts equal an OMP-unset run: only 2/32 pilot AMERESCO clips matched the pilot cache. Bisected with 4 scratch scripts: variable set before numpy reproduces the pilot, after numpy does not (mechanism not profiled).
- Replaced by `asr_cache/` + `features/`, generated with `scripts/improve/_pin_env.py` as the first import of every entry point and a transcript canary in `asr_cache.py` (aborts if the pilot clip is not reproduced). One canary failure happened once at process start (log overwritten, cause unknown); the 84/84 audit and 32/32 pilot match support the final caches. The invalid directories were never used for any reported number (only for debugging code paths on dev units in scratch; fold 5 debug rows were discarded).

## 1. Implemented (software)

- `mrcd/reproducibility.py`: `pin_asr_threads` (opt-out `MRCD_NO_OMP_PIN=1`, respects a pre-existing value, reports `late`/`effective`), `asr_environment` (versions, model.bin sha256, OMP). Engine integration NOT done (would edit `engine.py`).
- `mrcd/windowing.py`: pilot window plan, zero/noise-floor padding, abstention, pause statistics.
- `mrcd/decision.py`: window score aggregation (max/mean/top2), per-class threshold, speaker-grouped CV, 3 states (present/absent/uncertain), `ThresholdRule` JSON.
- `mrcd/adaptation.py`: pooled-feature hook, MIL head (max over windows), new-checkpoint writer (refuses overwrite, checks base hash).
- `scripts/improve/`: `asr_cache, asr_audit, verify_ameresco_cache, run_features, v0_baseline, select_dev, train_head, final_eval, properties_noref, evalkit, common, _pin_env`.
- Checkpoints: `checkpoints/ckpt_head_dihana_f123_zero.pt` sha256 `776c72f8e064e42c569e3046f5d237de03aec4a810cabfdef5054d234066ab4a` (MLP32, wd 1e-4, epoch 40); `..._adopted.pt` sha256 `078876ea924fd4e34a9af43f19df2350cc52c8b0d18eeb30869b91dc611f1564` (linear, wd 1e-3, epoch 50). Base `models/ckpt_audio_text.pt` sha256 `87f95108...b669` untouched.

## 2. Commands (repo root, `.venv\Scripts\python.exe`)

```
scripts/improve/asr_cache.py {ameresco|epic} 0 1 ; asr_cache.py dihana I N [rev|devfwd|offK] ; asr_cache.py CORPUS manifest
scripts/improve/verify_ameresco_cache.py ; scripts/improve/asr_audit.py 60 12 12
scripts/improve/run_features.py CORPUS {zero|noise} SHARD NSHARDS 1
scripts/improve/v0_baseline.py ; scripts/improve/select_dev.py
scripts/improve/train_head.py zero zero ; scripts/improve/train_head.py noise adopted
scripts/improve/final_eval.py ; scripts/improve/properties_noref.py 120
python -m pytest -q --no-header -p no:cacheprovider
```

## 3. Measured

Setting: DIHANA dev = all 3816 turns of folds 1-3 (previous study used a 33% sample), tune = fold 4, test = fold 5 (1223 turns, 45 speakers). V0 recomputed on caches: it differs from the older baseline because the ASR now runs at OMP=4 on unpadded audio (e.g. test filler V0 F1 0.51 vs 0.53 before, ASR-rule 0.58 vs 0.60).

Selection (dev/tune only): Ea `A_noise_tau0.5` passes (dev dMacroF1 +0.014 [+0.005,+0.025], tune +0.002). Eb: threshold rule adopted only for prolongation (max, thr 0.02 = grid floor, dev OOF F1 0.58 vs 0.45 V0 rule; the F1-optimal threshold is nearly "always present on the max score"). Ec head: tune mAP 0.59 (mlp) / 0.58 (linear).

Fold 5 F1 [95% CI by speaker], paired dF1 vs V0 [CI]:

| class (pos)        | V0   | Va                 | Vb                 | Vab               | Vc                                | Vac                               | best untrained (dev-chosen) |
| ------------------ | ---- | ------------------ | ------------------ | ----------------- | --------------------------------- | --------------------------------- | --------------------------- |
| filler (145)       | 0.51 | 0.51 (+0.006 n.d.) | = V0               | 0.51 (0.000 n.d.) | **0.58 (+0.071 [+0.011,+0.129])** | 0.56 (+0.049 n.d.)                | rule_asr 0.58               |
| prolongation (211) | 0.54 | = V0               | 0.55 (+0.007 n.d.) | 0.53 (n.d.)       | **0.63 (+0.091 [+0.009,+0.179])** | **0.63 (+0.089 [+0.015,+0.167])** | always_present 0.29         |
| repetition (60)    | 0.35 | 0.36 (n.d.)        | = V0               | 0.36 (n.d.)       | 0.32 (-0.032 n.d.)                | 0.34 (n.d.)                       | rule_asr 0.44               |
| revision (36)      | 0.09 | =                  | =                  | =                 | =                                 | =                                 | rule_asr 0.10               |

n.d. = not demonstrated (CI includes 0). Demonstrated improvements over V0: Vc filler and Vc/Vac prolongation only. Versus the best untrained comparator: Vc filler -0.001 [-0.072,+0.069] (not demonstrated, i.e. the head only reaches the one-line ASR rule); Vc prolongation +0.336 over always_present (demonstrated, but that comparator is weak); repetition: every version is WORSE than the ASR lexical rule (dF1 -0.08 to -0.12, CI < 0), and Va/Vb/Vab are worse than the ASR rule for filler (-0.07, CI < 0). Vc raises recall (filler 0.44 to 0.62; prolongation 0.45 to 0.74) at the cost of precision (filler 0.59 to 0.54, prolongation FP 46 to 130). Revision: nothing detects it.

Zero-shot: EPIC filler (136, 29 actors): V0 0.72, Vc 0.77 (+0.046 [-0.019,+0.133] n.d.), Vac 0.73, Vab 0.69; vs ASR rule 0.69 and always_present 0.67: no demonstrated gain. AMERESCO prolongation (98, 4 conversations, no CI): V0 0.27; Vb 0.54, Vc 0.56, Vac 0.60; always_present 0.55. McNemar vs V0 p = 0.37 / 0.14 / 0.59: unit-level evidence is not significant, and the gains are recall inflation toward the prevalence baseline (AMERESCO prevalence 38%, Vc recall 0.86, FP 46).

Cross-class (fold 5, units with no reference class, n=899): prolongation emitted on 29 (V0) vs 76 (Vc); filler 27 vs 24. FP/FN residuals (no labels created): V0 prolongation FPs are 80% in turns carrying DIHANA silent-pause mark `p` and 0% padded; Vc prolongation FNs are 50% short padded turns (median 3.0 s). Filler FNs: only 12% have a vocalic filler in the ASR, so the ASR caps the text rule near R 0.5; Vc filler FPs: 69% co-occur with another reference class.

No-reference classes (120 DIHANA tune turns for perturbations; all 6279 turns for rates; no F1):

- Determinism: repeat max prob diff 3.5e-10; torch threads 1/2/4 max diff 2.4e-7; 0 argmax changes.
- Robustness (acoustic only, words fixed): window argmax agreement 0.94 (SNR 30), 0.90 (SNR 20), 0.95 (x0.5), 0.93 (x2), 0.85 (x0.25). Unit emission flip rate for neutral_pause 6-14%, rhetorical_pause 4-13%, block 2-3%.
- Padding invariance FAILS for pauses: on 71 short turns, zero vs noise-floor padding flips rhetorical_pause emission in 46% and neutral_pause in 42% (rhetorical 56% to 10%, neutral 27% to 52%); extra zero padding (3 to 4.5 s) flips 28% / 46%. The pause classes depend on the padded tail. Block is stable (4%).
- 3-state: NaN score -> uncertain, monotone in score; emitted labels map to DISFLUENCY / RHETORICAL_PAUSE / NEUTRAL_PAUSE as in `CLASSIFICATION_OF`. No threshold exists for these classes (not calibrated, no reference).
- Emission on DIHANA (V0): rhetorical_pause in 45% of turns, neutral_pause 31%, block 15%. Spearman with turn duration: block +0.35, rhetorical -0.43, neutral +0.21; with measured max internal pause (VAD): block +0.23, rhetorical -0.13, neutral +0.22 (p<1e-20, n=6279); pause and duration correlate 0.58. Emission depends on duration and on pause length; this is not evidence of validity. Descriptive only: with mark `p` (n=1388) neutral 45% vs 26%, block 29% vs 11%, rhetorical 29% vs 49%.
- Missing reference: annotated pause onset/offset with duration and function (rhetorical vs neutral vs hesitation), and annotated blocks; DIHANA `p` has no duration/function and `<b>` is ambiguous.

## 4. Software-tested

New tests: `tests/test_reproducibility.py` (6), `test_windowing.py` (7), `test_decision.py` (6), `test_adaptation.py` (5) plus the pre-existing suite: 232 passed. Pooled-head equals `softmax(net.head(pooled))`; pipeline equals baseline given identical words (60/60 window labels, diff 0).

## 5. Limitations

- Weakly supervised base checkpoint; head trained on human DIHANA turn labels (telephone, peninsular, train-information domain); dev numbers are not independent of tune thresholds for Ec (fold 4 used for hyperparameters, epoch and thresholds, as declared). Probabilities uncalibrated.
- Prolongation = lengthening mark, not necessarily disfluent; Eb threshold hit the grid floor (0.02).
- Fold 5 was used once for the final tables; no re-selection after seeing it. AMERESCO: 4 conversations, no CI.
- ASR text is still the cap for filler/repetition; Whisper erases phenomena.
- Mechanism of the OMP/numpy order effect not profiled; one unexplained canary failure.

## 6. Pending

- Integrate pin/window handling/head into `Engine` (requires editing `mrcd/engine.py`, `__init__.py`).
- Terminal window in `Engine.analyze`; fix pause classes' dependence on padding (candidate: noise-floor pad plus abstention for pauses on short turns, not evaluated for pauses because no reference).
- Calibrate scores (not done); revision and repetition need a text-aware route (ASR rule beats every audio/text-pooled version); FS/REP-A absent.
- Reference for pause/block classes; reconcile `models/BASELINE.md`; README test count.

## Independent review addendum (2026-10-30)

Source: `results/revision_20261030/REVISION.md`. No leakage and no ordering violation found; V0/Vc fold-5 numbers reproduced exactly; 232 tests pass. The following wording supersedes any stronger claim above:

- "Improvement demonstrated" (filler Vc, prolongation Vc) means: F1 gain whose 95% CI lower bound is near 0 (+0.011, +0.009), uncorrected for ~25 comparisons (would not survive Holm/Bonferroni), obtained mostly via recall at the cost of precision (prolongation FP 46 -> 130).
- The +0.336 over `always_present` is not evidence of detector quality (weak comparator). Filler Vc only ties the one-line ASR rule. Repetition: every version is worse than the ASR rule.
- Dev/tune gains did not carry to test (e.g. Vb prolongation +0.14 dev, +0.007 test). V0 prolongation F1 varies 0.45-0.54 across folds, comparable to the claimed gain.
- "Adopted" (Ea, prolongation threshold rule) = passed the dev/tune gate; no effect on test.
- "Zero-shot" on EPIC/AMERESCO = not used for fitting or selection here (experiment choice was informed by earlier baseline inspection).
- One test fold (45 speakers, telephone, peninsular, train-information domain): no generalisation claim.
- `test_windowing.py` has 8 tests, not 7. `evalkit.py` has no tests (verified by independent reimplementation).
