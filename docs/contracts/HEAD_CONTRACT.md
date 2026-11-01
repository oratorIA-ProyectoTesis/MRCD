# Classification head contract: V0 network head and experimental Vc head (CURRENT, mrcd 0.3.0)

Sources: `mrcd/models/fusion.py`, `mrcd/inference.py`, `mrcd/adaptation.py`, `mrcd/constants.py`, `mrcd/dataset.py`, `mrcd/extractors/{acoustic,prosody,linguistic}.py`, checkpoint metadata read with `torch.load` for this document. "Head" means two different things in this code base; this contract keeps them apart.

|             | V0 head (`net.head`)                                                                | Vc head (`LoadedHead`)                                                                    |
| ----------- | ----------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------- |
| What        | last layer of `CrossModalFusion`: `Dropout(0.1)` + `Linear(128 -> 8)`, then softmax | small MLP trained on the **frozen** pooled 128-D vector of the base net                   |
| Status      | default, always on                                                                  | **experimental, opt-in** (`DisfluencyEngine.load(head="vc")`), turn-level, 3 classes only |
| Output      | 8 softmax scores per window                                                         | 3 sigmoid scores per window; unit decision = max over windows vs frozen threshold         |
| Calibration | **none**                                                                            | **none**                                                                                  |

## 1. Interface

```python
# mrcd/inference.py
load_ckpt(path, dev, expect_schema=FEATURE_SCHEMA_VERSION, allow_schema_mismatch=False)
    -> (net: CrossModalFusion, norm: {key: (mu, sd)}, labels: list[str], provenance: dict)
predict(net, norm, d: dict[str, ndarray], dev, bs=128) -> ndarray (N, 8)   # softmax, rows sum to 1
# mrcd/adaptation.py
predict_pooled(net, norm, d, dev, bs) -> (probs (N, 8), pooled (N, 128))   # forward hook on net.head input
load_vc(base_sha256, path=None) -> LoadedHead        # sha256 pinned in VC_SHA256
LoadedHead.window_scores(pooled) -> (W, C) sigmoid    # "scores, NOT calibrated probabilities"
LoadedHead.unit_decision(scores) -> {cls: {score, threshold, present, experimental: True, calibrated: False, reference_tune_f1}}
```

Batch dictionary `d` keys: `ac, pros, ling, lpos, lmask` (and `kin, kmask` for visual checkpoints that the engine does not run). `predict` z-normalises each numeric key with the checkpoint's own `(mu, sd)` (`inference.py:80-82`).

## 2. V0 checkpoint (`models/ckpt_audio_text.pt`)

| Fact                          | Value                                                                                                                                                                            |
| ----------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| sha256                        | `87f95108a765bfacbbc092ff1b7580cfc804fea84132a9813163d928d033b669` (matches TASKS T28)                                                                                           |
| mode / schema                 | `audio_text`, `FEATURE_SCHEMA_VERSION = 3` (load fails on mismatch or missing version)                                                                                           |
| parameters                    | 75 144                                                                                                                                                                           |
| labels (order = output index) | `fluent, filler_word, prolongation, repetition, block, revision, rhetorical_pause, neutral_pause`                                                                                |
| dims                          | `ac 5, pros 9, ling 12, d 64`; pooled vector 128 = 64 (attention) + 64 (prosody branch)                                                                                          |
| training config               | 15 epochs, lr 0.002, batch 64, seed 13, modality dropout 0.5 (inactive without video)                                                                                            |
| loss                          | **weighted cross-entropy**, class weights `[0.18, 2.83, 3.96, 4.66, 53.77, 8.36, 3.98, 0.80]` in label order                                                                     |
| provenance                    | `label_source: "auto"` (heuristic weak labels), `n_windows 18 927`, trained `2026-09-24T10:00:04`, `warning: "PRUEBA DE HUMO — etiquetas heurísticas, NO válida como evidencia"` |
| `kinesic_norm`                | `speaker` (irrelevant to audio_text)                                                                                                                                             |

Consequences for interpretation: (1) the supervision is weak heuristic labels (see `DISAMBIGUATION_SPEC.md` section 3), not human annotation; (2) class weights up to 53.8 shift the softmax away from class posteriors; (3) no held-out calibration set exists; therefore the output is a **score** that ranks and argmaxes, never a probability of presence. Probabilities that "sum to 1" are a property of softmax.

## 3. Features (names, dimensions, order, units)

All windows are 3 s (`WINDOW_S`), stride 0.5 s. Per window the net receives:

| Key            | Shape                | Content                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| -------------- | -------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `ac`           | (300, 5) at 100 fps  | `[log10(rms+1e-6), f0_st, voiced_prob, vad, df0]`; `f0_st` = semitones re 100 Hz, 0.0 = unvoiced, **centred per speaker** by subtracting the median voiced F0 (offline: whole file; live: causal median, 0 until 10 voiced frames); `df0` = frame difference of `f0_st`. Source: `acoustic.py:26-34`, `dataset.py:82-95`                                                                                                                        |
| `pros`         | (9,)                 | `PROSODY_FIELDS`: `sil_dur` (s, longest silence in the window), `sil_pos` (0-1), `f0_slope_pre` (st/s, 300 ms before), `f0_range_pre` (st), `f0_level_pre` (st, centred), `f0_reset` (st, F0 after minus before), `rms_slope_pre`, `final_lengthen` (proxy), `reset_valid` (1.0 when `f0_reset` could be measured). All zeros when the window has no silence. `sil_dur` enters as a number: **no threshold in seconds is applied at inference** |
| `ling`         | (up to 40 words, 12) | per word `WORD_FEAT_NAMES`: `is_filler` (lexicon match), `is_function_word`, `is_edit_term`, `repeats_prev`, `is_fragment`, `clause_end`, `log_duration`, `asr_prob`, `gap_before`, `gap_after`, plus window-dependent `zone` (-1 before / 0 inside / +1 after the 3 s window) and `dangling` (function word followed by a gap > 0.30 s). Words from a 10 s context (5 s before the window start, 5 s ahead)                                    |
| `lpos`         | (40,)                | position index (10 fps) inside the 10 s context                                                                                                                                                                                                                                                                                                                                                                                                 |
| `lmask`        | (40,) bool           | which word slots are real                                                                                                                                                                                                                                                                                                                                                                                                                       |
| `kin`, `kmask` | (30, 12), (30,)      | kinesic 12-D vector; **not used by the audio_text checkpoint and never produced by the engine**                                                                                                                                                                                                                                                                                                                                                 |

Versioning: `FEATURE_SCHEMA_VERSION` must be bumped when touching `KINESIC_FIELDS`, `PROSODY_FIELDS`, `WINDOW_S`, `STRIDE_S`, `VIDEO_FPS`, `ACOUSTIC_HOP_S` or the speaker normalisation (`constants.py`); older checkpoints then fail to load by design (`inference.py:load_ckpt`; `tests/test_leakage_guards.py::test_schema_version_is_stamped_on_new_datasets` covers the stamp on datasets; I found no dedicated test of the checkpoint mismatch error; history in `models/BASELINE.md`). Word-feature changes are not covered by that number; treat a change in `WORD_FEAT_NAMES` as breaking.

## 4. Vc head (`results/mejoras_20261030/checkpoints/ckpt_head_dihana_f123_zero.pt`)

| Fact                                                    | Value                                                                                                                                                                                                                                                                    |
| ------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| sha256 (pinned in code, checked before deserialisation) | `776c72f8e064e42c569e3046f5d237de03aec4a810cabfdef5054d234066ab4a` (`adaptation.py:VC_SHA256`)                                                                                                                                                                           |
| format                                                  | `mrcd-adapted-head-v1`, loaded with `weights_only=True` (no arbitrary pickle)                                                                                                                                                                                            |
| base checkpoint binding                                 | `base_ckpt_sha256 = 87f95108...` must equal the engine's checkpoint                                                                                                                                                                                                      |
| architecture                                            | standardise (`mu`, `sd` of 128) -> `Linear(128,32)` -> GELU -> Dropout -> `Linear(32,3)`; sigmoid per class                                                                                                                                                              |
| classes                                                 | `filler_word, prolongation, repetition` only                                                                                                                                                                                                                             |
| training                                                | human DIHANA turn-level presence labels (`f`, `a`, `r:repeticion`), folds 1-3, 3 816 units; epoch 40 / weight decay 1e-4 and thresholds chosen on fold 4 (selection fold); seed 13; MIL with max over windows                                                            |
| frozen thresholds                                       | `filler_word 0.70` (tune F1 0.651), `prolongation 0.66` (0.648), `repetition 0.90` (0.475)                                                                                                                                                                               |
| decision rule                                           | `max over windows of the sigmoid score >= frozen threshold` -> `present` (bool)                                                                                                                                                                                          |
| domain                                                  | DIHANA: telephone, peninsular Spanish, train-information dialogues; thresholds frozen on turns mostly shorter than 10 s; on long units presence saturates (the `unit.note` says so; real example in `examples/batch_p0_vc.json`: 20 s unit, all three classes `present`) |
| not covered                                             | `block`, `rhetorical_pause`, `neutral_pause`, `revision`: reported in `head.unvalidated_classes` with the reason                                                                                                                                                         |
| scope                                                   | does not change V0 events/windows/probabilities (checked by `tests/test_engine_terminal.py::test_vc_does_not_change_v0_outputs_and_marks_everything_experimental` and by `validate_examples.py`); not wired in the live path                                             |

`reference_tune_f1` is the F1 on the **tuning fold** at the chosen threshold: an optimistic development number, not a test result. There is no untouched test number for Vc in this contract.

## 5. Calibration status (explicit)

- V0: **no calibration was fitted.** No temperature scaling, Platt/isotonic, or binning. No reliability diagram, ECE or Brier score exists for the engine. The checkpoint was trained on weak labels with class-weighted cross entropy.
- Vc: sigmoid scores with thresholds tuned for F1 on a tuning fold; `scores_calibrated: false` and per-class `calibrated: false` are emitted on purpose.
- The only "calibration" experiments in the repository (equal-width and quantile bins, `scripts/rep_rev/calibration_quantile.py`) concern the research model `ctx_LR` on DIHANA text features, not this engine, and **did not demonstrate a gain** (all CIs contain 0).
- Therefore: never present `confidence`, `confianza`, softmax values or head scores as probabilities or percentages of certainty.

## 6. Failure modes

Schema mismatch -> `ValueError`; unpinned/modified head -> `ValueError` before unpickling; head trained on another base checkpoint -> `ValueError`; pooled path diverging from `predict()` (`allclose atol 1e-6`) -> `RuntimeError("refusing to score the head")` (`engine.py:582`); Vc file missing -> `FileNotFoundError` (the file lives in git-ignored `results/`).

## 7. Tests

`tests/test_adaptation.py` (pooled features reproduce the head; MIL max; save refuses overwrite; base-hash check), `tests/test_head_loading.py` (malicious pickle not executed; file read once; real Vc loads with `weights_only`), `tests/test_engine_terminal.py` (pinned hash, tampered file rejected, hash checked before unpickling, custom head needs hash and base, V0 untouched, experimental markers, uncovered reasons), `tests/test_decision.py`.
