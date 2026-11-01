# Live vs offline parity, terminal window, thread metadata, experimental Vc (2026-10-30)

Base: tag `t28-freeze-20261030` (232 tests). Raw outputs in this folder: `live_parity.json`, `rerun_comparisons.json`, `*.log`, `pytest_full.txt`, `golden_v0_pre_change.json` (V0 output of the pre-change code, used for bit-identity).
Environment: OMP_NUM_THREADS=4, torch CPU, cached ASR words (`results/mejoras_20261030/asr_cache`), checkpoint `models/ckpt_audio_text.pt` (sha256 `87f95108...b669`).

## 1. Implemented

| item                                                                                                                                                                                                                                                                                                          | where                                               |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------- |
| `SessionFeatureTracker`: Silero fed statefully from sample 0 (512-sample chunks) + `get_speech_timestamps_from_probs`, closed segments frozen after 1 s of quiet; RMS/pYIN recomputed on `[commit-1 s, now]`, frames committed only when >= 3 s old; absolute word times; `f0_center='causal'` or fixed float | `mrcd/live/session_features.py`                     |
| `AudioRingBuffer.read_since` (reports lost samples)                                                                                                                                                                                                                                                           | `mrcd/live/mic_stream.py`                           |
| Live demo uses the tracker, emits every 0.5 s-grid window once (catch-up), `--no-omp-pin`                                                                                                                                                                                                                     | `scripts/live_demo.py`                              |
| `terminal_window=False` (default) and `span=(a,b)` in `DisfluencyEngine.analyze`; `Window.terminal`                                                                                                                                                                                                           | `mrcd/engine.py`                                    |
| `AnalysisResult.metadata` (window plan, OMP status, effective yes/no, torch threads, faster-whisper/ctranslate2 versions, ASR device) + `RuntimeWarning` in `load()` when OMP was not set before numpy                                                                                                        | `engine.py`, `reproducibility.py`, `__init__.py`    |
| Entry-point pin before numpy with explicit opt-out                                                                                                                                                                                                                                                            | `scripts/_pin_omp.py` (live_demo, diagnose_session) |
| Opt-in `head='vc'` (experimental)                                                                                                                                                                                                                                                                             | `engine.py`, `adaptation.py`                        |

Old `build_latest_contextual_window` is unchanged (documented as not equivalent). The engine default ASR/env is never modified; `import mrcd` only records `OMP_NUM_THREADS` and whether numpy was already loaded.

## 2. Task 1: live vs offline

Command: `python scripts/live_parity/run_live_parity.py` (32 AMERESCO pilot clips + the 40 s clip; same audio, same words; 0.5 s mic ticks).

Diagnosis: the slice path differs because Silero is an LSTM that starts at zero state on the slice (and segmentation is local), the F0 centre is the slice median, and `word.start - offset` truncates `lpos` differently. pYIN Viterbi also depends on LATER frames: a 1 s or 2 s right margin gave 24 differing frames (2.72-2.95 s) on 1 of 33 clips (BAQ_001_03_16_a23); 3 s gave 0 on 33/33 (margin study, L=1: R=2 -> 24 rows differ, R=3 -> 0 of 40101).

| path (vs whole-file offline at the same start) | windows | ac max / windows>1e-5 | pros max / windows | ling, lmask | lpos windows | max prob diff | argmax changes |
| ---------------------------------------------- | ------- | --------------------- | ------------------ | ----------- | ------------ | ------------- | -------------- |
| old live (ring 13 s + slice)                   | 192     | 21.5 / 164            | 93.8 / 106         | 0           | 45           | 0.951         | 6 (3.1%)       |
| session, offline centre injected               | 622     | 0 / 0                 | 0 / 0              | 0           | 0            | 0.0           | 0              |
| session, causal centre                         | 622     | 23.1 / 577            | 5.5 / 283          | 0           | 0            | 0.889         | 20 (3.2%)      |

Committed frames: 40101 compared with whole-file extraction, 0 differ (all 5 columns, VAD included).

Declared tolerance:

- Frames (log-RMS, F0, voiced prob, VAD, dF0): exact, requires right margin >= 3 s. Justification: measured above; pYIN lookahead is empirical, not proven bounded.
- Tensors and probabilities with the offline centre: exact (tests use `assert_array_equal`; live demo rows equal offline probabilities exactly).
- Causal centre: not equivalent. Exact bound: every channel except centred F0 is exact; centred F0 differs by exactly `c_live - c_offline` (also `pros[4]`). Tested.
- Residual causal divergence by window start: 0-5 s: 13/330 argmax changes (median |ac diff| 0.80 st, max 23.1: centre is 0 until 10 voiced frames); 5-10 s: 6/235 (median 0.25); >=10 s: 1/57 (median 0.30, max 1.25). Pilot grid decision (prolongation anywhere in contained windows): offline 3/1/13/15, oracle identical (0 flips), causal 4/1/12/15 (1 flip). n=32: not evidence of accuracy.
- Other causal limits not removed: last ASR word has no next word; ASR text is re-transcribed (not covered here, words were given); latency grows to `max(5, 3+3)` = 6 s per window.

## 3. Task 2: terminal window

`terminal_window=False` output (windows, events, transcript) is bit-identical to the pre-change code (`tests/data/golden_v0_synth.json`, plus 4 real clips checked at generation: `golden_v0_pre_change.json`). `True` adds a window at `b-3` when the grid does not end there (`Window.terminal`); `span=(a,b)` gives the pilot plan (features still over the whole audio). Merge: the grid window before the terminal one is cut at the terminal start so events never overlap; the terminal window owns `[b-3, b-2.5)`.
Reproduction with `span=(target_start, target_end)` over the 32 pilot clips (cached ASR): `MRCD_SLOW=1 pytest tests/test_engine_terminal.py -k pilot_32` passes: with terminal TP/FP/FN/TN 5/3/11/13, without 3/1/13/15.

## 4. Task 3: threads

Fresh-process tests (`tests/test_omp_pin_subprocess.py`, 11): unset -> `set_before_numpy False`, warning; set in env -> effective, silent; assigned after `import mrcd` -> flagged ineffective ("after numpy"); value 8 recorded, not pinned; numpy imported first -> unverifiable (`None`); `_pin_omp` sets before numpy / keeps caller value / opt-out by `MRCD_NO_OMP_PIN=1` or `--no-omp-pin` / refuses to pretend when numpy is already loaded. `baselines/run_mrcd.py` was left untouched (frozen tooling).

## 5. Task 4: Vc (experimental, opt-in)

`DisfluencyEngine.load(ckpt, head='vc')`. sha256 pinned (`776c72f8...ab4a`) and verified BEFORE `torch.load`; base checkpoint hash must match; custom heads need `head_checkpoint` + `head_sha256`. Output `result.head`: `experimental: true`, provenance (checkpoint, hash, train folds [1,2,3], selection fold 4, task = turn-level presence, labels = human DIHANA f/a/r), `scores_calibrated: false`, per-class score/threshold/present for filler/prolongation/repetition, and `unvalidated_classes` (block, rhetorical_pause, neutral_pause: no compatible reference; revision: head not trained) that stay on V0 marked `validated: false`. V0 windows/events/transcript are identical with and without the head (tested). Thresholds were tuned on short DIHANA turns: on a long unit presence saturates (use `span`).

## 6. Task 5: affected comparisons (frozen caches, `rerun_comparisons.py`)

AMERESCO prolongation, TP/FP/FN/TN (P / R / F1). Frozen features and `DisfluencyEngine` end to end agree on all 98 units x {V0, Vc} x {with, without} (0 discordant units).

| set      | version | with terminal (frozen recipe) | without terminal (engine default) |
| -------- | ------- | ----------------------------- | --------------------------------- |
| pilot 32 | V0      | 5/3/11/13 (.625/.312/.417)    | 3/1/13/15 (.750/.188/.300)        |
| pilot 32 | Vc      | 14/13/2/3 (.519/.875/.651)    | 12/11/4/5 (.522/.750/.615)        |
| all 98   | V0      | 7/8/30/53 (.467/.189/.269)    | 5/5/32/56 (.500/.135/.213)        |
| all 98   | Vc      | 32/46/5/15 (.410/.865/.557)   | 29/39/8/22 (.426/.784/.552)       |

Every change in this table is caused only by the terminal window (the shared grid windows are identical; code and caches unchanged). The terminal default is False for contract/parity reasons (V0 exact), not because of any metric.

DIHANA fold 4 (tune), regression: Vc from the new code equals the previous code unit by unit (all 3 classes); V0 filler 84/54/84/1018, prolongation 95/39/154/952; Vc filler 123/87/45/985, prolongation 181/129/68/862.
DIHANA fold 5 (test), REGRESSION ONLY (nothing selected or tuned): reproduces the frozen tables exactly (TP/FP/FN/TN all equal for every class/version): V0 filler 64/44/81/1034 F1 0.506, prolongation 95/46/116/966 F1 0.540; Vc filler 90/77/55/1001 F1 0.577, prolongation 157/130/54/882 F1 0.631.

## 7. Software tests

`pytest` (OMP=4): **269 passed, 1 skipped** (232 previous + 37 new; the skipped one is the 32-clip pilot test, run separately with `MRCD_SLOW=1`: passed). Literal output: `pytest_full.txt`. New: `test_live_parity.py` (13), `test_live_engine_parity.py` (2), `test_engine_terminal.py` (12 + 1 slow), `test_omp_pin_subprocess.py` (11).

## 8. Limits / pending

- Equivalence is for frames/tensors given identical words; ASR streaming (13 s re-transcription) is not made equivalent. pYIN 3 s margin is empirical (33 clips, 40101 frames).
- Causal centre is a stated non-equivalence; early-session windows (< 10 voiced frames) use centre 0.
- Terminal window is not emitted by the live path. DIHANA was not re-run with/without terminal.
- Vc: experimental, DIHANA domain, recall gains at precision cost, CIs near 0 uncorrected for multiplicity (see REVISION.md); no class without reference is validated; scores uncalibrated.
- Not done: reconcile README test count; apply the pin to remaining scripts; `session.json` does not include live `gaps`.

## Corrections after independent review (REVISION2.md)

| # | change | before | after |
|---|---|---|---|
| F1 | Left margin added and 2-D study over all 98 AMERESCO clips (+ 8-clip, 4.5 min concatenation). Defaults L=3 s, R=3 s. `margin_study.py`, `margin_study_{A,B,C}.json` | L=1, R=3: BAQ_001_03_16_a24 had 36 frames (9.22-9.57 s) differing, max 1.44 st | see table below |
| F2/F3 | `load_adapted`: file read once, hashed, deserialized from the same bytes with `weights_only=True` (hash optional, never executes a pickle) | `weights_only=False`, hash-then-reopen | malicious pickle not executed with or without hash; real Vc loads (tests) |
| F4 | terminal tolerance = one acoustic hop (10 ms = 160 samples >= 1 sample) | 1e-6 s | 16.0001 s clip: no near-duplicate window; 16.05 s: terminal added (test) |
| F5 | `Event.validation` (only when a head is active), `Window.head_scores = {experimental, calibrated, head, scores}`, per-class `experimental/calibrated/reference_tune_f1` | marker only in `result.head` | tested |
| F6 | tests: growing live-loop centre bound, a24 regression, defaults, malicious pickle, single read | final centre, 1 s margin | added |
| F7 | warning also when OMP is set before numpy but != 4, or changed late | silent | tested in fresh processes |
| F8 | removed the fold-5 figure from `VC_UNCOVERED["revision"]` | hard-coded | text only |

2-D margin study, criterion = committed frames (all 5 columns) differing from whole-file extraction:

| (L, R) s | signals | with differing frames | differing frames | max abs F0 diff |
|---|---|---|---|---|
| 1, 3 | 98 | 2 | 39 | 12.3 st |
| 2, 3 | 98 | 1 | 7 | 1.14 st |
| 3, 3 (default) | 98 + concat | 0 | 0 | 0 |
| 4, 3 | 98 + concat | 0 | 0 | 0 |
| 3, 4 | 98 | 0 | 0 | 0 |
| 5, 3 | 98 | 0 | 0 | 0 |

Revised claim: with L=3 s, R=3 s, committed frames were identical to whole-file extraction on 99 signals (117152 frames; 98 clips + one concatenation; R=4 and L=4/5 also 0). That is an empirical result on these signals, not a proof: pYIN Viterbi dependence is two-sided and unbounded in principle, and larger R only hides failures if frames are not committed. Readiness delay is unchanged: `max(5, 3 + R)` = 6 s per window (L adds compute, not latency). Not tested: silence/noise beyond what the reviewer ran, other languages/devices.

Metric changes: none. Tensors/probabilities with the offline centre: still 0 differences on 622 windows; causal: still 20/622 argmax changes; pilot grid 3/1/13/15 (oracle) and 4/1/12/15 (causal). The terminal tolerance change does not alter the pilot reproduction (slow test: 5/3/11/13 with, 3/1/13/15 without). V0 default remains bit-identical (golden test). Fold 5 was not used for any choice; margins were chosen on AMERESCO signal-level exactness only.

Tests after review: `pytest_full_after_review.txt`: 279 passed, 1 skipped (OMP=4); pilot test with `MRCD_SLOW=1`: `pytest_slow_after_review.txt`.
