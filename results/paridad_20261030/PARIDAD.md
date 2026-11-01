# Parity: snapshot path vs current `mrcd/` Engine, real checkpoint (2026-10-30)

Only new files: `results/paridad_20261030/*`, `scripts/parity/*`. No existing code, checkpoint or result was touched. No commits.
Raw numbers: `PARIDAD.json`, `parity_snapshot_vs_current.json`, `asr_variability.json`, `attribution_slice*.json`, `paths/*.npz`.
Reproduce: `prep_jobs.py` -> `asr_run.py asr_run1|asr_run2` -> `run_paths.py --tag snapshot|current` (one process per code root; both are named `mrcd`) -> `compare.py snapshot current` -> `asr_compare.py` -> `attrib.py` -> `make_report.py`.

## Verdict

**No parity discrepancy.** Current `Engine.analyze` produces tensors and probabilities identical (max abs diff 0.0, 0 argmax changes, 0 decision changes) to the snapshot path, with the same checkpoint, on the same ASR words. The `engine.py` rewrite (single whole-file extraction) is numerically equivalent to the snapshot's per-window `build_window(audio_full, ...)` because the snapshot also extracted over the WHOLE audio each time (it just repeated it per window). The 2 files that differ (`dataset.py`, `engine.py`) do not change inputs on this path.

## Setup
- Inputs: 32 AMERESCO pilot clips (sha256 verified vs manifest) + 1 extra: 40 s excerpt (30-70 s) of `data/clips/yD9Igp27VNE.wav` (new derived WAV, hashes in JSON), targeted window a=5, b=35. Whole-file extraction is stressed more on this longer file.
- ASR words: pilot cache (extra: `asr_run1`). Same words in both paths.
- Paths (each run in snapshot root and in current root): `batch` (whole-clip once, pilot windows, bs=128), `bw` (`build_window` per window), `engine` (`DisfluencyEngine.analyze`, stub ASR returning cached words, tensors captured by spying `predict`), `slice` (see below).
- Env: python 3.12.10, torch 2.9.1 (CPU, 4 threads, OMP=4, seed 0), numpy 2.5.3, scipy 1.18.1, librosa 1.0.0, soundfile 0.14.0, faster-whisper 1.2.1, ctranslate2 4.8.2, onnxruntime 1.30.0, silero-vad 6.2.2. Whisper: small `536b0662` CPU int8, chunk 8 s / overlap 2 s, beam 5, T=0.
- Window 3 s, stride 0.5 s, context 10 s (5 s back). Pilot windows: a..b-3 step 0.5 + terminal b-3. Engine windows: grid 0..dur-3 step 0.5 (no terminal window).
- Checkpoint sha256 (both): `87f95108...b669`. Code sha256 per file in JSON; only `mrcd/dataset.py` and `mrcd/engine.py` differ.

## Tensor / probability differences (max abs; mean also 0)
| pair | windows | ac | pros | ling | lpos | lmask | probs | argmax chg | decision chg |
|---|---|---|---|---|---|---|---|---|---|
| snapshot/batch vs current/batch | 192 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| snapshot/bw vs current/bw | 192 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| snapshot/engine vs current/engine (820 grid windows) | 820 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| snapshot/bw (per-window) vs current/engine | 160 shared | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| snapshot/bw vs snapshot/batch | 192 | 0 | 0 | 0 | 0 | 0 | 3.6e-7 | 0 | 0 |
| current/batch vs saved pilot probs | 137 | - | - | - | - | - | 0 | 0 | - |

The only non-zero value is 3.6e-7 in probabilities: batch composition (bs=128 vs 1 window per call) in torch float32, never flips an argmax. Current Engine uses bs=1 and equals the per-window snapshot path exactly.
32 pilot terminal windows (b-3, off the 0.5 s grid) exist only in pilot paths; Engine cannot emit them, so they are not compared.

## Pilot metrics (target presence = any window argmax prolongation, target-contained windows)
| path | TP | FP | FN | TN | F1 |
|---|---|---|---|---|---|
| snapshot batch / per-window; current batch / per-window (pilot windows, incl. terminal) | 5 | 3 | 11 | 13 | 0.4167 |
| snapshot Engine; current Engine (grid windows only) | 3 | 1 | 13 | 15 | 0.3000 |
| any path restricted to the grid windows (control) | 3 | 1 | 13 | 15 | 0.3000 |

The 0.4167 vs 0.30 gap is NOT a model/extraction difference: it is the missing terminal window in `Engine` (control row proves it; the 160 shared windows have 0 decision changes). If `Engine.analyze` is used as the pilot's `predict-presence` it needs the terminal window to match 5/3/11/13. n=32, F1 values are uncalibrated smoke-test numbers (weak-label checkpoint); do not report as results.

## ASR variability (Whisper rerun, same params/machine, ~4 min/run)
Run1 vs run2 vs pilot cache: 33/33 and 32/32 identical text, max diff in times/probs = 0.0. Whisper is deterministic here (CPU int8, 4 threads, beam 5, T=0), so no ASR noise is mixed into the parity numbers. Caveat: other hardware/thread counts/ctranslate2 versions not tested.

## Controlled experiment: why per-slice features would differ (`attrib.py`, 160 grid windows)
Where the earlier concern could matter: any path that extracts on a SLICE (the live buffer path, `build_latest_contextual_window`) instead of the whole file. Slice = [t-5, t+8] s, same words.
- Frames identical in log_rms, f0_st, voiced_prob, df0 (max diff 0.0). Frames differ ONLY in the Silero VAD column: 24/160 windows with differing VAD frames, up to 190/300 frames.
- F0 centre (median over file vs slice): mean 0.29 st, max 2.10 st.
- Argmax flips vs whole-file: slice (both effects) 10/160; slice frames + whole centre 7; whole frames + slice centre 3. Max prob diff 0.65. In the all-window run, 3 of 32 pilot clips change decision (ameresco_014, _020, _027; F1 0.4348 vs 0.4167, within noise at n=32).
- So the dominant effect is VAD, then centring. Also `lpos` differs by 1 index in some windows with slices: float offset subtraction (`word.start - offset`) at an int truncation boundary; benign artifact.
- Implication: the Engine (offline, whole-file) is consistent with the pilot; the live/streaming path is NOT guaranteed to be, and model outputs are sensitive to VAD context. Worth a dedicated parity test for `live/` vs whole-file.

## Patch
None needed for parity (no discrepancy). Optional, not applied: an `Engine` option to also evaluate a terminal window `dur-3` (or caller-given starts) so it can reproduce pilot segment decisions. No `.patch` file written.

## Limits
Single ASR/hardware configuration; checkpoint is weak-label smoke test; extra clip is 1 file (no gold); the `slice` experiment is a sensitivity probe, not a claim about the live demo's accuracy.
