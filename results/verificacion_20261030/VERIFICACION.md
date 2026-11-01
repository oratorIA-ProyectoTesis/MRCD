# MRCD verification report, 2026-10-30

Scope: verification only. Nothing was modified, committed or overwritten. New files only, all in `results/verificacion_20261030/`.
Repo: `C:\zArnes-tesis\mrcd` (branch main, HEAD 9afc399, dirty tree; `mrcd/engine.py`, `mrcd/inference.py`, `mrcd/models/` are untracked).

## 1. Full test suite

Command (repo root, `.venv`): `python -m pytest -q -x --no-header -p no:cacheprovider`

Result: **207 passed, 0 failed, 0 skipped, 0 errors, 6 warnings in 46.19 s.** Matches the prior report (207).
Warnings are deprecations only (starlette/httpx TestClient, anyio BlockingPortal alias, FastAPI `on_event` at `scripts/live_demo.py:255`).
Note: README says "27 pruebas" (stale; real count is 207).

## 2. Claims vs. real state

(a) Live demo double stop/close fix: CONFIRMED in code and test.

- `scripts/live_demo.py`: `LiveEngine.stop()` is state-locked, bumps `_generation`, returns `was_running`; `start()` returns False if already running; `_run` exits when the generation changes. `/control/start` and `/control/stop` are idempotent via the `running` flag; the app-exit hook calls `engine.stop()` only if running.
- `tests/test_live_demo.py::test_live_controls_start_and_stop_the_engine` asserts `starts == 1 and stops == 1` after start, stop and app exit (stop not called twice). Passes. Both files are uncommitted modifications (+263/-39 lines vs HEAD).

(b) `results/reproduccion_piloto_20261030`: CONFIRMED (own computation; the dir has no comparison file itself).

- Contents: 32 `ameresco_NNN_asr.json`, `audio_predictions.jsonl` (32 rows), `audio_metrics.json`, `feature_equivalence.json` (34 files). No script is stored there; it came from the pilot's `run_audio_pilot.py` (with `MRCD_RESULTS_DIR`), which imports `engine_snapshot`, NOT the current `mrcd/`.
- Pilot vs reproduction (probabilities and window labels): 32/32 identical decisions, max abs probability diff 0.0.
- Metrics repeated: TP/FP/FN/TN 5/3/11/13, F1 0.4167; RTF 0.481 (pilot report says ~0.511; timing noise).
- Re-run with CURRENT `mrcd/` (dataset, extractors, inference) + cached pilot ASR words + current checkpoint: `python results/verificacion_20261030/verify_current_vs_pilot.py` -> 32/32 window-label sequences identical; max abs diff 0.0 vs pilot and vs reproduction (`current_vs_pilot.json`). Whisper was NOT re-run (cached words), so ASR determinism is untested here.

(c) Checkpoint hash: IDENTICAL.
`models/ckpt_audio_text.pt` and `piloto_corpus_20261029/engine_snapshot/ckpt_audio_text.pt`: sha256 `87f95108a765bfacbbc092ff1b7580cfc804fea84132a9813163d928d033b669` (matches TASKS.md T28).

(d) State of engine/inference/models:

- `mrcd/engine.py`: untracked; DIFFERS from the snapshot (section 3).
- `mrcd/inference.py`: untracked; byte-identical to the snapshot.
- `mrcd/models/`: untracked; `__init__.py` (empty) and `fusion.py` identical to the snapshot; also a stray `.ruff_cache/`.
- HEAD (9afc399) still holds `core/`, not `mrcd/` (TASKS.md T28 is right: no commit reproduces the tested code).
- `models/BASELINE.md` says the exact v1 checkpoint is unavailable, yet the current checkpoint hash equals the pilot's; both are `label_source=auto` smoke-test checkpoints. BASELINE.md may need reconciling.

(e) Discrepancies between reports and reality:

1. The pilot's `verify_results.py` FAILS today: `AssertionError: ...\mrcd\dataset.py` (sha256 differs from `protocol.json`). `mrcd/engine.py` would fail too. The other 19 source files and the checkpoint match. This is the "historical verification failure".
2. README says 27 tests; actual 207 across 12 test files. INFORME_AVANCE's "62 historical" tests refer to another version.
3. RTF 0.511 (report) vs 0.481 (`reproduccion_piloto_20261030/audio_metrics.json`): timing noise, not correctness.
4. INFORME_AVANCE says the 0.4.0 `Engine` tree was unavailable to that session; it exists here (untracked).
5. The pilot ran features on `engine_snapshot`; its "unchanged MRCD functions" claim holds for the snapshot only. `feature_equivalence.json` checks only the first window of the first clip.

## 3. Differences snapshot vs `mrcd/` (`diff -u`; extractors, constants, sync, live, annotation are byte-identical)

`mrcd/dataset.py` (none alter tensors from `acoustic_windows`, `linguistic_windows`, `center_f0`, `prosody_windows`):

- `read_human_events(..., *, strict=False)`: strict mode raises on missing tier, REF_ANNOTATION tiers, unknown/empty label, invalid interval. Default unchanged.
- `build_recording_windows` with `label_source="human"` now requires `rec["human_annotation"] = {coverage:"complete", provenance:<non-empty>}`, an existing EAF, and events within `duration_s`; otherwise ValueError. Before, the human path silently produced negatives. Affects LABELS and which recordings build, not input tensors. `auto`/`gold` paths unchanged.
- Cosmetic: trailing whitespace in an import.

`mrcd/engine.py`:

- `analyze` contract: rejects modes needing video, non-1D audio (before: `reshape(-1)` flattened 2D/stereo into one stream, silently altering audio), and non-finite samples. `analyze_wav` rejects non-PCM16 and reads `<i2`.
- `_evaluate` rewritten: acoustic and linguistic features extracted once for the whole file and windows built jointly (`acoustic_windows`, `linguistic_windows`, `prosody_windows`) instead of `build_window` per window; inference with `bs=1`. This CAN change input tensors vs the snapshot: the per-window path computes F0 centering and prosody from the sliced audio, the new path uses whole-file `speaker_f0_center` and whole-file Silero VAD/frames. Not measured against the real checkpoint here for `Engine.analyze`. Also raises if `ling_dim` exceeds produced dims; truncates `ling[:,:,:ling_dim]`.
- `Window.text` is now always filled (was empty when `keep_windows` False): output change, not tensors.

Conclusion: tensors behind the pilot predictions did NOT change (numerically verified, max diff 0). The hash failure comes from the `dataset.py` human-label/strict changes and the `engine.py` rewrite, not from extractor changes. To close the gap, re-pin hashes in `protocol.json` (new file) or compare `Engine.analyze` against per-window `build_window` on real clips.

## 4. Files created

- `results/verificacion_20261030/verify_current_vs_pilot.py`
- `results/verificacion_20261030/current_vs_pilot.json`
- `results/verificacion_20261030/VERIFICACION.md`

## Exact commands

- `python -m pytest -q -x --no-header -p no:cacheprovider`
- `sha256sum models/ckpt_audio_text.pt <pilot>/engine_snapshot/ckpt_audio_text.pt`
- `diff -rq <pilot>/engine_snapshot/mrcd mrcd -x __pycache__`; `diff -u` per file
- `python results/verificacion_20261030/verify_current_vs_pilot.py`
- `python <pilot>/verify_results.py` (fails at the dataset.py hash)
