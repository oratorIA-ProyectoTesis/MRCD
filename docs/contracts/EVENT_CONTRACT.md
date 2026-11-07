# Event and AnalysisResult contract (CURRENT implementation, mrcd 0.3.0)

Status: descriptive. It documents what `mrcd/engine.py` emits today, not an ideal design. Inspected tree: see `README.md` in this folder (commit hash). Machine-checkable companions: `schemas/engine_result.current.schema.json`, `schemas/engine_windows.current.schema.json`, validated by `validate_examples.py` against the real fixtures in `examples/`.

> **There is no `DisfluencyEvent` in the code.** The thesis plan used that name; the implementation has `Event` (batch), `Window`, `AnalysisResult`, and, on the live path, plain `dict` rows with Spanish keys (see `LIVE_CONTRACT.md`). `DisfluencyEvent` exists only as a name for the _proposed_ integration event (see `APP_INTEGRATION_CONTRACT.md`).

## 1. Entry points and input

| Item          | Real behaviour                                                                                                                              | Source                   |
| ------------- | ------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------ |
| Load          | `DisfluencyEngine.load(checkpoint, whisper_size="small", device=None, head=None, head_checkpoint=None, head_sha256=None, asr_profile="p0")` | `mrcd/engine.py:182`     |
| Analyse array | `analyze(audio, *, sample_rate=16000, asr_chunk_s=8.0, keep_windows=False, terminal_window=False, span=None)`                               | `engine.py:265`          |
| Analyse file  | `analyze_file(path, **kw)`: WAV, PCM 16-bit, mono only; other formats are rejected                                                          | `engine.py:477`          |
| Audio         | 1-D `float32` in [-1, 1], finite, exactly 16 000 Hz (no silent resampling), at least `MIN_AUDIO_S = CONTEXT_S + WINDOW_S = 13 s`            | `engine.py:159, 301-322` |
| Visual modes  | `analyze` raises `ValueError` for `trimodal`/`kinesic_only` checkpoints: this engine never receives video                                   | `engine.py:301`          |
| Thread safety | not thread-safe on a shared instance; one instance per process/worker                                                                       | module docstring         |

Errors are Python exceptions (`ValueError`) with human-readable (Spanish/English mixed) messages; there are no error codes. Example: `examples/error_short_audio.json`.

## 2. `AnalysisResult` (`engine.py:98`)

`to_dict()` (`engine.py:123`) is the only serialisation. Fields:

| Field        | Type            | Unit / meaning                                                                                                                                                                                                                             | Always present                    |
| ------------ | --------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------- |
| `mode`       | str             | checkpoint mode: `audio_only` or `audio_text`                                                                                                                                                                                              | yes                               |
| `duration`   | float           | seconds of input audio                                                                                                                                                                                                                     | yes                               |
| `warning`    | str or null     | checkpoint provenance warning. For the shipped checkpoint it is the literal text `PRUEBA DE HUMO — etiquetas heurísticas, NO válida como evidencia`; a non-default ASR profile appends the experimental note and, if any, the anomaly note | key yes, value may be null        |
| `transcript` | str             | words joined by spaces from the ASR; raw, not post-processed                                                                                                                                                                               | yes                               |
| `metadata`   | object          | window plan, effective environment, ASR profile block (see 6)                                                                                                                                                                              | yes                               |
| `events`     | list of `Event` | see 3                                                                                                                                                                                                                                      | yes (may be empty)                |
| `head`       | object          | experimental Vc report (see `HEAD_CONTRACT.md`)                                                                                                                                                                                            | only when loaded with `head="vc"` |

Not serialised by `to_dict()`: `windows` (kept only with `keep_windows=True`, available as `result.windows`). `examples/batch_p0_windows.json` shows them via `dataclasses.asdict`.

## 3. `Event` (`engine.py:55`, emitted keys at `engine.py:130-145`)

| Key                               | Type                   | Unit               | Meaning, and what it is NOT                                                                                                                                                                                                                                                                                                                                                         |
| --------------------------------- | ---------------------- | ------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `label`                           | str                    | -                  | one of 7 classes: `filler_word, prolongation, repetition, block, revision, rhetorical_pause, neutral_pause`. `fluent` (background) never becomes an event                                                                                                                                                                                                                           |
| `start`                           | float                  | s from audio start | start of the first window of a contiguous same-class run                                                                                                                                                                                                                                                                                                                            |
| `end`                             | float                  | s                  | `last_window.start + 0.5` (stride), or the start of the terminal window when one follows (`engine.py:645-648`)                                                                                                                                                                                                                                                                      |
| `confidence`                      | float in [0,1]         | -                  | **maximum over the run's windows of the softmax output of the predicted class.** A raw, uncalibrated model score. It is not a probability that the phenomenon is present, not a probability of the class given the data, and not comparable across classes or checkpoints (the model was trained with class-weighted cross entropy, weights 0.18 to 53.8, on weak heuristic labels) |
| `n_windows`                       | int >= 1               | windows            | how many 3 s windows (0.5 s stride) were fused; 6 consecutive windows cover one instant                                                                                                                                                                                                                                                                                             |
| `text`                            | str                    | -                  | text of the **first** window in the run that has any words (`engine.py:656`); it is the words whose midpoint falls in that 3 s window, not the words of the event                                                                                                                                                                                                                   |
| `modalities`                      | list[str]              | -                  | branches the checkpoint mode uses (`audio_text` -> `["acoustic","linguistic"]`), a property of the mode, **not** evidence that each branch contributed to this decision                                                                                                                                                                                                             |
| `validation`                      | str (optional)         | -                  | only with Vc: says which classes are not validated                                                                                                                                                                                                                                                                                                                                  |
| `asr_profile`, `asr_experimental` | str, `true` (optional) | -                  | only when a non-default ASR profile produced the transcript                                                                                                                                                                                                                                                                                                                         |

### Interval semantics (important)

`[start, end)` is the **decision stride of the windows**, not a human-annotated boundary and not the extent of the evidence. The model saw `3 s` of audio (and up to 10 s of word context) to decide each 0.5 s stride. Real example (`examples/batch_p0.json`): `filler_word` 0.0-2.0 s, 4 windows, `text` = "[text withheld]" (the example text is redacted in the public repository; originally 10 words for a 2 s interval). Never present the interval as "the filler lasted 2 s".

### What does NOT exist in the event (explicit)

| Concept                                              | Status                                                                                                                                             |
| ---------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| calibrated probability                               | does not exist. No calibration of any kind was fitted for V0                                                                                       |
| uncertainty / confidence interval / entropy / margin | does not exist; the full softmax vector exists only in `Window.probabilities` (opt-in)                                                             |
| `uncertain` / `functional` / `disfluent` state       | does not exist in the engine output. Only `mrcd/decision.py` has `UNCERTAIN/PRESENT/ABSENT`, used by research scripts, never by `DisfluencyEngine` |
| per-event provenance (model hash, ASR model)         | not per event: it lives once in `metadata` (and `head.provenance`)                                                                                 |
| event id                                             | none                                                                                                                                               |
| speaker / identity                                   | none; one speaker is assumed                                                                                                                       |
| visual or kinesic evidence                           | none; `modalities` never contains `kinesic` for the available checkpoint                                                                           |
| per-event warning                                    | none; `warning` is result-level. `validation` is a per-event string only with Vc                                                                   |

## 4. `Window` (`engine.py:82`, opt-in)

`start, end` (s), `label` (8-way argmax, including `fluent`), `confidence` (= `probabilities[label]`), `probabilities` (8 softmax values, sum 1), `text`, `terminal` (the extra window ending exactly at the end of the span), `head_scores` (Vc sigmoid scores per window, only with `keep_windows=True` and a head). Probabilities here are the softmax output: _scores_, not calibrated probabilities (they sum to 1 by construction of softmax, which says nothing about calibration).

## 5. Window grid and merge (what the numbers mean)

- Grid: `start = 0, 0.5, 1.0, ...` while `start + 3 <= end_of_span` (`engine.py:513, 521`). With `terminal_window=True` one more window starting at `end - 3` is added when the grid does not end there (`engine.py:523`). Default is `False` (V0 exact).
- Merge (`engine.py:607-661`): contiguous windows with the same label and start gap <= 0.75 s fuse; `fluent` windows are skipped. Events never overlap.
- Consequences: audio that ends within less than 0.5 s of the last grid window is not covered unless `terminal_window=True`; the pause/boundary classes are very sensitive to padding and to trailing silence (`results/ejecucion_material4_20261030/pausas/PAUSAS.md`).

## 6. `metadata`

- `window_plan`: `window_s` 3.0, `stride_s` 0.5, `terminal_window`, `span`, `asr_chunk_s`.
- `head`: name of the loaded head or null.
- `environment`: OMP threads, versions (python, numpy, faster-whisper, ctranslate2), CPU count, `environment_is_pinned`, `effective` (may be `null` = not verified), ASR model size/device/compute type. `device` there reflects the configuration that was requested/resolved at load time, not an observation of the device the engine actually ran on (for example, a silent CPU fallback is not detected by this field). Contains a `warning` when OMP was not set before numpy was imported. Absolute model paths are present in `asr_profile.model.model_dir` (the fixtures replace the home directory by `<HOME>`).
- `asr_profile`: see `ASR_PROFILES_CONTRACT.md`. An ASR object that exposes no profile is reported as `name: "unknown"`, `experimental: null`, never as p0.

## 7. Validations performed by the engine (and where)

| Validation                                                                                  | Failure                            |
| ------------------------------------------------------------------------------------------- | ---------------------------------- |
| mode must be audio-only/audio_text                                                          | `ValueError` before any decoding   |
| 1-D array; finite samples; sample rate == 16 000; duration >= 13 s                          | `ValueError` before ASR            |
| WAV must be PCM16 mono (`analyze_file`)                                                     | `ValueError`                       |
| `span=(a,b)`: `0 <= a`, `b <= duration`, `b - a >= 3`                                       | `ValueError`                       |
| checkpoint `schema_version` == `FEATURE_SCHEMA_VERSION` (3)                                 | `ValueError` on load               |
| Vc file sha256 matches the pinned hash before deserialisation, base checkpoint hash matches | `ValueError`                       |
| unknown ASR profile name                                                                    | `ValueError` before any model load |

Tests: `tests/test_public_api.py` (grid, merge, serialisation, errors), `tests/test_engine_evaluation.py`, `tests/test_engine_terminal.py`.

## 8. Evolution rules (proposal, not yet enforced)

1. Additive changes only (new optional keys); consumers must ignore unknown keys. The schemas in `schemas/` are intentionally closed so that a change is noticed here first.
2. A key that today means "raw score" (`confidence`) must not be re-interpreted as a probability; if calibration is ever fitted, add a new key (e.g. `calibrated_probability` with method and dataset provenance) and keep `confidence` unchanged.
3. Removing or renaming keys (`confidence`, `text`, `modalities`) is a breaking change and needs a version bump and a changelog entry.

## 9. Real examples

| File                              | Produced by                                              | Shows                                                            |
| --------------------------------- | -------------------------------------------------------- | ---------------------------------------------------------------- |
| `examples/batch_p0.json`          | default engine (V0, p0)                                  | 8 events, transcript, metadata, no `head`                        |
| `examples/batch_p0_vc.json`       | `head="vc"`, `terminal_window=True`, `keep_windows=True` | same events + experimental `head` block                          |
| `examples/batch_p0_windows.json`  | same run                                                 | 35 windows with probabilities and `head_scores`                  |
| `examples/batch_p1.json`          | experimental profile p1                                  | event marks `asr_profile`/`asr_experimental`, extended `warning` |
| `examples/error_short_audio.json` | 5 s input                                                | the only error format: an exception message                      |

Command, excerpt hash, engine and checkpoint hashes: `examples/provenance.json` and the docstring of `examples/generate_examples.py`.

## 10. Revision 1.2 of the service contract and `return_words` (S3, 2026-11-01)

`AnalysisResult.words` (list of word dicts with timestamps) is `None` unless `analyze(..., return_words=True)` is passed; the default output is bit-identical (golden test `tests/test_s3_engine_words.py`). The service uses it only for the opt-in text route. See `APP_INTEGRATION_CONTRACT.md` B.3b for the additive `mrcd.analysis/v1` fields.
