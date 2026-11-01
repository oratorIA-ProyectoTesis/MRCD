# Live row and `live_cfg_v1/v2/v3` contract (CURRENT implementation, mrcd 0.3.0)

Status: descriptive. Sources: `mrcd/live/session.py` (rows, configs), `mrcd/live/session_features.py`, `mrcd/live/streaming_asr.py`, `scripts/live_demo.py` (driver, threads, microphone, WebSocket). Schema: `schemas/live_row.current.schema.json`. Real rows: `examples/live_rows_v1.json`, `live_rows_v2.json`, `live_rows_v3.json` (replay of a 20 s excerpt, no microphone, ASR words taken from the offline decode, 0.5 s ticks; see `examples/provenance.json`).

## 1. What a live row is

The live path does **not** emit `Event`s. It emits one dict per 3 s window on the 0.5 s grid ("events from live rows (merge) not built", `INFORME_FINAL` section (g)). Keys are Spanish and historical; do not translate them in the wire format.

| Key                               | Type                         | Unit                  | Present                      | Meaning                                                                                                                                                                   |
| --------------------------------- | ---------------------------- | --------------------- | ---------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `timestamp`                       | float                        | s since session start | always                       | **start** of the 3 s window; the window covers `[timestamp, timestamp+3)`                                                                                                 |
| `texto_ventana`                   | str                          | -                     | always                       | words whose midpoint falls inside the window (ASR words known when the row was built)                                                                                     |
| `clase`                           | str                          | -                     | always                       | argmax over 8 labels (`fluent` + 7 classes); `fluent` rows are emitted too                                                                                                |
| `confianza`                       | float [0,1]                  | -                     | always                       | softmax maximum. **Raw score, not a calibrated probability** (the demo dashboard renders it as a percentage: that is a presentation choice the contract does not endorse) |
| `todas_las_probabilidades`        | object label -> float        | -                     | always                       | the 8 softmax outputs (sum 1; scores, not calibrated)                                                                                                                     |
| `status`                          | `"provisional"` or `"final"` | -                     | v2, v3                       | see 3                                                                                                                                                                     |
| `emitted_at_s`                    | float                        | s of audio fed so far | v2, v3                       | stream time at emission                                                                                                                                                   |
| `f0_center_st`                    | float                        | semitones             | v2, v3                       | F0 centre used for this row (causal for provisional rows; whole-session median for the final pass)                                                                        |
| `terminal`                        | bool                         | -                     | v2, v3                       | `true` only for the extra window `end - 3 s` scored at stop                                                                                                               |
| `revises`                         | float                        | s                     | v2, v3, only re-scored rows  | the `timestamp` of the earlier provisional row this final row replaces                                                                                                    |
| `asr_profile`, `asr_experimental` | str, bool or null            | -                     | only non-default/unknown ASR | `("p1_disfluent_prompt", true)`; `("unknown", null)` when the ASR object exposes no profile; absent for p0                                                                |

Other messages the demo WebSocket (`/ws`) sends: `{"status":"warming","timestamp":<s>,"message":"calentando..."}` before the first window is ready.

Session export (`GET /download/session.json` of the demo): `generado, checkpoint, modo, aviso, sin_video, audio_duracion_s, sample_rate, config_live, huecos (gaps), n_ventanas, etiquetas, ventanas (all rows)`. `aviso` repeats that the checkpoint was trained with heuristic labels and predictions show pipeline behaviour, not accuracy.

### What does NOT exist in a live row

Calibrated probability, uncertainty, disfluent/functional/uncertain state, merged events with start/end, speaker identity, visual evidence, an id, a session id (the demo has none; the app link is proposed in `APP_INTEGRATION_CONTRACT.md`).

## 2. Consumer rule (mandatory)

Take the **last row per `timestamp`**. Provisional rows are never edited or removed; a later `final` row with the same `timestamp` (and `revises`) supersedes it. A consumer that sums rows counts every window twice. The demo dashboard implements this de-duplication (`tests/test_live_demo.py::test_dashboard_counts_each_window_once_when_the_stop_pass_revises_rows`, `test_dashboard_v1_rows_without_status_are_all_kept`). Fixture check: in `live_rows_v2.json` there are 62 rows (27 provisional + 35 final: 27 revising + 8 windows only scored at stop) for 35 grid windows.

## 3. The three named configurations (`session.py:LIVE_CFG_V1/V2/V3`, default `live_cfg_v2`)

| Parameter            | v1    | v2 (default) | v3       | Semantics                                                                                                        |
| -------------------- | ----- | ------------ | -------- | ---------------------------------------------------------------------------------------------------------------- |
| `left_margin_s`      | 3.0   | 3.0          | 3.0      | past context kept when committing pYIN frames                                                                    |
| `right_margin_s` (R) | 3.0   | **4.0**      | 4.0      | pYIN frames are committed only R seconds after they were recorded                                                |
| `finalize_on_stop`   | False | True         | True     | score the tail at stop                                                                                           |
| `mark_status`        | False | True         | True     | add `status/emitted_at_s/f0_center_st/terminal`                                                                  |
| `terminal_window`    | False | True         | True     | score window `end-3` if the grid does not end there                                                              |
| `rescore_at_close`   | False | True         | True     | append one `final` row (with `revises`) per earlier window using the whole-session F0 centre and the final words |
| `lag_lookback_s`     | 3.0   | 3.0          | 3.0      | when the consumer lags, windows older than `latest - 3 s` are skipped and recorded in `gaps`                     |
| `fill_dropped_audio` | False | True         | True     | zero-fill microphone blocks flagged by the driver so the timeline keeps its length                               |
| `viterbi_carry`      | False | False        | **True** | resume pYIN's Viterbi from the exact forward scores of the last committed frame                                  |

Compatibility: v1 reproduces the behaviour of commit `80a921a` (rows without `status`, no end-of-session pass; the last about 6-8 s of every session are never scored). v2 and v3 have the same schema and scheduling; **v3 differs from v2 in exactly one variable**, so differences are attributable to the Viterbi resume. A consumer written for v2 reads v3 unchanged. Unknown config names raise `ValueError` (`get_config`). Adding a config is additive; changing a default is a behaviour change and has not been made.

Warm-up and latency: a window starting at `s` needs 5 s of future word context and R seconds of committed frames after its end, so it is ready `max(5, 3 + R)` s after its start (6 s for v1, **7 s for v2/v3**) plus the ASR time of that tick plus the 0.5 s tick (`session.py` docstring; `test_live_cfg_v2_right_margin_is_four_seconds_and_three_is_a_distinct_config`). By the same rule the first window (start 0) is ready after 7 s of audio in v2/v3 (`latest_start`, `session_features.py:338-344`, `CONTEXT_AHEAD_S = 5`); the 13 s minimum belongs to the batch path only.

## 4. Observed cost, latency and parity (existing evidence only)

All numbers below are copied from existing reports; none was re-measured for this document except the replay timing of the fixtures.

| Measure                                                                                                                                                                                                                                                   | Evidence                                                                    | Limitation                                                                                                                                                   |
| --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Per-tick cost on a 12 min real+synthetic session, 0.5 s ticks: mean 0.44-0.50 s (min 0.43, max 0.66; warm-up tick max 4.3 s); traced memory 134 MB at minute 2, 145 MB final, peak 166 MB; committed frames equal offline (0 of 72 006 differ)            | `results/ejecucion_material4_20261030/paridad/PARIDAD_LIVE_V2.md` section 6 | loaded machine, tracemalloc on; "close to real time" (0.45 s per 0.5 s tick); config not stated in the section (v2-era); one session                         |
| `SessionRecorder` keeps the whole session in memory                                                                                                                                                                                                       | same                                                                        | 3.8 MB per minute by design                                                                                                                                  |
| v2 vs v3 on 12 fresh 10.5 min sessions x 3 partitions: sessions with at least one differing committed frame 7/12 (58 %, CP 95 % [27.7 %, 84.8 %], 1-19 frames of about 63 000, up to 24 st) for v2 vs 1/12 (8 %, [0.2 %, 38.5 %], 1 frame, 5.5 st) for v3 | `PARIDAD_LIVE_V2.md` addendum A, `INFORME_FINAL` section (e)                | synthetic stress sessions (real AMERESCO clips + injected silence/noise/hum/clicks), Spanish only, no real microphone                                        |
| v3 residual: first frames of a long digital silence after speech; with R = 4 3/32 sessions pooled; with R = 30 0/12                                                                                                                                       | addendum round 5                                                            | dose-response rests on 2 synthetic sessions, post hoc R values; "0/12 at R=30" holds by construction of the generator                                        |
| v3 cost: per tick the Viterbi runs only over new frames; v2 and v3 took 460-570 s for 10.5 min under 18-way CPU load; R = 30 means commit lag +26 s and about 4x more pYIN work per tick                                                                  | `PARIDAD_LIVE_V2.md` section C and round-5 addendum                         | not a controlled benchmark                                                                                                                                   |
| Final rescore pass: one forward pass per emitted window at stop                                                                                                                                                                                           | `PARIDAD_LIVE_V2.md`                                                        | not timed                                                                                                                                                    |
| Replay of the 20 s excerpt (this task): v1 10.8 s, v2 15.2 s, v3 17.2 s wall for 20 s of audio with given words (no ASR in the loop)                                                                                                                      | `examples/provenance.json`                                                  | one run, machine shared with other jobs, **not a benchmark**; shows only that the replay finishes, not real-time capability                                  |
| Streaming vs offline ASR text: pooled word-error 0.76 on 8 clips (361 errors / 476 words), 2 s ticks, last 13 s decoded in one shot                                                                                                                       | `PARIDAD_LIVE_V2.md` section 5                                              | **it is a disagreement between two outputs of the same Whisper-small, not accuracy** (the reference is another ASR output); 8 clips, first by manifest order |

Parity statement that can be made today (from the report, section C): short clips and the 101 earlier signals are frame-exact for v1/v2; v2 is **not** exact on long sessions; v3 is exact in 11 of 12 fresh sessions with a residual of a handful of frames at the onset of long digital silences; live parity with offline is not exact end to end because the streaming ASR is not equivalent and provisional rows use a causal F0 centre. On this excerpt v2 and v3 rows are identical (`validate_examples.py`), which proves nothing about long sessions.

## 5. Decision on the default (status)

The default stays `live_cfg_v2`. v3 is recommended only where frame equality matters and remains experimental until the decision criteria are met (`INFORME_FINAL` (g): "v3 is not the default (decision needed)"). This document does not change that.

## 6. Limitations to carry into any integration

1. No microphone test was run by this task or documented as passed; the replay uses no device.
2. Live rows are windows, not events. Do not count rows as disfluencies and do not multiply by a rate.
3. `confianza` is not a probability of presence. Classes without a compatible human reference (`block`, both pauses) are emitted anyway; see `DISAMBIGUATION_SPEC.md`.
4. The ASR re-transcribes only the last 13 s on every tick; windows older than the ASR floor are skipped and recorded in `gaps` (`kind: windows_skipped`); lost audio is recorded as `audio_lost`.
5. Echo cancellation, noise suppression and AGC in the browser (the app) are ON, and the checkpoint was trained on other audio; the effect on F0/RMS features has not been measured.
6. Tests: `tests/test_live_parity.py`, `tests/test_live_stream_partitions.py`, `tests/test_live_engine_parity.py`, `tests/test_live_demo.py`.
