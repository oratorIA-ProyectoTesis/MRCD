# Live path parity, round 2 (2026-10-30)

Base: HEAD 80a921a. All commands run with `OMP_NUM_THREADS=4` (scripts pin it before numpy; pytest launched with it in the environment). Raw outputs are in this folder. Nothing here used DIHANA fold 5; the margin was NOT assumed valid and turned out to be insufficient (section 3).

## 1. What changed in the code (all under mrcd/live, scripts/live_demo.py, tests, scripts/live_parity)

| change | where | why |
|---|---|---|
| `SessionFeatureTracker.finalize()` / `closed` | `mrcd/live/session_features.py` | commits every remaining frame with no right margin; pads the last Silero chunk and clamps the last segment to the true length exactly like the offline path; idempotent; `append` after close raises |
| `LiveScorer`, `LiveConfig`, `live_cfg_v1`, `live_cfg_v2` | `mrcd/live/session.py` (new) | deterministic scheduling core (no threads, mic or ASR). Rows carry `status` (`provisional`/`final`), `emitted_at_s`, `f0_center_st`, `terminal`, `revises` (v2 only) |
| per-session state, `stop()` reads the ring tail synchronously, worker runs `finalize` | `scripts/live_demo.py` | fixes the stop->start race (old worker could append rows/words into the new session) and the lost 6-8 s tail |
| `StreamingASR.merge_recent` (pure) | `mrcd/live/streaming_asr.py` | session words are no longer shared mutable state |
| skipped windows recorded in `gaps` (`kind: windows_skipped`), lost audio `kind: audio_lost` | `mrcd/live/session.py` | the consumer lag skips were silent |
| `MicrophoneStream(fill_dropped=...)`, `dropped_blocks/samples` | `mrcd/live/mic_stream.py` | driver-flagged blocks were discarded and the timeline silently shortened |
| `--live-config {live_cfg_v1,live_cfg_v2}` | `scripts/live_demo.py` | choose the behaviour; default `live_cfg_v2` |

`live_cfg_v1` = previous behaviour (margins L=3/R=3, no status field, no end-of-session pass, tail unscored). `live_cfg_v2` = L=3 s / **R=4 s**, end-of-session pass with terminal window and rescoring, status fields, zero-fill of driver-dropped blocks.

Provisional vs final (documented in the `mrcd/live/session.py` docstring): a provisional row uses the causal F0 centre at that tick and the ASR words of that tick and is never edited. Future context used: pYIN commit lag R = 4 s plus 5 s linguistic look-ahead, so a window is ready `max(5, 3 + 4)` = 7 s after its start = 4 s after its end (measured: median/p95/max 4.0 s after window end, `v1_v2_comparison.json`), plus ASR time and the 0.5 s tick. `finalize` scores the tail with no future context, optionally the terminal window (`end-3 s`, only when more than one 10 ms hop past the grid), and appends `final` rows with `revises=<timestamp>` for every earlier window using the whole-session centre and final words. In v2 the final pass also fills every grid window that never got a provisional row (skipped at the start or under consumer lag), so after close every grid window up to `end-3 s` has a `final` row. A consumer takes the last row per timestamp.

## 2. Parity levels (a-f)

| level | state | evidence / tolerance |
|---|---|---|
| (a) stored features vs engine | unchanged, tested in the previous round | golden V0 + `rerun_comparisons.json` (not re-run here; engine untouched) |
| (b) acoustic extraction | tested, EMPIRICAL, exact tolerance, divergence found with R=3 | section 3. With v2 margins: 0 differing committed frames and 0 after `finalize` on 101 signals x 3 partitions x 8 margin pairs + 8 min session + four 10-10.5 min sessions. With v1 margins (3,3): 1 frame (3.16 st, 27.71 s) in one 10.5 min session. Tolerance `array_equal` because frames are deterministic functions of the audio |
| (c) tensors/probs, offline F0 centre injected | tested, exact | `test_c_offline_centre_injected_rows_equal_offline_exactly` (provisional and final rows, random chunking, fake score = random projection of ALL tensor values). **This does NOT show autonomous streaming operation: the centre is supplied by the offline pass, which a live system does not have.** |
| (d) F0 centre computed live (causal) | NOT equivalent; divergence measured | provisional rows: 20/622 argmax flips previous round (PARIDAD_LIVE.md); this round only the exact bound (non-F0 channels exact, centred F0 off by <= the centre difference) is asserted. Pilot clip decision with causal rows: TP/FP/FN/TN 4/1/12/15 vs offline 3/1/13/15 (1 flip, n=32, not an accuracy claim) |
| (e) ASR in streaming | measured, NOT equivalent | section 5 |
| (f) final decisions / events | partial, tested | after `finalize` the live centre equals the offline one, and the `final` rows equal the offline engine probabilities EXACTLY given the same words (`test_f_final_pass_...`; pilot: 0 windows differ in 105+ windows of 32 clips, `v1_v2_comparison.json`). Words are GIVEN (cache), so this is decision parity of the feature path, not of the ASR. Event merging (`DisfluencyEngine._merge`) is not run on live rows |

Tests added (`tests/test_live_stream_partitions.py`, `tests/test_omp_import_side_effects.py`): arbitrary partitions (random/tiny/mixed chunk sizes, 1-sample chunks, non multiples of 160/512, `commit=False` batching) vs offline and after close; sessions of 0.4/2.99/3.0/4.2/5.9 s; terminal-window limit at 0/1/159/160/161/4000 samples past the grid; device gap (`lost`) equals zero-filled audio; skipped windows fully accounted for; v1 keeps the historic schema and tail loss; stop scores the tail and is idempotent; quick stop->start with a stalled ASR leaks no stale rows or words (words tagged by audio, not by mutable state); 10.5 min real+synthetic session (MRCD_SLOW). Tolerances: exact everywhere except (d) (bound only), because any other tolerance would hide real divergences.

## 3. Extended margin study (task 2)

Script `scripts/live_parity/margin_study_v2.py`. Signals: 98 AMERESCO clips, the 40 s YouTube clip, a 4.5 min concatenation (`concat8`), an 8 min real+synthetic session, and 10-10.5 min sessions with seeds 3-6 (`synth_session.build_session`: real AMERESCO clips interleaved with silence up to 30 s, noise, hum, ramps, clicks, clipped bursts). There is no other local audio (`data/` has none). Partitions: fixed 0.5 s ticks, random tick 0.02-3 s, tiny 1-511 sample chunks with a random commit schedule, and a mixed 0.25-1 s tick partition. Raw: `margin_study_v2_main.{json,log}`, `margin_study_v2_long10_5_seed3.*`, `margin_study_v2_long_more.*`.

Main sweep (98 clips + yt40 + concat8 for every pair; 8 min session for 3:3, 3:4, 2:3), differing committed frames (also after finalize):

| L:R (s) | tick | rand | tiny | worst |
|---|---|---|---|---|
| 2:3 | 7 frames, 1 signal, 1.14 st | 0 | 0 | 7 frames (partition dependent) |
| 3:2 | 24 frames, 1 signal, 12.1 st | same | same | 24 frames, 12.1 st |
| 3:3 | 0 | 0 | 0 | 0 on all 101 short signals and the 8 min session |
| 4:3, 6:3, 3:4, 4:4, 3:6 | 0 | 0 | 0 | 0 |

Long-session finding: a 10.5 min session (seed 3, partition mixed 0.25-1 s ticks) has **1 frame (index 2771, 27.71 s) that differs at L=3,R=3** (F0 3.16 st vs unvoiced; also persists after finalize because committed frames are never rewritten). It persists at L=4,5 with R=3 and disappears at R=4 and above (3:4, 3:5, 4:4, 5:5, 6:6 all 0). Four further sessions (seeds 4, 5, 6 plus the seed-3 rand partition): 3:3, 3:4, 3:5 all 0. Therefore: the 3 s right margin is NOT validated; the failure is two-sided Viterbi dependence that no short clip exhibits. `live_cfg_v2` uses R=4 (smallest tested value with 0 on everything). It is still empirical, not a proof: the worst observed case at R=3 was 1 frame in about 64000, and R=4 has simply not failed yet. Cost: +1 s latency (window ready 7 s after start).

## 4. v1 vs v2 on the 32 pilot clips (same unit as PARIDAD_LIVE.md section 2; `v1_v2_comparison.json`)

Clip decision "prolongation anywhere in windows contained in [a,b]", real checkpoint, cached words, 0.5 s ticks.

| variant | TP/FP/FN/TN | flips vs offline | contained windows scored |
|---|---|---|---|
| v1 (provisional only) | 4/1/12/15 | 1 | 105/105 |
| v2 provisional | 4/1/12/15 | 1 | 105/105 |
| v2 final (last row per timestamp) | 3/1/13/15 | 0 | 105/105 |
| offline, offline centre | 3/1/13/15 | 0 | 105/105 |

v1 vs v2 provisional: 489 common windows, 0 with different probabilities; 64 windows emitted by v1 are not emitted provisionally by v2 (they are the last second that the larger margin delays; v2 scores them in the final pass). The change alters the clip decision of one pilot clip only through the final pass (centre), moving 4/1/12/15 to the offline value 3/1/13/15. The tail pass does not change this table because pilot targets lie away from the clip end; it matters for events near the end of a session (about 6-8 s were unreported under v1). `v1_v2_comparison_R3_superseded.json` is the same run with v2 at R=3 (before the margin change), kept for traceability.

## 5. Streaming ASR vs offline ASR (task 5; `asr_streaming_parity_v2.json`)

Setup: first 8 pilot clips, faster-whisper small int8 CPU (the model dir of `scripts/improve/asr_cache.py`), OMP=4. Offline = engine path `transcribe_chunked(8 s, 2 s)`; re-running it reproduces the frozen cache exactly (WER 0.0 on 8/8, confirming the environment). Streaming = `StreamingASR.merge_recent` on the last 13 s every **2.0 s** (not 0.5 s, CPU budget) plus a final tick at the end.
Result: **not equivalent.** WER between offline and final streamed text per clip: 2.69, 0.41, 0.97, 0.78, 1.00, 0.89, 0.22, 0.30; pooled 361 edit errors over 476 reference words (0.76). The reference is another ASR output, not the ground truth, so this measures disagreement, not accuracy; but both feed the model. Causes seen in the texts: the streaming path decodes 13 s in one shot (Whisper rewrites toward fluent prose, drops repetitions and fillers that the chunked 8 s path keeps) and one clip produced 75 words against 26 (decoding loop). Word start times of matched words differ by a median of 0.01-0.66 s. 8 clips, first by manifest order, not a random sample; a 0.5 s tick would give different intermediate texts. A full-length run (32 clips, 0.5 s ticks) was not done: about 5 min CPU per clip under load.

## 6. Cost and memory over a long session (task 3)

`long_session_cost.py` on a 12 min real+synthetic session, 0.5 s ticks, tracemalloc: mean tick 0.44-0.50 s in every minute (min 0.43, max 0.66; first minute 0.50, last 0.44), traced memory 134 MB at minute 2, 145 MB final (peak 166 MB), committed frames equal offline (0 bad of 72006). The growth hypothesis in VERIFICACION.md is not supported: per-tick cost is dominated by the constant-size pYIN recompute, not by session length, so no sliding-state change was made (nothing to compare before/after). The cost is however close to real time (0.45 s per 0.5 s tick with tracemalloc and a loaded machine; tick max 4.3 s on warm-up). `SessionRecorder` still keeps the whole session by design (3.8 MB/min).

## 7. OMP (task 4)

Fresh-process tests (`tests/test_omp_pin_subprocess.py`, 11 existing + `tests/test_omp_import_side_effects.py`, 4 new): (i) set before numpy is effective and silent; (ii) assigned after numpy/mrcd import is flagged `effective=False` with a warning; (iii) `MRCD_NO_OMP_PIN=1` leaves the caller's environment untouched; (iv) importing `mrcd`, `mrcd.engine` and the live modules changes no environment variable, warnings filters, logging root, numpy error/print state or torch thread count (with OMP unset, set to 7, or opted out), and no module under `mrcd/` imports `_pin_omp` or assigns the variable except the opt-in helper `pin_asr_threads` in `reproducibility.py`. Only `scripts/` entry points pin (documented in `scripts/_pin_omp.py`).

## 8. Suite

Literal output in this folder. `OMP_NUM_THREADS=4 pytest -q -rs`: **471 passed, 2 skipped** (the two MRCD_SLOW tests; `pytest_full.log`). `MRCD_SLOW=1 OMP_NUM_THREADS=4 pytest -q -rs`: **473 passed, 0 skipped** (`pytest_slow_full.log`, 6 min 22 s). Previous baseline 279 + 1 skipped / 280 passed. Two earlier slow runs failed and led to fixes: the long session exposed the R=3 divergence (section 3), and the final pass left the first grid windows without a `final` row.

## 9. Not proven / still open

- Margin R=4 is empirical; Viterbi dependence is unbounded in principle; only Spanish AMERESCO speech, 16 kHz, synthetic stretches; no 8 kHz telephone audio, no real device noise.
- (c) and (f) use GIVEN ASR words and (c) an offline-injected centre: they show the feature path is exact, not that the live system works autonomously. Autonomous operation = (d) + (e), both measured as divergent.
- Streaming ASR parity: 8 clips, 2 s tick; not 0.5 s.
- Events from live rows (merge) are not built; live rows are per-window.
- int16 peak rescale is not applied in the stream; Vc head not wired in the live path.
- Real microphone, real device gaps and PortAudio flags were not exercised (fake mic only); `fill_dropped` zero-fills a flagged block, which also discards valid samples in it.
- `SessionRecorder` memory grows with the session by design.
- The final rescore pass costs one forward pass per emitted window at stop (hundreds on long sessions); not timed.


---
# ADDENDUM 2026-10-31: statements above that are WITHDRAWN or narrowed (the original text is kept unedited)

Cause: independent round-3 review (`../revision3/REVISION3.md`, section 0.1) plus the root-cause work and the new-session study below. Corrections to this report were written by the same writer as the original (self-review) unless a new independent review is listed in `../INFORME_FINAL.md`.

**Withdrawn:** section 2(b) "0 differing committed frames ... exact", section 3 "R=4 is the smallest value with 0 on everything" and the `LIVE_CFG_V2` comment. The R=3 -> R=4 change did not fix anything structural: the single frame that vanished in seed 3 vanished by chance.

## A. Measured rate for `live_cfg_v2` (L=3, R=4)
12 NEW 10.5 min sessions, seeds 101-112 fixed in the script header before the first run (none used earlier: earlier seeds were 1-6 and 21-26), x 3 partitions (0.5 s ticks, random 0.02-3 s ticks, mixed 0.25-1 s ticks), exact criterion (any of the 5 columns differs from `extract_acoustic` of the whole file; `paridad/divergence_rate_fresh.jsonl`, `.summary.json`):

| config | sessions with >= 1 differing committed frame | 95% Clopper-Pearson | (session, partition) | max frames in a job | max F0 error |
|---|---|---|---|---|---|
| live_cfg_v2 | 7 / 12 (58%) | [27.7%, 84.8%] | 18 / 36 | 19 | 24.0 st |
| live_cfg_v3 | 1 / 12 (8%) | [0.2%, 38.5%] | 3 / 36 | 1 | 5.5 st |

Known failing sessions (`divergence_rate_known.jsonl`, not mixed into the fresh rate): v2 2/2 sessions (seed 23: 3 or 13 frames, seed 26: 3 frames), v3 1/2 (seed 23 keeps 3 frames at 438.35 s; seed 26 is exact).
So "exact" is replaced by: v2 differed from offline in a majority of long sessions (rate 58%, wide interval), by 1-19 frames of ~63 000 (<= 0.03%), up to 24 st on a frame; v3 differed in 1 of 12 fresh sessions (1 frame). Both numbers are from synthetic stress sessions (real AMERESCO clips interleaved with silence, noise, hum, clicks) and Spanish speech only; they are not a rate for real microphone sessions.

## B. Root cause (isolated with one-variable experiments; `scripts/live_parity/` and `mrcd/live/pyin_carry.py`)
1. The pYIN observation probabilities are bit-identical regardless of where the slice starts (max abs diff 0.0 over three slices of seed 23). The difference is only in the Viterbi decode.
2. Seed 23, frame 295.56 s (13 frames, voiced at 400 Hz = fmax with voiced_prob 0.01 vs unvoiced offline): decoding the slice [t0, end] gives the offline answer for every start t0 <= 292 s and the wrong answer for t0 >= 293 s, and the answer does not depend on the right end of the slice (296 s ... end of file). It is a LEFT dependence, which no left margin of 3 s fixes if the slice start lands in a stretch of room noise: in that stretch the voiced and unvoiced hypotheses keep a constant score gap, the optimal paths do not coalesce, and the restart from a uniform prior at the slice start picks a different path. This is a defect of the tracker (restart from a uniform prior is wrong when the true forward scores are known), not an inherent limit.
3. Fix = `live_cfg_v3`: keep the exact forward scores of the last committed frame and resume the Viterbi from them (`pyin_carry.py`; same arithmetic and tie-break order as librosa; verified bit-equal to `librosa.pyin` on whole signals and on resumed decodes, tests `test_pyin_resumable_*`). A 40 s slice (270-310 s of seed 23) reproduces the defect in 15 s: v2 fails, v3 exact (`test_seed23_left_dependence_v2_diverges_and_v3_is_exact`, MRCD_SLOW). With v3 the partition-dependent divergences disappeared (all 3 partitions identical for every session).
4. Residual of v3 (3 of 36 jobs, one session per case, identical in all 3 partitions): seed 23 at 438.35 s (3 frames, 5.6 st) and seed 111 at 321.27 s (1 frame, 5.5 st). Both are the first frames of a 19-23 s stretch of DIGITAL SILENCE that starts right after speech. A larger right margin does not remove them (seed 111 R=6, 8, 12: still 1 frame; seed 23 R=6, 8: still 3 frames), so it is the future-dependence of the backtrack through a perfectly degenerate stretch (all-tied observations) that is longer than any practical R. Documented as inherent for pYIN's global Viterbi: it cannot be bounded by a finite margin; it needs silence longer than R preceded by voiced speech.

## C. Contract that can be stated now
* Short clips and the 101 earlier signals: frame-exact for v1/v2 (unchanged evidence).
* v2: NOT exact on long sessions (see A). Keep it reproducible; do not use "exact".
* v3 (`--live-config live_cfg_v3`, not the default): exact in 11 of 12 fresh sessions x 3 partitions; in the 12th, 1 frame of 63 000 differs after a long digital silence; observed maximum 3 frames / 5.6 st per session. Tolerance for downstream use: a handful of frames (<= 0.005%) at the onset of long digital silences; the window tensors containing them can differ for those frames only. Latency identical to v2 (window ready 7 s after its start). Cost: per tick the Viterbi runs only over the new frames (not slower: v2 and v3 took 460-570 s for 10.5 min under 18-way CPU load).
* The default stays `live_cfg_v2` (no behaviour change without a decision); v3 is recommended where frame equality matters.
* Not done: v3 on the 32-clip pilot decision table (the pYIN path is exact on short signals so the table cannot change; not re-run); real microphone.

---

## Addendum 2026-10-31 (round 5, self-review; the text above is kept unchanged)
* **B.4 "residual inherent / cannot be bounded by a finite margin" is withdrawn.** It depends on the length of the degenerate stretch. Seed 111 (truncated at 360 s) is exact with R = 30 s. New seeds 301-312 (fixed before running), tick partition, v3: R = 4 s -> 2/12 sessions differ (CP 95% [2.1%, 48.4%]; 7 frames each, seeds 307 and 311, at the end of the same real clip followed by a digital silence of 29.09 s and 15.05 s); R = 30 s -> 0/12 (upper bound 26.5%). Dose (post-hoc, 2 sessions): seed 311 (silence 15.05 s) R = 4, 12, 14 -> 7 frames, R = 16 -> exact; seed 307 (29.09 s) R = 12 -> 7 frames, R = 30 -> exact. Pooled R = 4 over the three studies: 3/32.
* Cost: commit lag = R (+26 s at R = 30); about 4x more pYIN work per tick (loaded machine, not a controlled benchmark). Default unchanged (`live_cfg_v2`; v3 uses R = 4).
* Live parity is NOT exact: evidence is synthetic sessions with given ASR words and an injected F0 centre; streaming ASR is not equivalent (WER 0.76, 8 clips, 2 s tick).
* `pyin_carry` no longer patches librosa (thread-safe). Details: `results/revision5_20261031/REVISION5.md`.
