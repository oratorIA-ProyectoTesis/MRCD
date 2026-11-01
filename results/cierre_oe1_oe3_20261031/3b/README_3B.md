# Phase 3b summary (2026-10-31/11-01)

**Everything here is self-reviewed by the author; no independent review.** Exploratory. Defaults unchanged: V0, P0, `live_cfg_v2`; P1, Vc, `live_cfg_v3` and the new chunk-aligned streaming mode are experimental/opt-in; scores uncalibrated; DIHANA fold 5 never used; microphone check NOT run.

## Task 0 (committed `51cbaab`)
* `score.kind` is now `max_window_softmax_score` (contract revision 1.1, `provenance.service.contract_revision`); `max_window_class_probability` stays as a deprecated alias accepted by the schema (historical fixtures keep the old value on purpose). Schema file regenerated, tests added, contract docs and the unapplied app patch (`z.enum` accepting both) updated; real app untouched.
* `AnalysisResult.rate_per_minute()` returned 0.0 for audio <= 1 s (violated "absence is never zero"): now `None`; `legacy_zero=True` reproduces the old value. The service already reported such a rate as unavailable. ORATORIA-ENGINE not touched. Short suite of the touched modules: 70 passed.

## 3.2 Streaming ASR vs offline (`asr/ASR_STREAMING_VS_OFFLINE.md`)
* The "WER 0.76" was disagreement between two Whisper runs. Against human references (AMERESCO ELAN transcripts, 4 recordings; DIHANA folds 1-4, 39 speakers), streaming was NOT less accurate than offline: pooled WER 0.40 vs 0.55 on the same 67 units (AMERESCO -0.23, 4 groups; DIHANA -0.02, within noise); DIHANA WER is 0.14-0.19 in every condition. Streaming/offline disagreement is 0.40 (CI 0.19-0.52), not a loss of accuracy.
* Concrete finding: the OFFLINE 8 s chunked p0 decode falls into repetition loops on 8 of 75 AMERESCO clips (deterministic, equals the frozen cache); the 13 s streaming pass has fewer. P1 lowers offline WER by 0.054 (CI -0.100 to -0.017; DIHANA part not independent of P1's design). A 0.5 s tick was not better than 2 s (8 units). Window-class argmax changes in 31-48 % of windows across ASR paths (descriptive).
* Added (opt-in, tests): `StreamingASR.step_chunk_aligned`; makes streaming equal offline (disagreement 0) but WER vs human gets worse (it inherits the loops), so it is a consistency option, not an accuracy fix. Not measurable: streaming p1 at 0.5 s, full-set streaming (cost); ready commands in the report.

## 3.3 live_cfg_v3 (`live/LIVE_CFG_V3_DECISION.md`)
Rule pre-registered (commit `16ab5fd`, `docs/LIVE_CFG_V3_DECISION_RULE.md`) before running seeds 501-506 (10.5 min, serial). v2: 2/6 sessions with differing committed frames (24 frames, up to 13 st); v3: 0/6; same 4.0 s confirmation latency; no lost/duplicate rows; same silence behaviour; v1 leaves the tail unscored. Literal outcome: C1 and C4 not met (v1's by-design tail; one run above the 15 % load gate), so **default stays v2, v3 experimental**; v3 is the candidate if the maintainer/advisor accepts a re-run or an amended C1. Memory not isolated per config.

## 3.4 Cost (`cost/COST_REPORT.md`, `cost/COST_REPORT_tables.md`)
13 files 5-60 s (2 rejected as too short), 3 rounds + discarded cold round, idle machine. Default path (p0 + V0) RTF about 0.5; everything incl. P1 + Vc about 1.0; ASR is about 90 % of the time; features 0.04 RTF; heads and report negligible; cold start 1.5 s; peak working set 1.15 GB. Live acoustic path 0.17-0.20 s per 0.5 s tick (about one core); streaming ASR at a 2 s tick is about real time, at 0.5 s it is not. Video as a separate modality: decode + face landmarks about 7 ms/frame, face in 98.6-100 % of frames; no fusion, identity or synchrony evidence.

## Suite (OMP=4, official command of INFORME_FINAL (h))
`logs/pytest_full.log`: 815 passed, 10 skipped (218 s). `logs/pytest_slow_full.log` (MRCD_SLOW=1): 825 passed, 0 skipped (696 s).

## Files
`asr/` units.json, pred/ (per-unit transcripts), asr_analysis.json, class_descriptive.json, config.json, subset_*.json, report; `live/` runs.jsonl, config.json, live_tables.md, criteria.json, decision; `cost/` cost_runs.json, video_cost.json, tables, report, inputs/ (slices of local recordings, not for sharing); `logs/`. Code: `scripts/cierre3b/` (queue scripts `run_queue.sh`/`run_queue2.sh` record the exact order; `run_queue.sh` was interrupted after cost+video and the ASR part was re-run with `run_queue2.sh`, discarding nothing from cost/video). Note: an earlier trial of the live and cost scripts ran while a game used 35-40 % CPU; it was discarded and not reported.
