# live_cfg_v3 decision (Task 3.3)

Self-reviewed by the author; no independent review. All numbers come from synthetic stress sessions (real AMERESCO clips interleaved with silence, noise, hum, clicks; Spanish) driven through `LiveScorer` with a stub score and empty ASR words: they test the acoustic/scheduling path, not ASR or classification quality. They are not a rate for real microphone sessions. Exploratory; no class tuning; DIHANA fold 5 not involved.

**Pre-registered rule:** `docs/LIVE_CFG_V3_DECISION_RULE.md`, commit `16ab5fd` (before any run), sha256 `f3ac6e6b9326cd01ba2bd2351d023467c4aef4d1d167a550a0be583917642a6c`.
Runs: seeds 501-506 (fixed in the rule), 10.5 min each, 0.5 s ticks, one run per config (v1, v2, v3), serial, OMP=4, machine otherwise idle (system CPU 1.8-16.5 % before the runs). Raw data: `runs.jsonl`, `config.json`; tables: `live_tables.md`, `criteria.json`.

## Results (new seeds only)

| config | tick time median / p95 (s) | confirmation latency median / p95 (s after window end) | sessions with >= 1 differing committed frame | total differing frames | windows without a row | duplicate rows | finalize (rescore) s |
|---|---|---|---|---|---|---|---|
| v1 | 0.169 / 0.195 | 3.0 / 3.0 | 2/6 | 24 | 36 (6 per session: the tail is never scored) | 0 | 0 |
| v2 (default) | 0.200 / 0.275 | 4.0 / 4.0 | 2/6 | 24 (seed 501: 2 frames at 29.49 s, 3.5 st; seed 502: 22 frames at 252.14 s, up to 13.0 st) | 0 | 0 | 3.2 |
| v3 (experimental) | 0.192 / 0.262 | 4.0 / 4.0 | 0/6 | 0 | 0 | 0 | 3.1 |

* Ticks above the 0.5 s period: median of sessions 0 % in every config (the single slowest tick was 0.53 s). CPU time equals wall time (about 1 core for the acoustic path of one live session). The 1.8 GB peak working set is the cumulative process peak (10.5 min audio, offline reference extraction, torch/Silero) and is identical for all configs: it is NOT a per-config figure; per-config memory was not isolated.
* Silence (120 s of digital zeros and of -70 dBFS noise): v2 and v3 emit 462 rows each, 0 windows without a row, no gaps; v1 emits 229 rows and leaves 6 windows unscored. Tick time 0.09 s (zeros) and 0.22-0.24 s (noise).
* The divergence is the already-known event (committed frames that differ from the whole-file extraction): v1 and v2 coincide on every session (R=3 vs 4 s changes nothing here); v3 removes both cases.
* Earlier evidence (descriptive, not a gate): seeds 101-112, v2 7/12 sessions differ (Clopper-Pearson 95 % 27.7-84.8 %) vs v3 1/12; seeds 301-312, v3 with R=4: 2/12 differ (7 frames each, after long digital silence following speech). This study: v2 2/6, v3 0/6.

## Rule evaluation (literal)

| criterion | outcome | note |
|---|---|---|
| C1 completeness | NOT MET | the rule requires frame count = offline count in all 18 runs; v1 never commits the last ~3 s (62729 vs 63029 frames) by design. v2 and v3 are complete in all 6 sessions and both silence scenarios |
| C2 parity | MET | v2 shows 24 differing frames in 2/6 sessions; v3 0/6 |
| C3 latency | MET | v3 equals v2 (4.0 s) |
| C4 cost | NOT MET (not evaluable) | one v1 run started with 16.5 % background CPU (gate: < 15 %). Indicatively v3 tick median is 0.96x and p95 0.95x of v2, but the rule says not evaluable |
| C5 lost / duplicated | MET | 0 / 0 for v3 |
| C6 silence | MET | equal to v2 |

**Outcome by the rule as written: not all criteria are met, so the default stays `live_cfg_v2` and v3 stays experimental (opt-in by name). No code default was changed.**

## Evidence for and against, recommendation

For v3: it removed every divergence in this study (0/6 vs 2/6) and in 11 of 12 earlier fresh sessions, with the same latency and no measurable extra cost on an idle machine.
Against / limits: 6 sessions give very low power (v2 was also clean in 4/6); the residual after long digital silence (earlier seeds 307, 311) is not solved; synthetic sessions only; memory not isolated per config; C1 and C4 failed formally (v1's by-design tail; one load-gate violation).
Recommendation: keep v2 as default; offer v3 as the option where frame equality matters, and consider it the default CANDIDATE once the maintainer/advisor accepts either a re-run of the single load-invalid run or an amended C1 limited to v2/v3. Amending the rule now would be post hoc and is not applied here.
