# live_cfg_v3: pre-defined decision rule

Written 2026-10-31 BEFORE any run of the new-seed study. Self-review by the writer; no independent review.
The git commit that adds this file and its sha256 are quoted in `results/cierre_oe1_oe3_20261031/3b/LIVE_CFG_V3_DECISION.md`
(the file cannot contain its own hash).

## Question

Should the default live configuration change from `live_cfg_v2` to `live_cfg_v3` (same margins L=3 s / R=4 s and same
scheduling; the ONLY difference is the exact resume of the pYIN Viterbi pass, `viterbi_carry=True`)? `live_cfg_v1`
(R=3 s, no end-of-session pass) is the historical comparator.

## Study (fixed here, before running)

* Sessions: `synth_session.build_session(10.5, seed)` for NEW seeds **501, 502, 503, 504, 505, 506** (none used before:
  earlier seeds were 1-6, 21-26, 101-112, 201-208, 301-312). 10.5 minutes each, i.e. >= 10 minutes.
* Partition: 0.5 s ticks (8000 samples), the live driver tick. One run per (seed, config) for v1, v2, v3, executed
  strictly in series (one process, no pool).
* Scorer: `LiveScorer` with a constant stub score (class scoring cost is measured separately in the cost study) and
  GIVEN empty ASR words; the study measures the acoustic/scheduling path, not ASR or classification quality.
* Reference for parity: `extract_acoustic(whole session)` frames, exact criterion (any of the 5 columns differs), the
  same criterion as `divergence_rate_*` in `results/ejecucion_material4_20261030/paridad/`.
* Silence scenarios: 120 s of digital zeros and 120 s of -70 dBFS noise, every config.
* Background load: system CPU is sampled for 3 s before every run and logged with the heaviest processes.
  A run is **load-valid** when system CPU before it is < 15 %.
* No class tuning and no DIHANA fold 5 are involved.

## Criteria (v3 must satisfy ALL to be recommended as the future default; evaluated on the new seeds only)

| id | criterion | threshold |
|---|---|---|
| C1 completeness | all 18 runs complete; frame counts equal the offline count; the silence scenarios run without exception | all |
| C2 parity | v3 sessions with >= 1 differing final committed frame <= v2's, AND v3 total differing frames <= 50 % of v2's. If v2 shows 0 differing frames on the new seeds the criterion is **not met** (no measurable benefit; absence of evidence is not evidence of benefit) | as stated |
| C3 confirmation latency | median and p95 of (stream time at emission - window end) of v3 within +/- 0.05 s of v2 | +/- 0.05 s |
| C4 cost per tick | needs **load-valid** runs: median per-tick time of v3 <= 1.25 x v2 AND peak working set of v3 <= 1.25 x v2 AND p95 per-tick of v3 <= 1.25 x v2's. With any load-invalid run the criterion is **not evaluable** and counts as not met | 1.25 x |
| C5 lost / duplicated | v3 windows without any row and not recorded in `gaps`: 0; duplicate (timestamp, status, revises) rows: 0 | 0 |
| C6 silence | on both silence scenarios v3 emits the same row count as v2 and records no unexpected gap | equal |

## Outcome rule

* All of C1-C6 met: report "criterion met on 6 sessions; recommend v3 as the DEFAULT CANDIDATE, pending maintainer/advisor
  approval". The code default is NOT changed by this study.
* Any criterion unmet or not evaluable: the default stays `live_cfg_v2`, v3 stays experimental (opt-in by name).
* Evidence from earlier seed sets (101-112, 201-208, 301-312) is quoted descriptively with Clopper-Pearson intervals
  and is NOT a gate: those seeds took part in earlier diagnosis of the defect v3 fixes.
* Six sessions give very low power: "not met" because v2 also shows 0 differences is an expected and honest outcome.
