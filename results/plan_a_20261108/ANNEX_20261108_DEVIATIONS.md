# Annex to exploratory protocol A: deviations (2026-11-08)

**This annex was written AFTER the results.** It records deviations between `PROTOCOL_EXPLORATORY_A.md` (frozen
2026-11-08T04:29:27Z) and what was actually run. It records deviations only. It changes no figure and no verdict in
`RESULTS_A.md` or `results_a.json`, and nothing was re-run to write it.

## D1. Feature source (protocol section 2)

The protocol said "P0 features: the existing shards". They were not used. P0 and P1 engine features were both recomputed
from audio (`scripts/plan_a/build_cache.py:36-37`, `DisfluencyEngine.unit_features`), because the stored campaign
feature shards also hold DIHANA fold-5 rows, which this plan may not load. Effect: the P0 features of this plan are a
fresh recomputation, not the shards used by earlier campaigns. No P0 vs shard comparison was made.

## D2. G4 preload set (protocol section 5)

The protocol said "preload p1 only". The G4 run used `--preload p0,p1_disfluent_prompt`, so that the same-session
`o2026_11` reference (which needs P0) could be measured in the same process. The `o2026_12` path only uses
`p1_disfluent_prompt`; the extra resident P0 model adds memory but no work on the `o2026_12` path. Its effect on the
measured latency was not isolated.

## D3. Ordering of the P1 decode relative to the cache code

- Protocol frozen: 2026-11-08 04:29:27Z (23:29:27 local), before every result file.
- P1 decode began at 23:30:56 local, before `scripts/plan_a/build_cache.py` was first committed (`68a182b`, 23:33:13;
  the P0 path is in `513d037`). The decode therefore ran from uncommitted code.
- Mitigation evidence: an 8-turn re-decode check of frozen turns gave identical text on 8/8 turns. That evidence lives
  in the git-ignored cache (`cache/verify_decode.json`). Its aggregate, counts only and no text, copied here:
  `{"n": 8, "same_text": 8}`. It is not reproducible from tracked files.
- The 4 031 new P1 decodes were produced by the same code path as the frozen ones; this is asserted by the 8/8 check,
  not independently proven.
