# Annex 2026-11-07 - restart of the cw2 secondary arm after an external kill

Dated annex under section 7 of `PROTOCOL_ANNEX_20261107_SECONDARY_CW2.md` ("a hardware or environment fix could only be
made through a new dated annex that states the cause and changes nothing else, and only if no secondary result with metrics
was ever produced"). Written and committed BEFORE the restart.

## 1. What happened

- The secondary `run` started at 2026-11-07T20:10:04Z (`RUN_SECONDARY_CW2.lock.json`, status `started`, annex sha256
  `c5bafeb5...`, system commit `a0510677ae2d...`).
- The process was launched as a background shell of the coding-agent host. The host's background-shell
  memory-pressure reaper terminated it while the first recording (`berta_es_1`) was being decoded, right after the pinned
  model loaded. The host reported the kill as its own action, not as an error of the command.
- The log ends after the model load report. No Python traceback, no CUDA error and no out-of-memory error was written.
- `CONFIRMATORY_RESULT_SECONDARY_CW2.json` was never written. **No secondary metric, partial or complete, was ever
  produced or seen.**

## 2. Cause

An environment cause outside the evaluated system: the process supervisor of the host killed the process. It was not a
failure of cw2, of CUDA, of the weights or of the runner. This is the "environment fix" case of section 7.

## 3. What changes

Only this:

1. The aborted lock is kept, unchanged in content, as `RUN_SECONDARY_CW2.lock.aborted_20261107T201004Z.json` and committed
   with this annex, so the first attempt stays on record.
2. On the operator's instruction, the runner is launched as a process detached from the coding-agent host's shell
   supervisor (created through WMI `Win32_Process.Create`), so the host's memory-pressure reaper cannot terminate it. Its output is written to a log file. It runs the same command line as the first attempt.

## 4. What does not change

The system commit, the protocol, the primary harness, the secondary runner (`run_secondary_cw2.py`, sha256 `8b5ae2b2...`,
still matching the annex manifest), the annex and its freeze, the model id, revision and weights manifest, device (CUDA),
precision (float16), batch and chunk settings, the Python environment, the environment variables (`OMP_NUM_THREADS=4`,
`HF_HUB_OFFLINE=1`, `MRCD_CORPORA_ROOT`), the data, the statistics and the secondary Holm family {S1, S2}. The primary result
and `PRIMARY_CONCLUSIONS.md` are not touched.

## 5. Limit

This is the only restart. If the restarted run fails for any cause, the arm is recorded as `not_available` with that cause,
as section 7 says, and no further annex is written for it.
