# Microphone check - results

**STATE: NO REALIZADA**

> This file stays "NO REALIZADA" until YOU run `mic_check.py` on a physical microphone and complete every
> field below. The agent that prepared this kit did not access any microphone. A run made with
> `--simulate-wav` is not a microphone result, whatever its numbers say.

Change the state line above only when all "Required" fields are filled, to exactly one of:
`REALIZADA - PASS`, `REALIZADA - WITH WARNINGS`, `REALIZADA - FAILED`.

## Required (fill in by hand)

| Field                                                             | Value                  |
| ----------------------------------------------------------------- | ---------------------- |
| Date / time (local)                                               |                        |
| Run folder name (`results/microphone_check/<stamp>`)              |                        |
| Who ran it                                                        |                        |
| Input device (name as shown by `--list-devices`)                  |                        |
| Device type (built-in / USB / headset / Bluetooth)                |                        |
| Windows "Microphone privacy" permission for desktop apps was ON   | yes / no               |
| Room (quiet / normal / noisy) and distance to the microphone (cm) |                        |
| Only your voice was audible; nothing private was said             | yes / no               |
| `--keep-audio` used                                               | yes / no (default: no) |
| `--keep-text` used                                                | yes / no (default: no) |
| Duration requested (s) / audio seconds in `level_report.json`     |                        |

## Evidence to review (copy from `share_summary.json`)

| Check                                | Status (PASS/WARN/FAIL) | Value / reason |
| ------------------------------------ | ----------------------- | -------------- |
| signal                               |                         |                |
| clipping                             |                         |                |
| level (peak dBFS)                    |                         |                |
| noise_floor                          |                         |                |
| silence_edges (leading / trailing s) |                         |                |
| dc_offset                            |                         |                |
| dropped_blocks                       |                         |                |
| clock_drift                          |                         |                |

Engine side (`share_summary.json`): `n_rows` = ____, `rows_by_class` = ____, `gaps` = ____, `error` = ____

## Interpretation (your words)

- Did the console show the three prompts (silence / speak / silence)? yes / no
- Any WARN/FAIL and what you did about it (see TROUBLESHOOTING.md):
- Re-run needed? yes / no

## What this result does and does not say

- It says whether audio reached the live pipeline with an adequate level and no dropped blocks on this device.
- It does NOT say anything about the accuracy of the detected classes (they are uncalibrated and unvalidated),
  about your speech, or about any other device or room.
- The thresholds in `config.json` are operational heuristics, not calibrated standards.
