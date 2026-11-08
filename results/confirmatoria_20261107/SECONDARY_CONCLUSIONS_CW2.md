# Secondary arm cw2 - conclusions (2026-11-08)

Written from `CONFIRMATORY_RESULT_SECONDARY_CW2.json` (sha256 `0374416e...`, lock `completed`). This arm was declared in
`PROTOCOL_ANNEX_20261107_SECONDARY_CW2.md` before the primary run. It is reported separately and does not replace or change
`PRIMARY_CONCLUSIONS.md` or the primary Holm family H1-H4.

## Run identity

| Field         | Value                                                                                                                                |
| ------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| System commit | `a0510677ae2d2f906206a276e49cc88221edbe89` (same as the primary)                                                                     |
| Annex sha256  | `c5bafeb5...` (unchanged); secondary harness sha256 `8b5ae2b2...` (matches the annex manifest)                                       |
| Attempt       | second and last. The first attempt was killed by the host before any metric was produced (`ANNEX_20261107_SECONDARY_CW2_RESTART.md`) |
| System        | `cw2` ASR (`nyralabs/CrisperWhisper2.0_large`, rev `f4334f6e...`, weights manifest matches) + the primary lexical rule               |
| Device        | CUDA float16, NVIDIA GeForce RTX 5060 Laptop GPU                                                                                     |
| Status        | `ok`, 14 of 14 recordings, 3 events without a unit dropped                                                                           |

## Metrics (same cell, units and reference as the primary)

| Class                   | cw2_rule                                      | primary default      | p0_rule              |
| ----------------------- | --------------------------------------------- | -------------------- | -------------------- |
| Repetition F1 [95 % CI] | **0.781** [0.681, 0.852] (TP 91, FP 47, FN 4) | 0.286 [0.194, 0.380] | 0.525 [0.430, 0.632] |
| Filler recall [95 % CI] | **0.784** [0.658, 0.905] (29 of 37)           | 0.541 [0.379, 0.683] | 0.514 [0.381, 0.651] |

## Secondary family (Holm over {S1, S2})

| id  | Contrast                          | Delta [95 % CI]       | p_Holm | Verdict                     |
| --- | --------------------------------- | --------------------- | ------ | --------------------------- |
| S1  | repetition F1, cw2_rule - default | +0.495 [0.376, 0.625] | 0.0008 | **supported, `cw2_better`** |
| S2  | filler recall, cw2_rule - default | +0.243 [0.050, 0.441] | 0.023  | **supported, `cw2_better`** |

Descriptive only (no verdict): cw2_rule - p0_rule is +0.256 [0.166, 0.351] for repetition F1 and +0.270 [0.119, 0.414] for
filler recall. Most of the gain comes from the ASR, with the rule held fixed.

## Reading

The expectation written in annex section 4 (S1 and S2 both positive) held. Limits that apply:

- The reference is a lexical rule applied to a transcription. It favours a system that writes a verbatim text and applies
  the same kind of rule (protocol section 7). S1 is the contrast most exposed to this circularity.
- 5 Spanish speakers, one narration task. Nothing here is about human-judged disfluency, other populations or calibration.
- cw2 weights and outputs are licensed for non-commercial research only. No transcript or per-unit file is stored.
- The primary verdicts stand as written: the MRCD default fails A1, A2 and criterion (d). This arm shows that, on this cell,
  a configuration with a verbatim ASR plus the rule does better than the default. It is not the default.
