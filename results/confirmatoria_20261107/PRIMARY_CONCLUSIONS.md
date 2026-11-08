# Primary conclusions - confirmatory run on NOCANDO (2026-11-07)

Written from `CONFIRMATORY_RESULT.json` after the primary run finished and BEFORE the cw2 secondary arm
(`PROTOCOL_ANNEX_20261107_SECONDARY_CW2.md`, section 6) is started. The verdict rules are those of
`PROTOCOL_CONFIRMATORIO.md` section 5, applied as written: nothing was rerun, re-thresholded or re-framed. The secondary
arm cannot change anything in this file.

## Run identity

| Field | Value |
|---|---|
| System commit | `a0510677ae2d2f906206a276e49cc88221edbe89` (freeze commit `6b6f059`) |
| System tree sha256 | `fca42f09db7c0a60d1986cafbcbf6b463d6dcc97e8ecbfa46a753e060621af5a` (132 files) |
| Protocol / harness sha256 | `ba0c9290...` / `f5cd329b...` |
| Default decision profile | `o2026_11` (ASR profile `p0`) |
| Finished (UTC) | 2026-11-07T20:09:01Z, single run, no crash annex |
| Cell | NOCANDO: 14 recordings, 5 speakers, 1 511 units |
| Recording failures | 0 of 14 (`interpretable: true`) |

## Evaluability gate (section 6)

| Class | Positives | Recordings | Speakers | Evaluable |
|---|---|---|---|---|
| repetition (literal) | 95 | 14 | 5 | yes |
| filler_word (recall only) | 37 | 12 | 5 | yes |

## Metrics

Repetition (transcript-derived literal reference; circular in favour of lexical comparators):

| System | TP | FP | FN | Precision | Recall | F1 [95 % CI] |
|---|---|---|---|---|---|---|
| default (`o2026_11`) | 32 | 97 | 63 | 0.248 | 0.337 | **0.286** [0.194, 0.380] |
| v0 | 32 | 97 | 63 | 0.248 | 0.337 | 0.286 [0.194, 0.380] |
| rule (same p0 ASR words) | 52 | 51 | 43 | 0.505 | 0.547 | **0.525** [0.430, 0.632] |
| always_present | 95 | 1 416 | 0 | 0.063 | 1.000 | 0.118 [0.068, 0.176] |

`default` and `v0` emit identical repetition events (`default_events_equal_v0_events: true`). Leave-one-speaker-out F1 of
the default: 0.218 to 0.330.

Filler word (recall on transcript-marked non-lexical filled pauses; no precision, no F1):

| System | TP | FN | Recall [95 % CI] |
|---|---|---|---|
| default (`o2026_11`) | 20 | 17 | **0.541** [0.379, 0.683] |
| v0 | 14 | 23 | 0.378 [0.250, 0.514] |
| rule | 19 | 18 | 0.514 [0.381, 0.651] |

## Hypotheses (Holm family of 4)

| id | Contrast | Delta [95 % CI] | p_Holm | Verdict |
|---|---|---|---|---|
| H1 | F1 default - always_present (one-sided > 0) | +0.167 [0.098, 0.247] | 0.0008 | **supported** |
| H2 | F1 default - rule (two-sided) | -0.240 [-0.346, -0.161] | 0.0012 | **supported, `default_worse`** |
| H3 | filler recall default - v0 (two-sided) | +0.162 [0.000, 0.306] | 0.194 | not demonstrated |
| H4 | filler recall default - rule (two-sided) | +0.027 [-0.176, 0.209] | 0.877 | not demonstrated |

## Absolute criteria

| id | Criterion | Observed lower bound | Met |
|---|---|---|---|
| A1 | repetition F1 (default), lower 95 % bound >= 0.30 | 0.194 | **no** |
| A2 | filler recall (default), lower 95 % bound >= 0.40 | 0.379 | **no** |

## Failure criteria (section 5, "What counts as failure")

| Criterion | Status |
|---|---|
| (a) H1 not supported | not triggered: the default beats the trivial detector |
| (b) A1 not met | **triggered** |
| (c) A2 not met | **triggered** (by 0.021 on the lower bound) |
| (d) H2 or H4 supported with `default_worse` | **triggered by H2**: a one-line lexical rule on the same ASR text detects literal repetition better than the MRCD default |
| (e) recordings with a service error | none (0 / 14) |

## Conclusion

On the only untouched audio cell, the MRCD default is shown to be non-trivial for literal repetition (H1). It does not
reach its historical repetition level (A1), its filler recall is not shown to reach 0.40 (A2), and it is worse than a
one-line lexical rule on the same ASR words for literal repetition (H2). The H2 result is reported as a failure, as the
protocol requires. The reference is transcript-derived and circular in favour of the rule (section 7). That bias is
declared, but it does not soften the verdict. On fillers, the product profile is not shown to differ from v0 (H3) or from
the rule (H4).

The outcome written in the protocol before the run (default repetition F1 in [0.25, 0.45], H2 supported with
`default_worse`, default filler recall in [0.30, 0.55], H1 supported, A1 likely not met) was matched on every point.

Scope (section 1): these figures are about literal repetition against a transcript-derived reference and sensitivity to
transcribed filled pauses, in 5 Spanish speakers in one narration task. They say nothing about human-judged disfluency,
other populations, calibration, or the visual channel.

## Second look (not confirmatory)

AMERESCO, prolongation, same conversations and speakers as the spent 98-segment pool:
- default recall on marked segments: 0.355 [0.283, 0.431], 61 of 172;
- v0: 0.110 [0.068, 0.167].

The flag rate in unmarked segments (0.237 against 0.065) is not a false-positive rate, because unmarked segments are not
verified negatives. This is never called confirmatory.

## Descriptive notes

- Events with no unit, dropped before scoring: `o2026_11` 7, `v0` 10, `rule` 260.
- The default flags 2.36 filler units per minute, against 0.65 per minute of reference units with a filler. This is
  descriptive only: there is no precision for fillers.
