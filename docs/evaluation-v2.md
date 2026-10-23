# Score only the evidence that humans and systems actually covered

`mrcd-evaluation-v2` separates each system's eligible universe from the common universe used for paired differences. Unknown human audio and failed/unrun cores are not negatives. Model abstention never removes human reference time. Frozen snapshots and historical evaluations are not rewritten or silently re-scored.

## Quick path

1. Freeze a reviewed snapshot. New snapshots include the canonical `recording_speakers` map as part of their hashed content.
2. Select one run per actual system and recording. Duplicate IDs (including aliases across system names), or multiple selected runs of the same actual system/recording, fail before an evaluation is saved.
3. Evaluate through the existing evaluation API. Read each system's `coverage`, `evaluable_ms`, exclusions, availability diagnostics, and metric reasons before interpreting a score.
4. For differences, use **`paired_results`**, not subtraction of the independently scored rows. The UI renders own and paired results separately and renders unavailable counts as `—`, never as zero.

Operational test commands remain exclusively in `fuente-informacion/comandos-ejecutar/COMANDOS.MD`. This document defines the contract, not a new command catalog.

## Three explicit universes

| Universe | Definition | Evidence retained |
| --- | --- | --- |
| Human `H` | Union of reviewed item regions, minus human `uncertain` / `not_evaluable` spans. | `reference_view.reviewed_coverage`, `human_exclusions`, `human_excluded_ms`, `coverage`, source kind, adapter and frozen speaker map. |
| Own `C_s` | `H` intersected with system availability. Availability is the union of exact `core_start_ms` / `core_end_ms` for segment statuses exactly `succeeded`, intersected with explicit inference coverage when supplied. | Per-run availability/reasons, each result's coverage/duration and unavailable human duration. Context/evidence padding does not count as completed output. |
| Paired `C_ab` | `C_a ∩ C_b`. Both reference comparisons and the paired bootstrap are recomputed on this same universe. | `paired_results[second-first]`: common coverage/duration, both recomputed results, delta F1, reason and shared-draw CI. |

Explicit inference coverage is `run.data.result.coverage`, either a list of millisecond intervals for that run's recording or a recording-to-interval-list map. It can narrow successful cores, never expand them. Missing segment success/core evidence is unavailable, even if the overall run says `succeeded`; there is no implicit full-recording fallback. Failed, cancelled, queued and running runs do not supply a completed evaluation universe.

Human contextual uses marked `is_disfluent: false` remain **evaluable negatives**, not exclusion masks. A disfluency false alarm overlapping an eligible functional use contributes to false alarms/minute and `false_alarms_on_legitimate`. Uncertainty emitted by a model is counted separately, not scored as an event: a real human event can still be a false negative when the model abstains.

## One symmetric boundary rule

`fully_contained_v2` requires the **entire** reference, prediction, or functional interval to lie inside one merged eligible span. Exact endpoints are allowed. An event crossing an unavailable boundary/hole is excluded whole, on both sides of matching; it is never clipped or split into a different-IoU event. Exclusions distinguish wholly outside audio from boundary/hole crossers and report reference/prediction/abstention/functional counts separately.

This conservative rule can remove a valid boundary event from a short evaluation universe. The exclusion counts are part of the result, not silently converted into TN, FP or FN. Different legitimate overlapping phenomena remain separate; exact duplicate positive reference marks are ambiguous and rejected instead of double-counted.

## Unavailable is not zero

| Evidence state | Counts / rates | F1 / reason |
| --- | --- | --- |
| No eligible coverage | Scored counts and metrics are `null`; duration is the known `0 ms`. | `no_evaluable_coverage`. Raw input counts and exclusion diagnostics remain observable. |
| Reviewed negative-only audio, no predictions | TP/FP/FN = 0 and false alarms/min = 0. | F1 is undefined: `no_reference_or_predictions`; precision/recall have their own denominator reasons. |
| Positive human event, only model abstention | TP = FP = 0, FN = 1; uncertainty count = 1. | F1 = 0. Human duration is unchanged. |
| No matched events | Boundary error summaries unavailable. | `no_matched_events`, rather than a fictitious 0 ms error. |

False-alarms/minute uses the interval **union** duration of the actual evaluated universe. A contradictory caller-supplied denominator is rejected. The lower-level event-only API remains usable without coverage for compatibility, but temporal coverage/false-alarm-rate evidence cannot be inferred from events alone.

## Speaker-cluster uncertainty

The bootstrap population comes from speakers with positive eligible **coverage**, including speakers with no reference or prediction events. Each draw copies the speaker's events, functional examples, and coverage together, gives repeated copies distinct recording/event identities, and recalculates the denominator. Paired hypotheses share the exact draw and coverage.

Outputs include `requested`, `valid`, `omitted`, `n_speakers`, seed, population mode and an unavailability reason. `n_boot` remains a compatibility alias for `valid`. Replicates with an undefined statistic are omitted explicitly, not replaced with zero. No covered speakers yields no CI; a negative-only sample may have a valid zero false-alarm rate but undefined F1. A one-speaker or small synthetic CI is algorithm evidence, **not** a claim of population representativeness.

`paired_results` is authoritative for paired CIs. The old top-level `bootstrap_ci[second-first]` alias is provided only when it cannot overwrite a named system's own CI; omitted aliases are listed explicitly. Ambiguous paired display-name collisions are rejected, not silently overwritten.

The coverage-aware evaluation path uses `covered_speakers_v2`. The old callable-only `bootstrap_ci` interface without coverage is retained and labeled `events_only_legacy`; it cannot represent negative-only speakers and is not used for new HTTP evaluation CIs.

## Snapshot compatibility and limits

- Native v2 coverage must equal the union of frozen item regions; invalid coverage is diagnosed. The canonical speaker map in new snapshot content is authoritative even if current recording/project metadata later changes.
- Earlier v2 snapshots without that map use their frozen item speaker IDs. Older unversioned snapshots use the explicit non-mutating `legacy_items_v1` adapter and retain `legacy_unspecified` reference kind when provenance was not recorded. Missing identity/ambiguous inputs are diagnosed, not invented or relabeled adjudicated ground truth.
- `reference_kind` distinguishes a single independent annotation from an adjudicated reference. Evaluation does not manufacture a second reviewer, missing consent, permission, or human validity.
- Symmetric system agreement remains **agreement, not accuracy**, and uses only common completed availability. It is separate from paired differences against a human reference.
- Its dedicated `system_agreement` path compares positive model decisions symmetrically. A model's uncertain/not-evaluable/functional output is never treated as a human unknown mask: reversing A/B leaves class F1, shared coverage, and duration unchanged. Human-reference evaluation retains its separate human exclusion semantics.
- Availability is based on the adapter's recorded completion evidence. Without a finer inference mask, a successful segment supplies its reported core; this does not independently prove that a legacy decoder covered every short/tail ownership slot. Honest finer temporal coverage and target/decoder alignment remain U4 work.
- Local numerical/HTTP/Node-renderer regression fixtures verify software semantics. No browser interaction, real-human accuracy, PostgreSQL concurrency, broader-suite acceptance, temporal retraining, calibration improvement, or visual benefit is established by U3.

Full requirements, unit status and execution evidence: `docs/tp2-closure-plan.md`. U4–U9 remain separate units.

## Reload without losing the saved protocol

`GET /api/experiments` preserves the stored v2 protocol, paired results/CI and relevant availability/reference diagnostics. A saved HTTP evaluation therefore renders the same own/paired tables after history reload, including null reasons and protocol version. Fields absent from a historical record remain absent; the projection does not inject a v2 label, migrate, recalculate, or mutate an immutable evaluation/history entry.
