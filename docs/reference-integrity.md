# Freeze one traceable reference, not duplicate decisions

Adjudication is append-only and transactional. A case has one effective decision, while an explicit correction preserves its predecessor. New snapshots validate the source chain, metadata, and intervals before freezing. Existing snapshots and historical artifacts are never rewritten by these operations.

## Normal adjudication

1. A data administrator creates a case from two submitted independent annotations of the same region. The adjudicator must be an active global `adjudicator` or `admin`, distinct from A/B. Campaign-only accounts and ordinary annotators are rejected **before** either direct or bulk case creation; invalid choices do not consume the pair.
2. `GET /api/review/adjudication/{id}` returns `case.rev` and A/B annotation revisions. The browser includes those values when submitting.
3. `POST /api/adjudications` requires `case_id`, `case_revision`, `based_on` (`A`/`B` revisions), events, contextual uses, and optional reasons.
4. The server locks the case, A/B annotations/tasks, recording, and actor in deterministic order. It rechecks eligibility, source status/metadata/revisions, and interval bounds, inserts the immutable decision, and closes the case with a compare-and-swap in the **same transaction**.

The two source modes are recorded separately as `source_modes.A/B`. A blind+assisted pair is valid for development and remains **assisted** in the aggregate case/decision/snapshot; it is never relabeled blind. A later change to one track's recorded exposure is rejected even when the aggregate would remain assisted. Test references must still be blind.

The result contains decision `id` and `revision`. An identical request retry returns the same result without adding a document or history revision. A changed request based on a stale case or A/B revision returns `409`. A failure during saving/closing rolls back both the decision and the case/history changes.

## Explicit corrections

Closed decisions cannot be overwritten with `Store.update`. An authorized correction uses the same API with the **current** case revision and A/B revisions, plus:

- `supersedes`: the current decision ID;
- `revision_reason`: a meaningful explanation of at least 10 characters;
- the corrected complete event/contextual lists.

This creates another immutable document and advances the case's current-decision pointer; the original decision remains unchanged. New snapshots select the effective head only after validating the entire chain. A repeated correction request is also idempotent. The graphical closed-case view displays the saved decision read-only; correction creation is currently an explicit API operation, not a claim of a verified graphical revision editor.

Every revision must have a contiguous positive integer `decision_revision` (root = 1), valid A/B task/revision references, and a meaningful correction reason after the root. The case's `adjudication_revision` must equal the effective head's revision. These checks apply to the whole chain, not just its topology or last node. New corrections cannot extend a corrupt chain.

## Resume unfinished pre-upgrade work safely

An old open case without media/taxonomy provenance cannot finalize just by copying current metadata into a decision. An active global administrator must review it explicitly; the A/B documents, assignments, submitted status, and progress are preserved.

1. Use authenticated `GET /api/admin/adjudication-cases/{id}/review-provenance` to obtain the current case, recording, and A/B documents. Non-admin access is forbidden.
2. Review the **current** analysis audio and both annotations against the recorded guide and current taxonomy. Resolve any uncertainty outside this operation; do not claim an unchanged historical file without historical evidence.
3. Submit `POST` to the same path with the fields below. The server locks and rechecks the case, tasks, A/B annotations, recording, and administrator, then hashes the actual local `analysis.wav`. A missing file, digest mismatch, changed revision, or unconfirmed review prevents every write.
4. Reload the case after review. Its revision changed; finalize normally with the fresh case revision and unchanged A/B annotation revisions, then freeze a new snapshot.

| Request field | Required value |
| --- | --- |
| `expected_case_revision` | Current case `rev`. |
| `expected_task_revisions`, `expected_annotation_revisions` | Separate `{ "A": integer, "B": integer }` maps from the context response; positive exact revision tokens. |
| `media_sha256` | Current **derived analysis WAV** digest from recording analysis metadata, verified against the actual local file by the server. |
| `guideline_version`, `taxonomy_version` | Explicitly reviewed guide matching A/B and `mrcd-taxonomy-v1`; not a silent annotation relabel. |
| `current_source_reviewed`, `reason` | Strict `true` and a meaningful 10–500-character explanation of the actual administrator review. |

The immutable `provenance_review` records administrator/time/reason, previous missing identity, observed analysis digest, current A/B/task/recording revisions, and scope `current_source_and_annotations_only`. It explicitly records `historical_identity_verified: false`. The case and new decision reference that review; the snapshot retains the reference. Review and case update commit or roll back together. Concurrent requests over the same case revision create one audit, not duplicates; losers must reload.

**Keep identities distinct:** `recording.data.sha256` is the original upload/container identity and media-directory key. `recording.data.media.analysis.sha256` and reference `media_sha256` identify the derived `analysis.wav`. A video container and its WAV normally have different hashes. The audit labels the verified variant `analysis.wav` and retains the original's metadata separately as `recorded_original_sha256`; it does **not** claim to have verified historical original bytes. Neither ingestion identity is relabeled or overwritten by this review.

Unverified historical finals remain excluded. A closed case may renew its review **only** if it already has an immutable current-source review and a valid `mrcd-adjudication-v2` chain. An unversioned/unverified legacy final still requires a separately reviewed, curated migration preserving originals; this endpoint does not certify it, choose between duplicates, or rewrite snapshots.

### Renew a reviewed v2 case after A/B corrections

Reopening/editing/resubmitting A/B invalidates the old source review for **new certification**; it does not alter saved decisions, reviews, or snapshots. A correction without a new explicit administrator review is rejected, not silently blessed.

1. Reload the closed case and A/B context, review the current sources again, then submit the same administrator review contract with fresh case/task/annotation revision tokens, current analysis-WAV digest, affirmative confirmation, and a meaningful reason.
2. The server validates the previously saved v2 chain independently of the changed A/B, appends a new immutable review, and sets `pending_provenance_review`. Its `previous_review` preserves the preceding review (including a previously superseded pending renewal); `revises_adjudication` identifies the exact saved decision being corrected. The active review, decision pointer, decision count, and existing snapshots remain unchanged. Concurrent renewals over one case revision have one CAS winner.
3. A pending review **does not certify the old decision**: snapshot/comparison reference certification remains unavailable with a diagnostic until an explicit correction is saved. Reload once more and submit `/api/adjudications` with the fresh case/A/B revisions, current decision ID in `supersedes`, complete corrected event/contextual lists, and `revision_reason`.
4. That single correction transaction revalidates the pending review against current task/annotation revisions and media identity, appends decision revision N+1, promotes the pending review to active, and clears the pending pointer. A retry returns the same decision; failure rolls back the decision and case update together.

If sources change again after renewal, the pending review is stale and the correction is rejected. An administrator can explicitly append another renewal with fresh CAS tokens; no earlier audit is overwritten. Region/guide/speaker/split/exposure metadata cannot be silently changed through renewal. Historical source identity remains explicitly unverified, and the original-versus-analysis hash distinction still applies. These are tested API workflows, not a claim of a verified graphical renewal editor.

## Current comparison references

Comparison responses use the same validated effective-head resolver as snapshots. A correction that removes an event removes it from the **current reference lane**, while the predecessor stays stored as history. Duplicated, legacy-unverified, stale, or malformed references produce `reference: []`, `reference_status: unavailable`, and a diagnostic `reference_reason`; the UI displays that reason instead of presenting historical decisions as current truth. No-reference is not an observed count of zero. This changes reference selection only, not U3 metric/coverage semantics.

## Snapshot checks

| Check | Behavior |
| --- | --- |
| Effective adjudication | One connected revision chain and one head per case, matching its current pointer and decision counter. Sequence, reasons, and A/B provenance are checked in every revision. Disconnected, duplicated, cyclic, or otherwise ambiguous historical finals fail diagnostically; no automatic winner. |
| Source provenance | The case is closed; tasks/annotations remain submitted and match the recorded A/B revisions and metadata. Changed sources require explicit re-adjudication. |
| Identity and partitions | Ready recording, valid media SHA-256/duration, source-consistent canonical speaker, recognized split, and any existing frozen speaker partition must agree. Speaker/media leakage across splits is rejected. |
| Guide and taxonomy | One guide version per snapshot; current validated taxonomy and case/decision metadata agree. Missing/unverifiable historical provenance is a diagnostic, not an automatic certification. |
| Bounds | Reviewed regions are inside media. Events/contextual spans have valid intervals, remain inside the media, and touch their reviewed region; crossing a region boundary is not silently clipped. |
| Multiple phenomena | Different legitimate events may overlap in time within a region; they remain separate. Exact duplicate events or ambiguous overlapping review regions require explicit reconciliation. |
| Coverage | New snapshot-level `coverage` is an interval union, including touching regions, not a sum of repeated review minutes. This is coverage assembly, not uncertainty exclusion or a new metric. |
| Independent annotations | A `submitted` snapshot is identified as `human_single_annotation`, never adjudicated. Two overlapping independent annotations cannot be pooled into a single ground truth merely by freezing them. |

New artifacts use `mrcd-adjudication-v2` and `mrcd-snapshot-v2`. Snapshot content is canonically serialized and hashed; identical requests over unchanged sources return the same snapshot even under concurrent requests. Human-origin fields describe source provenance; synthetic regression fixtures do not establish real human annotation or accuracy.

## Compatibility and limits

- Old snapshots/evaluations remain immutable and readable; they are not silently upgraded, corrected, or re-scored.
- Older clients must reload to send `case_revision`. Existing IDs and annotations are not migrated by login or by opening a case.
- The administrator initial-review/renewal API is tested; no graphical migration/renewal editor or real-record migration was executed. It deliberately cannot certify finalized unversioned historical decisions.
- A historical final without a verifiable case pointer, A/B provenance, or a coherent chain cannot be newly certified automatically. The API diagnoses it for explicit integrity review/migration; it does not merge, delete, or choose human decisions.
- Missing human/EAF material remains deferred as requested. No participant, annotation, permission, agreement, or experimental result is fabricated.
- U2 reference integrity and U3 evaluation have separate evidence. The new coverage/partial/paired contract is documented in `docs/evaluation-v2.md`; a valid snapshot alone does not prove browser, scientific, or full workflow validation. Historical evaluations are not silently re-scored.
- SQLite HTTP/concurrency/rollback regressions are verified locally. PostgreSQL concurrency, interactive browser correction workflow, and real-human-reference validation remain unverified.

The full acceptance and execution evidence are tracked in `docs/tp2-closure-plan.md`; operational test commands remain in the canonical `fuente-informacion/comandos-ejecutar/COMANDOS.MD`.
