# Confirm human type and context without promoting proposals

**U5a software semantics passed scoped independent review (107 focused + 18 supplied + 6 new probes).** New opt-in annotations use
`semantic_schema=mrcd-human-annotation-v1`; normalized marks carry
`human_event_schema=mrcd-human-event-v1`. Synthetic tests exercise actual UI
action functions → HTTP PATCH → submit → adjudicate → freeze. This is not real
human annotation, model accuracy, calibration or browser acceptance.

## Independent human judgments

| Field | Meaning |
| --- | --- |
| `start_ms`, `end_ms` | Original human-marked integer recording-clock bounds; no clipping, quantization or refinement. |
| `label`, `type_known` | Canonical phenotype and explicitly confirmed knowledge. Proposed label alone is not positive evidence. Functional filler stays `filler_word`, never `fluent`. |
| `is_disfluent` | Explicit human context: true, false (functional), or null (not determined). Never inferred from `decision=event`. |
| `contextual_state` | Server-normalized disfluent/functional/null view of the explicit context judgment. |
| `decision`, `uncertainty` | Preserve event/uncertain/not_evaluable and class/boundaries/audio. Uncertain context or boundaries do not erase a genuinely known type. |
| `source_kind`, `provenance` | Derived by the authenticated server, not trusted client claims; actor, exposure, confirmation and source binding remain explicit. |

To establish human conclusions, a PATCH must include a `confirm_human_event` action for a mark's unique `event_id`,
with `reviewed_event` equal to its current seven-field content: `start_ms`,
`end_ms`, `label`, `decision`, `uncertainty`, `type_known`, `is_disfluent`.
Absent confirmation clears type knowledge/context. Stale confirmation for changed
bounds/type/context rejects atomically. Revocation clears current conclusions but
retains action history. Unchanged marks retain the original confirmation's
provenance on unrelated autosaves, not a later writer's identity. Reuse is valid
only when the complete original proposal binding (suggestion ID, run/event
reference and original source bounds) is unchanged. New UI actions also include
`reviewed_proposal`; a changed source requires a new attestation, not replay of
an old confirmation. Proposal revisions are strict integers, not booleans or
floats that compare equal to integers.

The UI separates **Copy without confirming** from **Review and confirm**. Type
confirmation starts unchecked; contextual choice starts not determined. Copying
A/B for adjudication likewise requires personal review. Copying A/B retains its
explicit event/uncertain/not_evaluable decision and class/boundaries/audio reason;
copying does not turn boundary uncertainty into a confirmed event. Switching an
existing mark to functional use retains its ID and original proposal binding. Drag/split/merge revoke
conclusions pending confirmation. Functional edits keep phenotype/source binding.

## Source binding and atomic updates

Assisted GET projects unconfirmed proposals from the assigned completed run.
`proposal_ref` binds run ID/revision, event ID and canonical source-event SHA-256.
Copy this returned reference exactly. Proposal context is separately named;
human type/context remain false/null until confirmation. Scores are not exposed.

PATCH validates assigned recording/run, visible event, digest and current run
revision. Spoofed/stale/wrong-run references reject before annotation/task writes.
Annotation, task and run are locked during normalization and annotation CAS; an
idempotency key cannot name different edits. Blind tasks expose no suggestions or
authorization to proposal references.

Adjudication validates current A/B revisions and explicit confirmations. Its
server-derived provenance retains source-annotation revisions and original model
proposal bounds. Freeze preserves event fields/provenance and item-level semantic
version, including negative-only regions. Existing documents and frozen bytes are
not migrated or rewritten. Historical unversioned reads remain unchanged; new
claims/proposal verification require the explicit versioned route.
Unbound legacy `suggestion_id` links cannot be certified as verified model
provenance: the new adjudication route rejects them clearly without a partial
write. A fresh independent human judgment can retain A/B source revisions without
inventing an old model binding; the original legacy records remain intact.

## Historical drafts: edit first, upgrade only explicitly

Unversioned historical read/write remains available, including contextual marks
without IDs. The actual UI autosave envelope keeps that version; historical
edits do not create v1 confirmation actions or type/context knowledge. A 422 is a
validation rejection, not a connectivity failure to retry forever. Only a fresh,
empty annotation at revision 1 opts into v1 automatically.

Both PATCH and adjudication reject recognizable new type/context/proposal or
provenance claims outside the authenticated versioned route. This includes
residual event markers even when the outer schema was removed. A stored v1
annotation cannot be downgraded by omitting its schema. Versioned A/B sources
require versioned adjudication even when both tracks contain no events. These
checks execute before writes and before idempotent-return paths; freeze also
rejects retained unversioned new claims. Genuine frozen history is not rewritten.

A legacy draft's UI offers an explicit transition, separate from editing:

- `POST /api/annotations/{id}/upgrade-semantics` requires current
  `expected_annotation_revision`, `confirm_upgrade=true` and a meaningful reason.
  Normal assignment/role restrictions still apply; stale revisions reject.
- The server preserves the original immutable revision, records its canonical
  content hash, actor/reason/time in `semantic_transition`, and retains each
  complete old mark in `legacy_original` with unchanged raw bounds.
- Current marks receive stable IDs but **no invented confirmation**: type is
  unknown and context null. Old contextual judgments remain in the original
  revision/mark, not silently certified under v1. Old unbound model links are
  likewise retained as historical originals, not authenticated new references.
- Subsequent explicit human confirmation can establish type/context independently.
  The original survives editing, adjudication and freeze; clients cannot invent
  or replace its binding. The transition proves data preservation, not correctness
  of a historical judgment or historical source identity.

### Derived original lineage without implicit human knowledge

`source_lineage.schema=mrcd-human-source-v1` is a reference handle, **not an
authentication credential or human confirmation**. It binds annotation ID,
annotation revision, source event ID and SHA-256 of the complete source event
excluding its own reference field. The server checks both the actual immutable
revision and unchanged current source content, scoped to this annotation or the
case's authorized current A/B. Foreign, forged, missing-child and edited-source
stale references reject atomically. A status-only revision does not invalidate
an otherwise identical verified source. No external authenticity is claimed.

A new split child's editable ID is separate from that source identity. The
immutable `legacy_original` survives; server-derived `provenance.parent_lineage`
retains its verified derivation. Successful saves return current self-reference
handles; the UI refreshes **only references**, not actors/confirmation, for pending
edits. Review GET can project handles for pre-lineage v1 marks without rewriting
the stored document. Existing same-ID pre-lineage marks can retain their exact
original; new IDs require explicit verified lineage. Removing/replacing an
existing immutable original is rejected, not treated as a fresh judgment.

The actual A/B form mapping carries original and lineage independently of model
suggestion binding. A fresh adjudicator confirmation records that adjudicator,
not a copied source actor. Contextual→event edits retain a nonempty existing ID or
allocate a unique one, preserving expression/function and original fields. All
geometric mutations share a semantic-family guard: legacy edits stay legacy;
v1 drag/split/merge revoke conclusions until explicitly confirmed. Copying or
splitting never invents knowledge.

### Undo after autosave: preserve snapshots, rebase only transport

Undo JSON snapshots retain their original events, intervals, judgments and
provenance. Successful authenticated responses populate the UI-only
`_lineage_cache`; Undo carries that cache forward separately. `humanSaveEnvelope`
clones the outgoing body and rebases reference handles from this server knowledge,
never rewriting the historical snapshot. The cache and read-only registry are
removed from the submitted envelope. Client cache data is not authority: the
server still validates every reference. Ordinary `mrcd-human-source-v1` stale,
foreign and forged content checks are unchanged.

A deleted source or an origin removed by merge is absent from current events, so
ordinary source references correctly cannot restore it. The bounded repair uses
**explicit `mrcd-human-restore-v1`**, not a fallback accepting arbitrary history:

- Server-owned `lineage_restore` entries are created only for legacy-original
  marks actually removed from this same annotation. They bind annotation ID,
  **current** annotation revision, immutable source revision/event/hash and exact
  original hash. The registry is in the existing annotation document; no new
  service, global authorization storage, config or database migration.
- Assignment/role authorization and annotation CAS still apply. Validation requires
  a registered handle, exact immutable source/original and current revision. A
  public-looking handle grants no access and confirms no human knowledge.
- Handles are reissued only against the validated current revision and consumed
  when that event returns. Old handles cannot replay after another revision;
  identical client_op retries retain the existing idempotent result. Transaction
  failure rolls back both the annotation and handle consumption.
- Restoration always records authenticated actor and
  `restored_pending_human_review`: type is unknown and context null, even when
  Undo's untouched historical snapshot contained an older confirmed judgment.
  This pending status survives further autosaves. A new explicit human action
  with a fresh nonempty string `confirmation_id` is required. It must be unseen
  across this annotation's immutable versioned action history, even if an older
  Undo snapshot shortened the editable action list; replaying the old identifier
  with changed content is rejected. UI transport recognizes confirmation IDs independently of JSON key
  order and never promotes restored conclusions automatically.
- The reference cache can carry pending/restoration handles across successive
  autosaves and repeated delete/merge Undo. Copies of independent original
  intervals remain available; nothing silently discards `legacy_original`.

Pending review follows the **authenticated current source**, not just the editable
event ID. Splitting a restored mark, deriving another child, moving it between
event/contextual tracks, or copying an authorized A/B source into adjudication
cannot reset that obligation. After the unchanged source-v1 identity/content/
history checks, the server inherits pending status from the prior same-ID mark
or its verified source. Neither client `provenance` nor `parent_event_id` proves
pending state or human knowledge. Unconfirmed descendants keep unknown type,
null context and `restored_pending_human_review` through further saves/freeze.

An explicit confirmation on this pending lineage requires a nonce unseen in the
editable annotation **and the authorized source annotation's immutable versioned
action history**, even when a new child ID or a shortened action list is used.
An old/invalid nonce rejects the complete write; it cannot create a final decision
or close a case. A genuinely fresh human action records the authenticated reviewer
and can establish type/context; ordinary later autosaves preserve it. This adds
no new authorization, registry, migration or relaxation of stale-source/RESTORE
validation. Existing historical decisions/snapshots are not rewritten.

This is repair of editable history usability and attestation integrity, not a
scientific feature or confirmation that a historical judgment was correct.

A direct v1 PATCH of nonempty historical data is not an implicit migration; it
rejects until this explicit transition. Historical final/frozen records are not
mutated by this operation.

These checks prove binding/self-consistency and authenticated attestation, not
that a real person listened, that the judgment is correct, or that permission
exists. Synthetic fixtures never establish human validation.

## Acquisition is not model use or quality

New cells add `evidence.schema=mrcd-cell-evidence-v1`:

- **Acquired:** actual observed per-measure support.
- **Effective:** masks after registered mode, trained modality capabilities,
  actual visual-frame eligibility and trained-column availability. Owned versus
  contextual support stays separate. Existing features/targets/model eligibility
  are unchanged.
- **Quality:** null with `quality_not_assessed`, never a good-quality boolean
  inferred merely from feature presence. Missing modality reasons persist.

Acquired visual features can be true while effective visual input is false for
audio/text or untrained visual weights (`mode_disabled`, `modality_not_trained`);
quality remains unmeasured. This is tested on synthetic features. Product visual
acquisition is still unavailable until U6.

## Boundaries of this slice

Only the tested assisted proposal/confirmation adapter is enabled for new
cell/window results. Task-unaware evaluation and CSV/EAF guards remain. New human
references, including negative-only snapshots, reject those operations pending
the corresponding adapters; JSON retains full records.

| Later slice | Still open |
| --- | --- |
| U5b | Separate phenotype/context eligibility and named uncertainty/boundary scoring policy; no metrics implemented here. |
| U5c | Validated `snapshot_cells` dataset adapter. |
| U5d | Complete API/JSON/CSV/EAF/UI interchange roundtrips. |
| HU059 | Separate experimental boundary proposals, never overwriting authoritative raw intervals/cells. |

Commands remain only in `fuente-informacion/comandos-ejecutar/COMANDOS.MD`, U5a
section. Accepted U4a/U4b contracts and original data/models/report are protected.

### U5a fix-review evidence

Eight findings (nine scenarios) were reproduced before fixes. The independent
UI legacy probe hardcoded the old broken unconditional-v1 request rather than
executing autosave. The original is preserved; the authorized copy executes the
actual JS `save()` to generate its envelope, preserving every original assertion
AST, including successful historical save. Its obsolete hardcoded request remains
invalid, not a claimed product PASS or FAIL. Synthetic tests additionally cover
legacy edit/autosave, explicit revision-bound transition, raw-history preservation,
reconfirmation with changed source, transition→adjudicate→freeze, negative-only
schema protection and residual marker rejection. Full browser/PostgreSQL/human
validation and U5b–d/HU059 remain open.

Effective-unavailable evidence now checks the intersection of **actually observed
columns and trained column counts**, not whether some different column trained.
Acquired measurements can therefore be present while effective input is absent
with `no_trained_measurements`; quality stays null/quality_not_assessed. Accepted
U4a/U4b masks, targets and eligibility are unchanged.

Final bounded writer evidence: 47 focused + 9 supplied probes + 548 full-suite
passes, all process/tool exits 0. The full suite has nine known warnings; it is
not an independent review, browser/PostgreSQL verification or scientific result.
Only docs/evidence changed afterward. Await fresh U5a fix review.

Four additional actual-UI residuals were reproduced independently after the first
fix batch. Writer focused evidence now includes current/legacy mutations, IDs
missing/existing, parent/child/original preservation, foreign/stale/forged lineage,
reference refresh and actual A/B→adjudication→freeze. Original reviewer probes and
assertions remain unchanged; U5a still requires fresh fix review.

Final writer evidence for the four UI residuals: **70 focused, 14 supplied probes,
571 full-suite passes**, all process/tool exits 0. The full suite emitted nine
known warnings. No source changes after that full run; docs/evidence only. This
is local synthetic verification, not independent acceptance; await fresh review.

Final local Undo evidence: **90 focused, all 16 supplied probes, 591 full-suite
passes**, process/tool exits 0. Full runtime 77.51 s with nine known warnings.
Only docs/evidence changed after that full run. This is not independent
acceptance, human/browser/PostgreSQL/scientific validation or actual-Silero proof;
stop for fresh Undo review before U5b–d/HU059.
