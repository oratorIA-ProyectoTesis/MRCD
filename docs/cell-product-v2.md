# Analyze whole recordings with explicit cell decisions

**U4c is awaiting fresh fix review.** The additive offline product adapter now
uses the reviewed U4a grid/targets and U4b feature/checkpoint contracts. Local
synthetic tests exercise HTTP queue → actual worker dispatch → a tiny newly
trained temporary checkpoint → strict JSON result. This is software evidence,
not real-human accuracy, calibration, visual benefit or production acceptance.

## Read and use safely

1. Review `core/cell_engine.py` and `core/cell_decode.py`, then
   `app/systems.py` and `app/worker.py`. The synthetic integration/oracles are in
   `tests/test_cells_product.py`.
2. An analysis request explicitly persists `config.engine_kind`:
   `legacy_window_v1` (declared compatibility default) or `cell_v2`.
   V2 requires an explicit, existing, safely loadable **trained v2** checkpoint;
   it never guesses from dimensions or falls back to legacy/random weights.
3. Use full JSON or the safe Markdown report. Reference-relative evaluation,
   system agreement, assisted-review projection, CSV and EAF for new cell/window
   results fail clearly until their task-aware U5 adapters exist. A lossless JSON
   roundtrip is not proof of CSV/EAF interchange or a usable browser workflow.
4. Operational commands have one source: the U4c saved isolation wrapper in
   `fuente-informacion/comandos-ejecutar/COMANDOS.MD`. No new ASR/model download,
   service or real-data processing was run for this slice.

## One global clock and ownership ledger

| Concept | Implementation |
| --- | --- |
| Authoritative duration | Whole analysis WAV `samples / sample_rate`, checked against integer `duration_ms` and `PreparedRecording.duration_s`; 16 kHz mono product input. Nonzero origins are rejected, not offset twice. |
| Reused input | Full `PreparedRecording` acoustic timestamps/frame matrix/raw F0 and full ASR words. Raw tracks/words use seconds; cells/cores use integer milliseconds. Frame count never substitutes for duration. |
| Global features | One `decision_cells(D)` + centered evidence grid and one `build_cell_features` call. Execution batches select existing global rows, not zero-origin clip re-extractions. |
| Execution ownership | Ordered unique cores must partition `[0,D)` exactly. A cell belongs to the core containing the midpoint of its **actual owned** interval, including the partial tail. |
| Visual input | Empty aligned arrays with `product_visual_acquisition_not_integrated`. Research visual weights/metadata do not certify product video. Audio/text fallback remains usable; U6 supplies real visual acquisition later. |
| Errors | Invalid duration/core/model/output contracts abort. A `RuntimeError` from a validated inference batch records failed cells for that batch, preserving completed cells. Cancellation callbacks run outside that recovery block and propagate to the worker. |

The result schema is `mrcd-cell-result-v2`. It retains every global cell in order,
with unique recording-scoped ID/index, execution segment, owned/requested/observed
bounds, actual support references, status, input availability, per-type decisions,
per-context probabilities/states/reasons and model capabilities. Unsupported heads
have **JSON null plus a reason**, never zero or JSON NaN. Successful inference and
input eligibility are separate: a completed but unobserved cell is not evaluable.

The exclusive `run_coverage` adapter checks complete grid/identity/status,
capabilities, scores/policy, core assignment, decoded events and claimed coverage
against the ledger. Unknown/missing version markers cannot use legacy core
availability: retained ledger, temporal/window contracts and tagged events are
recognized even if both top-level tags are missing/empty/unknown. Only genuinely
unmarked historical payloads retain core-based compatibility. Coverage is the union of **successful input-evaluable owned cells**;
neither threshold abstention nor missing contextual supervision creates a human
exclusion hole. This is execution/input availability, not per-task accuracy or
reference coverage. Task-specific scoring remains blocked until U5.

**Numeric oracle:** D=6250 with cores `[0,2300)`, `[2300,4600)`, `[4600,6250)` assigns
13 cells as 5/4/4. A failed middle execution batch leaves `[0,2500) ∪ [4500,6250)`
=4250 ms, not the old cores' union. The tail owns `[6000,6250)`, nominal center
6250, request `[4750,7750)`, observed `[4750,6250)`. D=0 is an explicit empty
ledger/result; there is no arbitrary 13-second minimum on the v2 engine contract.

## Shared decoder: type is not contextual disfluency

Each type has an independent multilabel trajectory. Exact adjacent cells merge
only with compatible type decision, contextual state/reason, quality and
capabilities. Negative, failed, non-evaluable or incompatible cells break a group;
there is no bridging across gaps. Decoding happens after all batches, so healthy
output does not depend on batch size or an execution-core boundary.

- `type_decision`: positive, negative, abstained or unavailable (complete cell
  ledger); groups emit positive/abstained trajectories, not negative evidence.
- `contextual_state`: disfluent, functional, uncertain or null. A positive type
  with unavailable context is **not** a confirmed disfluency. Functional use keeps
  its phenomenon type and has `is_disfluent=false`; unknown/abstained context or
  type gives null, never an invented boolean.
- `cell_ids`, raw cell bounds, support refs and execution segment IDs remain
  traceable. Group boundaries have `boundary_resolution_ms=500` and
  `boundary_source=owned_cells_unrefined`.
- Scores are `uncalibrated_sigmoid_conditional_softmax`; the explicit
  `cell-threshold-v1` policy has `calibrated=false` (default positive .6,
  negative .4, context .6). Fitted calibration requires separate dev evidence U8.

An oracle decoder recovers the **cell projection**, not arbitrary raw timestamps:
human `[490,510)` projects to `[0,1000)`. The 20 ms source event is not recovered
exactly and no boundary-accuracy gain is claimed. HU059 word/silence refinement is
not implemented; source cells/unrefined bounds are retained for that later work.
Live v2 rejects until centered lookahead and tail flush exist; streaming is not a
requirement of this offline slice.

## Stable checkpoint identity and honest legacy behavior

The product cache is keyed by checkpoint content SHA-256, configuration and
registered versions, not path alone. A second hash check rejects a checkpoint
changed while loading. Results report the hash bound to loaded weights rather
than rehashing a path after analysis and announcing different bytes. Replacing a
file at the same path therefore cannot reuse its former cached weights under a
new identity. Actual promotion/registry selection by **new** analyses remains
HU060/U8/U9, not completed by this cache change.

`core/legacy_windows.py` is shared by the legacy engine and `scripts/infer_video.py`.
New v1 outputs explicitly say `prediction_kind=window_classification` and
`window_contract=legacy-forward3000-stride500-v1`. A start at 0/500 ms supports
`[0,3000)`/`[500,3500)`, capped by actual duration; it is not a precise 500 ms
event. Groups retain original label order/probabilities and each window's support.
Raw logits are unavailable because legacy `predict` exposes only probabilities;
they are null with a reason, not reconstructed as purported original logits.
Nondefault legacy temperature is rescaling only; its historical score-kind name
does not establish recorded dev fitting or calibrated confidence.

The offline video research script still requires its verified legacy checkpoint
contract and uses this shared v1 decoder. Its visual path does not certify product
video. Its CLI and new-contract QA report call outputs **candidate support groups**,
never confirmed detections or event rates. The QA route reports unavailable
context and does not consume/generate historical judge queues. Unsupported or
inconsistent new contracts reject before the historical report path; genuinely
unmarked historical reports remain compatible without rewriting input artifacts.
Legacy training/weights/NPZ, historical results and frozen snapshots are
not rewritten or relabeled. The historical pure `core.engine.decode` helper is
kept for compatibility, but is no longer the new legacy product output route.
Legacy word/silence event refinement is rejected rather than marking support
intervals as improved events.

## Review limits and next gate

- UI minimum guards show quantized bounds, type/context/unavailability, unsupported
  heads, input coverage and legacy support. They do not count all non-pause types
  as disfluencies or fabricate rates. Summary and uncertainty filter share one
  predicate: contextual uncertainty remains visible even with positive type and
  `decision=event`; absent context remains separately unavailable. Browser
  interaction has not been verified.
- U5 must implement complete task-aware API/JSON/CSV/EAF and human-snapshot
  adapters/evaluation; U6 must acquire actual synchronized video; U7/U8/U9 retain
  governance, promoted-model selection, scientific and operational evidence gates.
- No human-trained checkpoint, human/EAF acquisition, accuracy, calibrated
  probabilities, visual gain, PostgreSQL concurrency, production or performance
  acceptance is claimed. Missing human work remains explicitly deferred.
