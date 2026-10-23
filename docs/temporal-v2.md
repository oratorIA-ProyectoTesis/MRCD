# Separate decision ownership from temporal evidence

**U4a provides pure contracts, not a new trained engine.** `core/temporal.py` defines
the registered `cell500-centered3000-v2` grid; `core/targets.py` projects original
human intervals into `mrcd-target-projection-v2` masked arrays. Legacy extractors,
NPZ files, checkpoints, inference and export adapters are unchanged. Decoder
integration is deferred to U4c. An additive offline feature/model/masked-training
path now exists in [U4b](cell-model-v2.md), awaiting its independent review; legacy
adapters remain unchanged and no human-trained v2 capability is claimed.

## Review path

1. Inspect exact ownership and evidence bounds in `core/temporal.py`.
2. Inspect explicit human evidence and per-head masks in `core/targets.py`.
3. Check numerical oracles in `tests/test_temporal.py`; operational commands live
   only in the canonical `fuente-informacion/comandos-ejecutar/COMANDOS.MD`.

## Registered grid

Actual recording duration `D` is an authoritative nonnegative integer in
milliseconds, **not** an extracted-feature row count. For index `i`:

| Field | Half-open bounds / meaning |
| --- | --- |
| Owned decision cell | `[500*i, min(D, 500*(i+1)))` |
| Nominal center | `500*i + 250`, even on a partial tail |
| Requested evidence | `[500*i-1250, 500*i+1750)`; may exceed real media |
| Observed support | Request intersected with `[0,D)` |
| Left/right padding | Requested support outside actual media; not observed evidence |

`decision_cells(D)` covers `[0,D)` exactly, including media shorter than 500 ms and
the final partial cell. `D=0` returns an explicit empty grid. Each cell records
`temporal_contract`, `media_duration_ms`, `index`, owned bounds and nominal center.
`evidence_windows(cells)` returns copies with requested/observed bounds and padding.
`validate_temporal_grid(cells,D)` requires exact order, adjacency, full cover and
matching duration/configuration; if evidence fields exist they must be complete
and exact. Other stride/context configurations require another registered version
and are rejected here. Nothing imposes a 13-second minimum.

Nominal/requested bounds may extend beyond media; owned/observed bounds never do.
Observed support describes time, not whether a feature is available or reliable.

## Human interval projection

`project_targets(D, events, reviewed_coverage, cells=None)` accepts original
millisecond intervals (`0 <= start_ms < end_ms <= D`) and reviewed coverage pairs.
Coverage is union-normalized; malformed/out-of-bounds input fails, not clipped.
An event activates every owned cell with **any positive half-open overlap**.
Touching an endpoint is not overlap. Evidence-window membership never defines a
label's ownership. Inputs are not mutated; raw events, IDs and notes are copied.

| Human field | Meaning |
| --- | --- |
| `label` | Canonical phenomenology type, or null for unknown type |
| `type_known: true` | Explicit assertion of genuinely known human type; a proposed label alone is not positive evidence |
| `decision` | `event` (default), `uncertain`, or `not_evaluable`; presence of an event is not a contextual judgment |
| `is_disfluent` | Explicit boolean contextual judgment (`false` functional, `true` disfluent); null/missing is unknown |

Known type and known context are independent: `decision=uncertain` with an
explicit known type retains that type but does not invent context. A context is
eligible only when explicitly boolean on an `event` decision. `not_evaluable`
cannot assert a known type. The future human adapter must establish `type_known`
from actual annotation semantics, never from model proposals or label presence.
This pure validator certifies consistency, not annotator identity or human truth.

## Array contract

`N` is the number of owned cells, `C=7` is canonical taxonomy order. Boolean arrays
are NumPy boolean; targets are signed integers with `-1` as **masked**, never a
class. The projector emits `int64` softmax and `int8` contextual targets; validation
accepts compatible integer dtypes only when all values/masks match raw evidence.

| Field / shape | Meaning |
| --- | --- |
| `multi_hot [N,C]` | Every explicitly known type positive, including functional uses and overlapping distinct phenomena |
| `fully_reviewed [N]` | Entire owned cell covered by reviewed union, with no unknown/proposed-type or not-evaluable overlap; context-only uncertainty does not erase known type coverage |
| `type_mask [N,C]` | Known positives remain eligible; type negatives require `fully_reviewed` |
| `type_ambiguity [N]` | Multiple known types, or any unknown/proposed-type overlap |
| `softmax_target/mask [N]` | Optional single-class view: 0 fluent only in fully reviewed negative cells; 1..C a single known type; otherwise -1/masked |
| `context_multi_hot [N,C,2]` | Explicit functional/disfluent evidence retained, including conflicts and partial cells |
| `context_conflict [N,C]` | Both context states observed for the same type in a cell |
| `context_target/mask [N,C]` | 0 functional / 1 disfluent only for present types in fully reviewed cells, exactly one context state, no same-type unknown-context mark; otherwise -1/masked |
| `event_indices [N]` | Ordered source-event indices overlapping each owned cell, including unknown marks |

Metadata includes both contract IDs, taxonomy version/label orders, actual duration,
complete evidence cells, original `source_events`, normalized `reviewed_coverage`,
and `requires_masked_training=True`. `validate_targets` reprojects retained inputs
to reject shape/dtype/value/mask inconsistencies, invented negatives, missing
evidence fields and wrong versions. A bare owned grid is valid projector input,
not a complete projected output. It does not open files or convert legacy arrays.

**Training must enforce all masks** in class counts, loss reduction, metrics,
and checkpoint eligibility. The additive `core/training_v2.py` now does this with
masked multilabel and contextual heads, pending U4b review. A zero array under a
false mask is unknown, not a negative. Current legacy training does not consume
this contract and is not declared compatible. Multiple phenomena never become one dominant softmax class.
Functional filler remains filler, not fluent; opposite contextual marks mask the
context head while preserving both marks and the known type.

## Quantization and remaining work

- `[490,510)` projects into `[0,500)` and `[500,1000)`. The original 20 ms remains in
  `source_events`; an oracle roundtrip can recover **cell projection**, not arbitrary
  original timestamps. A later decoder must preserve cell identity, actual duration,
  and honest unrefined bounds. No decoder is implemented in U4a.
- `[2400,2600)` activates only `[2000,2500)` and `[2500,3000)`, never an earlier
  cell merely because its 3 s evidence overlaps the event.
- Known filler in a partially reviewed cell remains a positive binary type target,
  but other types are unknown, softmax is masked, and the conservative context head
  is masked. Retained raw/context evidence is not silently discarded.
- U4b defines feature-specific real support/availability and exact 10 s linguistic
  membership/truncation in [its guide](cell-model-v2.md); a frame at exactly `D` is
  outside media. Audio/visual padding or interpolation is not actual measurement.
- Live v2 is not supported until centered lookahead and tail flush are implemented
  and verified. This slice adds no streaming requirement or engine support claim.
- The additive v2 loader fails explicitly if compatible newly trained weights are
  absent; product integration remains U4c. No random fallback or relabeling old
  weights/data as v2 is acceptable.
- HU059 experimental word/silence boundary refinement is separate and **open**, with
  honest unrefined fallback required. This projection does not fulfill HU059.

See [the full closure register](tp2-closure-plan.md) for U4b/U4c and U5–U9 gates.
