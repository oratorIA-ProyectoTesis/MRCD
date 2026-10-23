# Train additive v2 cells without relabeling legacy artifacts

**U4b is an additive, offline feature/model/training path.** Explicit local raw
signals plus original interval evidence can run through build → masked training →
new checkpoint → strict safe load → per-cell probabilities. Synthetic tests prove
software behavior, not detection accuracy, human validity, calibration or visual
benefit. The [U4c product ledger/decoder adapter](cell-product-v2.md) now connects
this strict loader through the worker/API with synthetic fixtures and is awaiting
independent review. Task-aware interchange/evaluation, human-snapshot adapters and
legacy ablations remain U5/U8 work; this does not certify real-human weights.

## Review and use

1. Review `core/features_v2.py` and `core/cell_contracts.py` against the accepted
   [temporal and human-target contract](temporal-v2.md).
2. Review `core/models/cell_fusion.py`, `core/cell_normalization.py` and
   `core/training_v2.py` for effective mask/position/ownership use.
3. Review `core/cell_dataset.py`, `core/cell_inference.py` and the synthetic
   build/train/load/predict and subprocess CLI tests in `tests/test_cells_v2.py`.
4. Operational commands and complete isolated test environments are maintained
   only in `fuente-informacion/comandos-ejecutar/COMANDOS.MD`. Build/train CLI inputs
   and new output paths are mandatory; existing output paths are refused.

No ASR, visual extractor, model download, provider call, server or human-data
collection is invoked by these builders. Permission and canonical split governance
remain U7; this API does not invent those records or verify that an input asserting
human provenance is truthful.

## Explicit inputs and artifacts

An input JSON manifest identifies `source_kind` (`synthetic` or `human`) and a
nonempty `recordings` list. Each recording requires:

| Field | Contract |
| --- | --- |
| `recording_id`, `speaker_id`, `source_identity` | Explicit nonempty identities; duplicate recording IDs rejected |
| `split` | Persisted `train`, `dev` or `test`; only train rows fit statistics or training |
| `duration_ms` | Actual nonnegative integer recording duration, never inferred from frame count |
| `raw_npz`, `words_json`, `events_json` | Explicit local paths, resolved relative to the input manifest |
| `reviewed_coverage` | Original reviewed millisecond intervals for U4a projection |

Raw NPZ must contain `ac_times` (seconds), `ac_frames [T,5]`, raw voiced `f0 [T]`
(Hz, unavailable/nonvoiced not positive), `k_times` (seconds), `k_vectors [K,12]`
and boolean `k_detected [K]`. Tracks must be aligned and strictly increasing in
time. Missing tracks use explicit empty aligned arrays; timestamps outside real
media, including a frame at exactly D, do not become evidence or pitch references.
The acoustic frame columns provide log RMS, legacy pitch, voiced probability, VAD
and legacy pitch delta; v2 **recomputes pitch and delta from raw F0**, not zero
sentinels or the legacy columns. This is not conversion of aggregated window y.
Optional old word embeddings/feature widths are not concatenated or cropped.

Word JSON retains actual text, interval and identity. It accepts millisecond
`start_ms/end_ms`, or persisted ASR second `start/end`; probability may be unknown.
Event JSON uses the explicit U4a human-type/context semantics, not heuristic labels
silently promoted to truth. Synthetic source kind remains synthetic throughout.

`build_dataset` writes a **new** directory:
- `arrays.npz`: numeric/boolean feature arrays and `target_` arrays; no object
  arrays or pickle loading.
- `manifest.json`: exact dataset/feature contracts, source kind/identities/splits,
  current input SHA-256 hashes, words, original events/reviewed coverage, U4a
  target metadata, row offsets and NPZ hash. File hashes are distinct from
  `retained_input_sha256`: canonical JSON bindings for retained words, original
  events and reviewed coverage. These hashes identify current inputs and check
  self-consistency, not historical media verification, permission or an external
  authenticity signature.

Loading validates hash identifiers/source schema and row alignment, replays the
deterministic word selection, positions, identity and observed lightweight features,
and revalidates U4a targets against **outer** original reviewed coverage as well as
retained target metadata. Active word IDs must be in range and unique per cell;
bounds, center-relative positions, context membership and owned overlap determine
`ltarget`, never the other way around. Neutral masked storage is not an observation.
Original raw/JSON/manifest files need not remain present: NPZ+sidecar are portable.
Someone coherently rewriting content and all its hashes is not detected as an
external-authenticity violation; independent governance remains necessary.
Loading does not mutate
old NPZ, frozen snapshots, original raw data or model files. Speaker alias/leakage
governance and retained-test experimental protocol remain U7/U8.

## Features and real availability

All positions are actual seconds relative to the nominal cell center. The feature
contract stores ordered fields, shapes, temporal/target/taxonomy versions, mask,
position and truncation policies. Matching dimensions alone are insufficient.

| Arrays | Semantics |
| --- | --- |
| `ac [N,300,5]`, `ac_mask` | Real intersecting 10 ms samples copied into centered 3 s evidence; per-measure availability. Raw positive finite F0 defines voiced pitch: 100 Hz → **observed 0 semitones**. Delta needs adjacent voiced samples within the same requested evidence. |
| `ac_support`, `ac_target`, `ac_pos [N,300]` | Actual sample time, owned-cell membership and position, distinct from measured-column availability. No invented samples/interpolation/padding VAD. |
| `pros [N,8]`, `pros_mask` | Silence selected by greatest positive overlap with **owned** cell, then earliest start. Not longest silence somewhere in context. `reset_valid` becomes the reset mask, not a ninth duplicated feature. |
| `ling [N,40,12]`, `ling_mask` | Ten lightweight features computed on the selected context **before truncation**, plus owned-zone and function-word/dangling-gap indicator. Neighbor gaps/repetition are unknown at context edges, not observed zeros. Probability unavailable is masked. |
| `lmask`, `lpos`, `ltarget`, `word_indices`, `word_intervals`, `word_source_count` | Retained tokens, position, owned overlap, original identity/index, intervals and original word-list size; active identities/geometry are validated. |
| `kin [N,30,12]`, `kin_mask`, `kin_support`, `kpos`, `ktarget` | Actual detected frames only, with per-column finite masks; no interpolation or extrapolation. At most one persisted real frame per 100 ms evidence slot. |
| `real_frame_count`, `has_video` | At least 24 actual detected frames in observed evidence to enable visual use; sparse video does not invalidate audio. Detected frames and measured columns remain distinct. |
| `media_duration_ms`, owned bounds, nominal center | Explicit per-row authoritative temporal metadata, validated against support/ownership. |

Prosody uses only real samples in observed evidence. Exact silence duration requires
observed speech boundaries on **both** sides; censored silence is unavailable, not
an exact shortened pause. Observed absence of silence supplies duration 0 only
when owned VAD support is complete; absent pause position remains unavailable.
Pitch/RMS slopes require at least three actual points, reset requires at least
three voiced points on both sides, and the recording pitch reference excludes
nonvoiced/out-of-media frames. `final_lengthen` is an energy-based proxy, not a
measured syllable length or a contextual judgment.

Boundary-relative features additionally require their **actual anchor**: pitch
slope/range/level, pre-onset RMS slope and lengthening need an observed onset;
reset needs both observed onset and end. A censored observed-run edge is not an
actual boundary. An unknown end masks duration/reset but leaves valid onset-based
measurements available; this is not a blanket all-features exclusion.

### Exact offline linguistic membership

For nominal center `c`, select word centers in `[c-5000,c+5000)` milliseconds **plus
every word positively overlapping owned bounds**. Compute lightweight neighbor
features on that context, without outside neighbors. Reserve all owned words,
then fill remaining capacity by distance to c, start, end and original index;
restore chronological order. More than 40 owned words raises
`target_word_overflow`, never silently drops targets. The dangling feature means
function word followed by a known gap of at least 250 ms; without a next context
word it is unavailable. It does not infer a disfluency.

## Effective masked model and training

`CellFusionV2` embeds sanitized values **and per-column masks**, actual positions,
modality and owned/context membership. The attention query explicitly summarizes
owned acoustic observations, not a whole-context average. It attends over acoustic,
word, visual and prosodic tokens with key-padding masks. A null token is enabled
for all-empty evidence, avoiding NaN attention. Missing values are selected away
**before** subtraction/projection; `NaN*0` is never the masking policy.

`input_evaluable` requires measured **owned** acoustic samples, owned words in an
enabled linguistic mode, or measured owned real frames in an enabled/gated visual
mode. Context-only evidence and prosody alone cannot invent owned eligibility.
Modes are audio-only, audio-text and trimodal; absent/unsupported visual evidence
falls back to actually available audio/text. This is not live support.

| Supervision | Effective behavior |
| --- | --- |
| Primary types | `type_logits [N,7]`, masked binary cross-entropy; simultaneous phenomena retained |
| Context per type | `context_logits [N,7,2]`, eligible contextual cross-entropy only |
| Optional U4a softmax view | Retained in dataset for compatibility; not the model's primary head |
| Statistics | Train rows with useful supervision and owned input only; observed per-column entries only; count 0 → neutral mean 0/std 1 and unavailable |
| Loss/counts/metrics | Masked labels do not contribute; contextual conflicts remain masked; training-only metrics explicitly labeled |
| All-masked minibatch | Skip `optimizer.step()` entirely; zero loss is insufficient because AdamW decay can mutate weights |
| All-masked dataset | Reject before checkpoint publication; no claimed trained model |

The training report records eligible positive/negative/state counts, useful rows,
actual optimizer/head steps, skipped batches and observed modality support. A
single contextual state (or no state) does **not** establish contextual
discrimination. Likewise each type needs both positive and negative supervision
to be reported as discriminatively supported. Instantiated parameters or mode
metadata are not proof of trained visual/contextual capability.
The loader enforces positive+negative counts per type no greater than useful rows,
context counts no greater than known positive type evidence, real head-step
support, and optimizer/skipped totals consistent with epochs/rows/batch size.
Disabled modalities cannot have observed statistics or claimed support, even when
both mode and provenance are edited together. Contextual outputs require a
supported type head as well as a discriminatively supported contextual head.

## Checkpoint and inference boundary

Checkpoints are newly named, atomically published without overwrite, using plain
metadata and tensors compatible with `torch.load(weights_only=True)`. They include
feature/temporal/target/taxonomy/model implementation versions, dimensions, mode,
seed, normalization counts, actual supervision/capabilities, current code hashes,
input/dataset identities and source kind. This is a safe-load boundary, not a
license to load arbitrary untrusted files or evidence of model accuracy.

`CellInference.from_checkpoint` rejects missing/legacy/incompatible/malformed
files, invalid normalization, unsupported capability claims, nonfinite weights
and provenance/mode inconsistencies. It never retries unsafe loading, random
initialization or a legacy model. Inference disables modalities lacking observed
training support, irrespective of a trimodal model instance. Output type sigmoid
and conditional context softmax probabilities are labeled **uncalibrated**.
Unsupported heads and no-owned-input rows return NaN arrays plus explicit support
flags/reasons, not fake probabilities of zero. This internal numerical boundary
is not yet a JSON/API product contract; that remains U4c/U5.

`D=0` is valid inference input: type/context outputs have shapes `[0,7]` and
`[0,7,2]`, the availability/reason lists are empty, and no attention reshape is
attempted. Empty training still rejects without writing a checkpoint. Pre-fix U4b
artifacts missing the strengthened feature/input-binding/support contract are
rejected; rebuild at a new path rather than relabeling or overwriting them.

## Still open

- U4c: shared product/CLI/export loaders, original-cell decoder, actual-duration
  bounds and explicit live rejection at product entry points. No decoder here.
- HU059 experimental word/silence refinement, with honest unrefined fallback.
- U7/U8: authorized human data, canonical splits/leakage guards, dev calibration,
  B0–B4/ASR protocols and genuine compatible human-trained weights. Legacy
  `run_ablation.py` stays legacy; no weights are relabeled as retrained v2.
- U9: browser/product roundtrip, real performance and operational verification.

See [the closure ledger](tp2-closure-plan.md) for scoped test/review evidence.
