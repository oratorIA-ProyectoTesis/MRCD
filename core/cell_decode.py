"""Recording-global, quantized cell decisions. No temporal refinement is implied."""
from __future__ import annotations

import json
import math

from core.constants import TAXONOMY
from core.temporal import validate_temporal_grid, integer_ms, TEMPORAL_CONTRACT

CELL_RESULT_SCHEMA = "mrcd-cell-result-v2"
CELL_EVENT_SCHEMA = "mrcd-cell-event-v2"
SCORE_KIND = "uncalibrated_sigmoid_conditional_softmax"
DEFAULT_THRESHOLDS = dict(version="cell-threshold-v1", positive=.6, negative=.4, context=.6, calibrated=False)


def threshold_policy(policy=None):
    policy = dict(DEFAULT_THRESHOLDS if policy is None else policy)
    if (set(policy) != set(DEFAULT_THRESHOLDS) or policy["version"] != "cell-threshold-v1"
            or policy["calibrated"] is not False):
        raise ValueError("unsupported cell threshold policy; calibration is not established")
    for key in ("positive", "negative", "context"):
        if isinstance(policy[key], bool) or not isinstance(policy[key], (float, int)) or not math.isfinite(policy[key]):
            raise ValueError("thresholds must be finite probabilities")
    if not 0 <= policy["negative"] < policy["positive"] <= 1 or not 0 <= policy["context"] <= 1:
        raise ValueError("invalid cell threshold bounds")
    return policy


def partition(cells, segments, duration_ms):
    """Validate execution cores, returning global indices owned by each core."""
    validate_temporal_grid(cells, duration_ms)
    ids, cursor, assignments = set(), 0, {}
    for segment in segments:
        sid = segment.get("id")
        a = integer_ms(segment.get("core_start_ms"), "core_start_ms")
        b = integer_ms(segment.get("core_end_ms"), "core_end_ms")
        if not isinstance(sid, str) or not sid or sid in ids or a != cursor or not a < b <= duration_ms:
            raise ValueError("execution cores must uniquely partition actual media, in order")
        ids.add(sid)
        assignments[sid] = [c["index"] for c in cells if 2 * a <= c["owned_start_ms"] + c["owned_end_ms"] < 2 * b]
        cursor = b
    if cursor != duration_ms:
        raise ValueError("execution cores do not cover actual media")
    return assignments


def ledger_coverage(cells):
    spans = []
    for cell in cells:
        if cell["status"] != "succeeded" or not cell["input_evaluable"]:
            continue
        a, b = cell["owned_start_ms"], cell["owned_end_ms"]
        if spans and a == spans[-1][1]:
            spans[-1][1] = b
        else:
            spans.append([a, b])
    return spans


def require_legacy_consumer(result):
    """Fail closed until U5 supplies explicit task-aware consumer contracts."""
    try:
        family = result_family(result)
    except ValueError as exc:
        raise ValueError(f"task-aware consumer rejected invalid result contract: {exc}") from exc
    if family != "historical":
        raise ValueError("task-aware consumer not integrated (U5): cell type/context or legacy window support is not a human disfluency event")


def result_family(result):
    """Classify declared contracts, refusing recognizable downgrade payloads.

    Genuine historical output has no cell/window contract markers. A missing or
    corrupt top-level tag cannot erase a retained ledger, contract or event tag.
    Research artifacts may group their events by mode rather than in one list.
    This checks the dispatch boundary; cell validation still checks the ledger.
    """
    if not isinstance(result, dict):
        raise ValueError("result must be a mapping")
    events = result.get("events", [])
    groups = list(events.values()) if isinstance(events, dict) else [events]
    tagged_events = [event for group in groups if isinstance(group, list) for event in group if isinstance(event, dict)]
    cell = (any(key in result for key in ("result_schema", "cells", "execution_cores", "temporal_contract"))
            or result.get("prediction_kind") == "cell_type_context"
            or any(any(key in event for key in ("cell_ids", "raw_cell_bounds", "event_schema"))
                   or event.get("prediction_kind") == "cell_type_context" for event in tagged_events))
    window = ("window_contract" in result or result.get("prediction_kind") == "window_classification"
              or any("window_contract" in event or event.get("prediction_kind") == "window_classification" for event in tagged_events))
    if cell:
        if window or result.get("result_schema") != CELL_RESULT_SCHEMA or result.get("prediction_kind") != "cell_type_context":
            raise ValueError("recognizable cell payload requires consistent supported top-level schema/kind; no historical fallback")
        return "cell_v2"
    if window:
        from core.legacy_windows import LEGACY_WINDOW_CONTRACT

        if result.get("prediction_kind") != "window_classification" or result.get("window_contract") != LEGACY_WINDOW_CONTRACT:
            raise ValueError("recognizable window payload requires supported kind/window_contract; no historical fallback")
        return "legacy_window_v1"
    if "prediction_kind" in result:
        raise ValueError("unknown or empty prediction_kind is not a historical contract")
    return "historical"


def validate_cell_result(result):
    if result.get("result_schema") != CELL_RESULT_SCHEMA:
        raise ValueError("unsupported cell result schema")
    from core.cell_contracts import cell_feature_contract

    if (result.get("temporal_contract") != TEMPORAL_CONTRACT or result.get("feature_contract") != cell_feature_contract()
            or result.get("prediction_kind") != "cell_type_context" or result.get("score_kind") != SCORE_KIND):
        raise ValueError("cell result implementation/feature/score contract mismatch")
    cells, duration, rid = result.get("cells"), result.get("duration_ms"), result.get("recording_id")
    if not isinstance(rid, str) or not rid:
        raise ValueError("cell ledger recording identity required")
    n = validate_temporal_grid(cells, duration)
    assignments = partition(cells, result.get("execution_cores", []), duration)
    owners = {index: sid for sid, indices in assignments.items() for index in indices}
    policy = threshold_policy(result.get("threshold_policy"))
    capabilities = result.get("capabilities", {})
    for key in ("type_discrimination", "context_discrimination"):
        values = capabilities.get(key)
        if not isinstance(values, list) or len(values) != 7 or any(type(v) is not bool for v in values):
            raise ValueError("cell capabilities must declare each trained head")
    if any(c and not t for t, c in zip(capabilities["type_discrimination"], capabilities["context_discrimination"])):
        raise ValueError("contextual discrimination requires type support")
    for cell in cells:
        i = cell["index"]
        if (cell.get("cell_id") != f"{rid}:cell:{i}" or cell.get("recording_id") != rid
                or cell.get("segment_id") != owners[i] or cell.get("status") not in ("succeeded", "failed")
                or type(cell.get("input_evaluable")) is not bool or type(cell.get("available")) is not bool
                or cell["available"] != (cell["status"] == "succeeded" and cell["input_evaluable"])):
            raise ValueError(f"cell {i}: invalid identity/status/effective availability")
        if cell.get("capabilities") != capabilities:
            raise ValueError("per-cell capabilities disagree with loaded model")
        if "evidence" in cell:
            from core.cell_evidence import EVIDENCE_SCHEMA
            if cell["evidence"].get("schema") != EVIDENCE_SCHEMA or cell.get("quality") != cell["evidence"].get("quality"):
                raise ValueError("cell acquisition/effective-input/quality contract mismatch")
        support = cell.get("support_refs", {})
        if support.get("evidence_start_ms") != cell["observed_start_ms"] or support.get("evidence_end_ms") != cell["observed_end_ms"]:
            raise ValueError("observable support and real evidence bounds disagree")
        if cell["status"] == "failed" and (cell["input_evaluable"] or not cell.get("failure_reason")):
            raise ValueError(f"cell {i}: failed input cannot be available")
        for key in ("type_probabilities", "context_probabilities", "type_decisions", "contextual_states", "context_reasons", "type_reasons"):
            if not isinstance(cell.get(key), list) or len(cell[key]) != len(TAXONOMY):
                raise ValueError(f"cell {i}: invalid {key}")
        for k in range(len(TAXONOMY)):
            p, cp = cell["type_probabilities"][k], cell["context_probabilities"][k]
            decision, context = cell["type_decisions"][k], cell["contextual_states"][k]
            if p is not None and (isinstance(p, bool) or not isinstance(p, (int, float)) or not math.isfinite(p) or not 0 <= p <= 1):
                raise ValueError("invalid type probability")
            if not isinstance(cp, list) or len(cp) != 2:
                raise ValueError("invalid contextual probability shape")
            if any(v is None for v in cp):
                if cp != [None, None] or context is not None or not cell["context_reasons"][k]:
                    raise ValueError("context unavailable without reason")
            elif (any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not 0 <= v <= 1 for v in cp)
                  or not math.isclose(sum(cp), 1., abs_tol=1e-5)):
                raise ValueError("invalid contextual probabilities")
            expected = "unavailable" if p is None else "positive" if p >= policy["positive"] else "negative" if p <= policy["negative"] else "abstained"
            expected_context = None if cp == [None, None] else "uncertain" if max(cp) < policy["context"] else ("disfluent" if cp[1] >= cp[0] else "functional")
            if decision != expected or context != expected_context or (p is None and not cell["type_reasons"][k]):
                raise ValueError("cell decisions disagree with registered threshold policy")
            if not cell["available"] and (p is not None or cp != [None, None]):
                raise ValueError("unavailable cell exposes evaluated scores")
            if ((p is not None) != (cell["available"] and capabilities["type_discrimination"][k])
                    or (cp != [None, None]) != (cell["available"] and capabilities["context_discrimination"][k])):
                raise ValueError("probability availability disagrees with task/head support")
    if result.get("coverage") != ledger_coverage(cells):
        raise ValueError("claimed coverage disagrees with complete cell ledger")
    expected_status = {sid: ("succeeded" if all(cells[i]["status"] == "succeeded" for i in indices)
                             else "partial" if any(cells[i]["status"] == "succeeded" for i in indices) else "failed")
                       for sid, indices in assignments.items()}
    if result.get("segments") != expected_status:
        raise ValueError("segment aggregate disagrees with cell ledger")
    if result.get("events") != decode_cells(cells, recording_id=rid):
        raise ValueError("decoded events disagree with authoritative cell ledger")
    json.dumps(result, allow_nan=False)
    return n


def decode_cells(cells, *, recording_id):
    """Merge adjacent compatible trajectories, independently for each phenomenon.

    This decodes cell projection, not original timestamps. Failed, unavailable,
    negative or incompatible context/quality cells break a trajectory.
    """
    validate_temporal_grid(cells, cells[0]["media_duration_ms"] if cells else 0)
    for cell in cells:
        if cell.get("recording_id") != recording_id:
            raise ValueError("decoder recording identity mismatch")
        for key in ("type_decisions", "contextual_states", "context_reasons", "type_probabilities"):
            if not isinstance(cell.get(key), list) or len(cell[key]) != 7:
                raise ValueError("decoder requires aligned per-type arrays")
    events = []
    for k, label in enumerate(TAXONOMY):
        active, signature, previous = [], None, None

        def flush():
            if not active:
                return
            first, last = active[0], active[-1]
            context, decision = first["contextual_states"][k], first["type_decisions"][k]
            known = decision == "positive" and context in ("disfluent", "functional")
            events.append(dict(
                event_schema=CELL_EVENT_SCHEMA, prediction_kind="cell_type_context", recording_id=recording_id,
                label=label, type=label, start_ms=first["owned_start_ms"], end_ms=last["owned_end_ms"],
                type_decision=decision, contextual_state=context, is_disfluent=(context == "disfluent") if known else None,
                decision="event" if decision == "positive" else "uncertain", score=max(c["type_probabilities"][k] for c in active),
                score_kind=SCORE_KIND, calibrated=False, context_reason=first["context_reasons"][k],
                task_evaluable=dict(type=True, contextual=context is not None, disfluency=known),
                cell_ids=[c["cell_id"] for c in active], raw_cell_bounds=[[c["owned_start_ms"], c["owned_end_ms"]] for c in active],
                segment_ids=list(dict.fromkeys(c["segment_id"] for c in active)),
                boundary_resolution_ms=500, boundary_source="owned_cells_unrefined", refinement_reason="HU059_not_implemented",
                capabilities=first["capabilities"], quality=first["quality"],
                support_refs=[dict(cell_id=c["cell_id"], **c["support_refs"]) for c in active],
            ))
        for cell in cells:
            eligible = cell["available"] and cell["type_decisions"][k] in ("positive", "abstained")
            sig = (cell["type_decisions"][k], cell["contextual_states"][k], cell["context_reasons"][k],
                   json.dumps(cell["quality"], sort_keys=True), json.dumps(cell["capabilities"], sort_keys=True),
                   tuple((key, value["available"], value["owned_available"]) for key, value in sorted(cell.get("evidence", {}).get("effective", {}).items())))
            adjacent = previous is not None and previous["index"] + 1 == cell["index"] and previous["owned_end_ms"] == cell["owned_start_ms"]
            if active and (not eligible or sig != signature or not adjacent):
                flush()
                active = []
            if eligible:
                active.append(cell)
            signature, previous = sig, cell
        flush()
    events.sort(key=lambda e: (e["start_ms"], e["end_ms"], e["label"]))
    for i, event in enumerate(events):
        event["event_id"] = f"{recording_id}:cell-event:{i}"
    return events
