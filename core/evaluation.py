"""Event evaluation, human agreement, calibration and usability scoring.

Events are dicts with integer `start_ms`/`end_ms` ([start, end)), `label`, optional
`recording_id`, `speaker_id` and `decision` ("event" | "uncertain"). Protocol values
(IoU, tolerance, classes) are arguments fixed before evaluation, never tuned on test.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment, minimize_scalar
from sklearn.metrics import cohen_kappa_score

from core.constants import TAXONOMY, TAXONOMY_VERSION

PAUSES = ("rhetorical_pause", "neutral_pause")
DISFLUENCIES = tuple(c for c in TAXONOMY if c not in PAUSES)
DURATION_BINS_MS = (0, 300, 1000, 10**9)
PROTOCOL_VERSION = "mrcd-evaluation-v2"
BOUNDARY_POLICY = "fully_contained_v2"


def union_intervals(spans) -> list[list[int]]:
    if not isinstance(spans, (list, tuple)):
        raise ValueError("coverage: expected an explicit interval list")
    result = []
    for span in spans:
        if not isinstance(span, (list, tuple)) or len(span) != 2 or any(type(x) is not int for x in span) or not 0 <= span[0] < span[1]:
            raise ValueError("coverage: expected positive integer half-open intervals")
    for a, b in sorted(spans):
        if result and a <= result[-1][1]:
            result[-1][1] = max(result[-1][1], b)
        else:
            result.append([a, b])
    return result


def normalize_coverage(coverage: dict) -> dict:
    if not isinstance(coverage, dict):
        raise ValueError("coverage: expected recording-to-intervals mapping")
    return {rid: union_intervals(spans) for rid, spans in coverage.items()}


def coverage_duration(coverage: dict) -> int:
    return sum(b - a for spans in normalize_coverage(coverage).values() for a, b in spans)


def intersect_coverage(a: dict, b: dict) -> dict:
    a, b = normalize_coverage(a), normalize_coverage(b)
    return {rid: union_intervals([[max(x, u), min(y, v)] for x, y in spans for u, v in b.get(rid, [])
                                 if max(x, u) < min(y, v)]) for rid, spans in a.items()}


def subtract_coverage(coverage: dict, holes: dict) -> dict:
    result = {}
    holes = normalize_coverage(holes)
    for rid, spans in normalize_coverage(coverage).items():
        remaining = []
        for a, b in spans:
            cursor = a
            for x, y in holes.get(rid, []):
                if y <= cursor:
                    continue
                if x >= b:
                    break
                if x > cursor:
                    remaining.append([cursor, x])
                cursor = max(cursor, y)
                if cursor >= b:
                    break
            if cursor < b:
                remaining.append([cursor, b])
        result[rid] = union_intervals(remaining)
    return result


def reference_view(snapshot: dict, recording_speakers: dict) -> dict:
    """Pure adapter: frozen human coverage minus human unknowns, never model abstention."""
    items = snapshot.get("items")
    if not isinstance(items, list) or not items:
        raise ValueError("snapshot.items: reviewed regions are required")
    if any(item.get("semantic_schema") for item in items) or any(e.get("human_event_schema") for item in items for kind in ("events","contextual") for e in item.get(kind, [])):
        raise ValueError("task-aware human semantic evaluation not integrated (U5b)")
    schema = snapshot.get("snapshot_schema")
    if schema not in (None, "mrcd-snapshot-v2"):
        raise ValueError("snapshot_schema: unsupported reference adapter")
    if schema and snapshot.get("taxonomy_version") != TAXONOMY_VERSION:
        raise ValueError("taxonomy_version: incompatible snapshot reference")
    if schema and snapshot.get("reference_kind") not in ("human_adjudicated", "human_single_annotation"):
        raise ValueError("reference_kind: native snapshots must identify their human source kind")
    reviewed, holes, reference, legitimate, item_speakers = {}, {}, [], [], {}
    for item in items:
        rid = item["recording_id"]
        reviewed.setdefault(rid, []).append([item["start_ms"], item["end_ms"]])
        item_speakers.setdefault(rid, set()).add(item.get("speaker_id"))
        for kind in ("events", "contextual"):
            events = item.get(kind, [])
            if not isinstance(events, list):
                raise ValueError(f"snapshot.{kind}: expected explicit lists")
            for event in events:
                # Validate spans even if a human has explicitly excluded the region.
                union_intervals([[event["start_ms"], event["end_ms"]]])
                tagged = {**event, "recording_id": rid}
                if event.get("decision") in ("uncertain", "not_evaluable"):
                    holes.setdefault(rid, []).append([event["start_ms"], event["end_ms"]])
                elif event.get("is_disfluent") is False:
                    legitimate.append(tagged)
                elif kind == "events":
                    if event.get("label") not in TAXONOMY:
                        raise ValueError("snapshot.events.label: incompatible reference taxonomy")
                    reference.append(tagged)
    reviewed = normalize_coverage(reviewed)
    if schema and normalize_coverage(snapshot.get("coverage")) != reviewed:
        raise ValueError("snapshot.coverage: must equal the union of frozen reviewed items")
    frozen = snapshot.get("recording_speakers")
    if frozen is not None and (not isinstance(frozen, dict) or set(frozen) != set(reviewed)):
        raise ValueError("recording_speakers: frozen map must cover exactly the reviewed recordings")
    speakers = {}
    for rid in reviewed:
        if frozen is not None:
            speaker = frozen[rid]
        else:
            known = item_speakers[rid] - {None, ""}
            if len(known) > 1:
                raise ValueError("speaker_id: inconsistent frozen reference speakers")
            speaker = next(iter(known)) if known else recording_speakers.get(rid)
        if not isinstance(speaker, str) or not speaker.strip():
            raise ValueError("speaker_id: a known reference speaker is required for clustered evaluation")
        speakers[rid] = speaker
    tag = lambda events: [{**e, "speaker_id": speakers[e["recording_id"]]} for e in events]
    identities = [(e["recording_id"], e["start_ms"], e["end_ms"], e["label"]) for e in reference]
    if len(identities) != len(set(identities)):
        raise ValueError("duplicate positive reference marks require explicit reconciliation")
    coverage = subtract_coverage(reviewed, holes)
    return {"adapter": "native_v2" if schema else "legacy_items_v1", "reference_kind": snapshot.get("reference_kind", "legacy_unspecified"),
            "reviewed_coverage": reviewed, "human_exclusions": normalize_coverage(holes), "coverage": coverage,
            "reviewed_ms": coverage_duration(reviewed), "human_excluded_ms": coverage_duration(reviewed) - coverage_duration(coverage),
            "recording_speakers": speakers, "reference": tag(reference), "legitimate": tag(legitimate)}


def run_coverage(run: dict, segments: list[dict]) -> dict:
    """Availability comes from exact successful cores, not status or context windows."""
    rid, result = run["parent"], run["data"].get("result", {})
    from core.cell_decode import result_family

    family = result_family(result)
    if family == "cell_v2":
        from core.cell_decode import validate_cell_result, partition

        validate_cell_result(result)
        if result["recording_id"] != rid:
            raise ValueError("cell ledger and run recording identities disagree")
        # Validate recording-side cores too: never certify another segmentation.
        partition(result["cells"], segments, result["duration_ms"])
        fields = ("id", "core_start_ms", "core_end_ms")
        if [{k: s[k] for k in fields} for s in segments] != [{k: s[k] for k in fields} for s in result["execution_cores"]]:
            raise ValueError("execution and recording cores disagree")
        complete = run["status"] in ("succeeded", "partial")
        spans = result["coverage"] if complete else []
        reasons = sorted({c["failure_reason"] for c in result["cells"] if c["status"] == "failed"})
        if not complete:
            reasons.append("run_not_completed")
        if any(c["status"] == "succeeded" and not c["input_evaluable"] for c in result["cells"]):
            reasons.append("no_owned_input_for_some_cells")
        return dict(coverage={rid: spans}, available_ms=sum(b-a for a, b in spans),
                    completed_segments=[s for s, value in result["segments"].items() if value == "succeeded"] if complete else [],
                    reasons=reasons, coverage_protocol="successful_owned_input_cells_v2")
    if family == "legacy_window_v1":
        return dict(coverage={rid: []}, available_ms=None, completed_segments=[], reasons=["legacy_window_support_not_temporal_events"])
    reasons, spans, completed = [], [], []
    if run["status"] not in ("succeeded", "partial"):
        return {"coverage": {rid: []}, "available_ms": 0, "completed_segments": [], "reasons": ["run_not_completed"]}
    statuses = result.get("segments", {})
    if not isinstance(statuses, dict) or len({s["id"] for s in segments}) != len(segments):
        raise ValueError("segments: invalid run/recording segment identity")
    for segment in segments:
        if statuses.get(segment["id"]) != "succeeded":
            reasons.append(f"segment_not_succeeded:{segment['id']}")
            continue
        if "core_start_ms" not in segment or "core_end_ms" not in segment:
            raise ValueError("segments: successful segment requires exact core bounds")
        spans.append([segment["core_start_ms"], segment["core_end_ms"]])
        completed.append(segment["id"])
    coverage = {rid: union_intervals(spans)}
    if "coverage" in result:
        explicit = result["coverage"]
        explicit = explicit if isinstance(explicit, dict) else {rid: explicit}
        narrowed = intersect_coverage(coverage, explicit)
        if narrowed != coverage:
            reasons.append("explicit_inference_coverage_restriction")
        coverage = narrowed
    if not coverage_duration(coverage):
        reasons.append("no_successful_core_coverage")
    return {"coverage": coverage, "available_ms": coverage_duration(coverage), "completed_segments": completed, "reasons": reasons}


def iou(a: dict, b: dict) -> float:
    inter = max(0, min(a["end_ms"], b["end_ms"]) - max(a["start_ms"], b["start_ms"]))
    union = (a["end_ms"] - a["start_ms"]) + (b["end_ms"] - b["start_ms"]) - inter
    return inter / union if union else 0.0


def _same_rec(a: dict, b: dict) -> bool:
    return a.get("recording_id") == b.get("recording_id")


def match(ref: list[dict], hyp: list[dict], score) -> list[tuple[int, int]]:
    """One-to-one matching maximizing the number of pairs, then total score.
    `score(r, h)` returns a float for a compatible pair or None."""
    if not ref or not hyp:
        return []
    w = np.zeros((len(ref), len(hyp)))
    for i, r in enumerate(ref):
        for j, h in enumerate(hyp):
            s = score(r, h) if _same_rec(r, h) else None
            if s is not None:
                w[i, j] = len(ref) + len(hyp) + s
    rows, cols = linear_sum_assignment(w, maximize=True)
    return [(i, j) for i, j in zip(rows, cols) if w[i, j] > 0]


def by_iou(threshold: float, same_class: bool = True):
    def score(r, h):
        v = iou(r, h)
        return v if v >= threshold and (not same_class or r["label"] == h["label"]) else None

    return score


def by_tolerance(tol_ms: int):
    def score(r, h):
        ok = (
            r["label"] == h["label"]
            and abs(r["start_ms"] - h["start_ms"]) <= tol_ms
            and abs(r["end_ms"] - h["end_ms"]) <= tol_ms
        )
        return -abs(r["start_ms"] - h["start_ms"]) / 1e6 if ok else None

    return score


def in_coverage(ev: dict, coverage: dict | None) -> bool:
    if coverage is None:
        return True
    return any(a <= ev["start_ms"] < ev["end_ms"] <= b for a, b in union_intervals(coverage.get(ev.get("recording_id"), [])))


def _prf(tp: int, fp: int, fn: int) -> dict:
    p = tp / (tp + fp) if tp + fp else None
    r = tp / (tp + fn) if tp + fn else None
    f = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else None
    return {"tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": r, "f1": f, "support": tp + fn,
            "reasons": {"precision": None if p is not None else "no_predictions", "recall": None if r is not None else "no_reference",
                        "f1": None if f is not None else "no_reference_or_predictions"}}


def _counts(ref, hyp, pairs, classes):
    matched_r, matched_h = {i for i, _ in pairs}, {j for _, j in pairs}
    return {
        c: _prf(
            sum(1 for i, _ in pairs if ref[i]["label"] == c),
            sum(1 for j, h in enumerate(hyp) if h["label"] == c and j not in matched_h),
            sum(1 for i, r in enumerate(ref) if r["label"] == c and i not in matched_r),
        )
        for c in classes
    }


def _micro(per_class: dict, classes) -> dict:
    return _prf(*(sum(per_class[c][k] for c in classes) for k in ("tp", "fp", "fn")))


def evaluate(
    ref: list[dict],
    hyp: list[dict],
    *,
    classes=TAXONOMY,
    iou_threshold: float = 0.5,
    sensitivity=(0.3, 0.7),
    tolerance_ms: int = 100,
    coverage: dict | None = None,
    evaluable_ms: int | None = None,
    legitimate: list[dict] = (),
) -> dict:
    """Per-class and grouped P/R/F1 with one-to-one same-class IoU matching.

    `coverage` maps recording_id -> reviewed intervals; events outside are ignored so
    unreviewed audio never counts as negative. Pauses are reported apart from
    disfluencies. Uncertain decisions are counted, never scored as events.
    """
    if any(e.get("event_schema") == "mrcd-cell-event-v2" or e.get("prediction_kind") == "window_classification" for e in list(ref) + list(hyp) + list(legitimate)):
        raise ValueError("task-aware event evaluation not integrated (U5)")
    inputs = {"reference": len(ref), "prediction": len(hyp), "legitimate": len(legitimate)}
    if coverage is not None:
        coverage = normalize_coverage(coverage)
        holes = {}
        for event in ref:
            if event.get("decision") in ("uncertain", "not_evaluable"):
                holes.setdefault(event.get("recording_id"), []).append([event["start_ms"], event["end_ms"]])
        coverage = subtract_coverage(coverage, holes)
        if evaluable_ms is not None and evaluable_ms != coverage_duration(coverage):
            raise ValueError("evaluable_ms: must equal the union of human-masked eligible coverage")
        evaluable_ms = coverage_duration(coverage)
    keep = lambda e: e.get("label") in classes and in_coverage(e, coverage)  # noqa: E731
    positive_ref = [e for e in ref if e.get("decision", "event") == "event" and e.get("label") in classes]
    positive_hyp = [e for e in hyp if e.get("decision", "event") == "event" and e.get("label") in classes]
    abstentions = [e for e in hyp if e.get("decision") in ("uncertain", "not_evaluable")]

    def excluded(events):
        outside = [e for e in events if not in_coverage(e, coverage)]
        boundary = sum(any(max(e["start_ms"], a) < min(e["end_ms"], b) for a, b in coverage.get(e.get("recording_id"), [])) for e in outside) if coverage is not None else 0
        return {"count": len(outside), "boundary_or_hole": boundary, "outside_coverage": len(outside) - boundary}

    exclusions = {kind: excluded(events) for kind, events in (("reference", positive_ref), ("prediction", positive_hyp),
                  ("uncertain", abstentions), ("legitimate", legitimate))}
    uncertain = [h for h in abstentions if h.get("decision") == "uncertain" and in_coverage(h, coverage)]
    ref, hyp = [e for e in positive_ref if keep(e)], [e for e in positive_hyp if keep(e)]
    legitimate = [e for e in legitimate if in_coverage(e, coverage)]
    pairs = match(ref, hyp, by_iou(iou_threshold))
    per_class = _counts(ref, hyp, pairs, classes)
    groups = {"disfluency": [c for c in classes if c in DISFLUENCIES], "pause": [c for c in classes if c in PAUSES]}
    onset = [abs(ref[i]["start_ms"] - hyp[j]["start_ms"]) for i, j in pairs]
    offset = [abs(ref[i]["end_ms"] - hyp[j]["end_ms"]) for i, j in pairs]
    duration = lambda e: e["end_ms"] - e["start_ms"]  # noqa: E731
    matched_r = {i for i, _ in pairs}
    by_duration = {}
    for lo, hi in zip(DURATION_BINS_MS, DURATION_BINS_MS[1:]):
        idx = [i for i, e in enumerate(ref) if lo <= duration(e) < hi]
        by_duration[f"{lo}-{hi if hi < 10**9 else 'inf'}ms"] = {
            "n": len(idx),
            "recall": sum(i in matched_r for i in idx) / len(idx) if idx else None,
            "reason": None if idx else "no_reference_in_duration_bin",
        }
    tol_counts = _counts(ref, hyp, match(ref, hyp, by_tolerance(tolerance_ms)), classes)
    fp_events = [h for j, h in enumerate(hyp) if j not in {j for _, j in pairs}]
    result = {
        "protocol": {
            "version": PROTOCOL_VERSION,
            "boundary_policy": BOUNDARY_POLICY,
            "universe": "coverage_v2" if coverage is not None else "duration_only_legacy" if evaluable_ms is not None else "events_only_legacy",
            "iou": iou_threshold,
            "tolerance_ms": tolerance_ms,
            "classes": list(classes),
            "matching": "one_to_one_same_class_max_cardinality",
        },
        "evaluable_ms": evaluable_ms,
        "coverage": coverage,
        "n_input": inputs,
        "exclusions": exclusions,
        "n_ref": len(ref),
        "n_hyp": len(hyp),
        "n_uncertain": len(uncertain),
        "per_class": per_class,
        "micro": {g: _micro(per_class, cs) for g, cs in groups.items() if cs},
        "macro_f1": {
            g: _mean([per_class[c]["f1"] for c in cs if per_class[c]["support"]]) for g, cs in groups.items() if cs
        },
        "macro_classes": {g: [c for c in cs if per_class[c]["support"]] for g, cs in groups.items() if cs},
        "boundaries": {
            "matched_proportion": len(pairs) / len(ref) if ref else None,
            "onset_abs_ms": _summary(onset),
            "offset_abs_ms": _summary(offset),
        },
        "sensitivity": {
            str(t): _micro(_counts(ref, hyp, match(ref, hyp, by_iou(t)), classes), groups["disfluency"])
            for t in sensitivity
        },
        "tolerance": {g: _micro(tol_counts, cs) for g, cs in groups.items() if cs},
        "by_duration": by_duration,
        "false_alarms_per_min": (
            len([h for h in fp_events if h["label"] in DISFLUENCIES]) / (evaluable_ms / 60000) if evaluable_ms else None
        ),
        "false_alarms_on_legitimate": sum(
            any(_same_rec(h, g) and iou(h, g) > 0 for g in legitimate) for h in fp_events
        ),
        "errors": error_types(ref, hyp, pairs, iou_threshold),
    }
    result["reason"] = "no_evaluable_coverage" if evaluable_ms == 0 else None
    result["undefined_reasons"] = {
        "false_alarms_per_min": "no_evaluable_coverage" if evaluable_ms == 0 else "evaluable_duration_not_supplied" if evaluable_ms is None else None,
        "boundaries": "no_matched_events" if not pairs else None,
        "macro_f1": {g: "no_supported_reference_classes" if value is None else None for g, value in result["macro_f1"].items()},
    }
    if evaluable_ms == 0:
        def unavailable():
            return {**{k: None for k in ("tp", "fp", "fn", "precision", "recall", "f1", "support")},
                    "reasons": {k: "no_evaluable_coverage" for k in ("precision", "recall", "f1")}}
        result.update(n_ref=None, n_hyp=None, n_uncertain=None, false_alarms_per_min=None, false_alarms_on_legitimate=None)
        for key in ("per_class", "micro", "tolerance", "sensitivity"):
            result[key] = {k: unavailable() for k in result[key]}
        result["macro_f1"] = {k: None for k in result["macro_f1"]}
        result["by_duration"] = {k: {"n": None, "recall": None, "reason": "no_evaluable_coverage"} for k in result["by_duration"]}
        result["errors"] = {k: None for k in result["errors"]}
        result["undefined_reasons"]["boundaries"] = "no_evaluable_coverage"
    return result


def system_agreement(a: list[dict], b: list[dict], *, coverage: dict, **protocol) -> dict:
    """Symmetric model-event F1 on fixed shared availability, never human masks.

    Model abstentions and functional decisions are not positive events on either
    side. They cannot become human exclusion holes simply because A is passed in
    the reference argument of the underlying one-to-one matcher.
    """
    for event in a + b:
        if event.get("event_schema") == "mrcd-cell-event-v2" or event.get("prediction_kind") == "window_classification":
            raise ValueError("task-aware agreement not integrated (U5)")
    decided = lambda events: [e for e in events if e.get("decision", "event") == "event" and e.get("is_disfluent") is not False]
    result = evaluate(decided(a), decided(b), coverage=normalize_coverage(coverage), **protocol)
    result["protocol"]["comparison_kind"] = "symmetric_model_agreement_not_accuracy"
    return result


def error_types(ref, hyp, pairs, threshold) -> dict:
    """Automatic part of the error taxonomy: class vs interval vs unsupported."""
    out = {"class": 0, "interval": 0, "unsupported": 0, "missed": 0}
    mr, mh = {i for i, _ in pairs}, {j for _, j in pairs}
    for j, h in enumerate(hyp):
        if j in mh:
            continue
        near = [r for i, r in enumerate(ref) if i not in mr and _same_rec(r, h) and iou(r, h) > 0]
        out[
            "class"
            if any(iou(r, h) >= threshold for r in near)
            else "interval"
            if any(r["label"] == h["label"] for r in near)
            else "unsupported"
        ] += 1
    out["missed"] = len(ref) - len(mr)
    return out


def _mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def _summary(values):
    return {"n": len(values), "mean": float(np.mean(values)), "median": float(np.median(values))} if values else None


def agreement(a: list[dict], b: list[dict], *, iou_threshold: float = 0.5) -> dict:
    """Independent A/B annotations: existence, class and boundaries reported apart.

    Class kappa uses common units = existence-matched pairs plus each unmatched event
    against "none", so omissions count as disagreement rather than being dropped.
    """
    pairs = match(a, b, by_iou(iou_threshold, same_class=False))
    ma, mb = {i for i, _ in pairs}, {j for _, j in pairs}
    units = [(a[i]["label"], b[j]["label"]) for i, j in pairs]
    units += [(e["label"], "none") for i, e in enumerate(a) if i not in ma]
    units += [("none", e["label"]) for j, e in enumerate(b) if j not in mb]
    la, lb = zip(*units) if units else ((), ())
    names = sorted(set(la) | set(lb))
    return {
        "n_a": len(a),
        "n_b": len(b),
        "matched": len(pairs),
        "existence_agreement": 2 * len(pairs) / (len(a) + len(b)) if a or b else None,
        "class_raw_agreement": _mean([x == y for x, y in units]),
        "class_kappa": float(cohen_kappa_score(la, lb)) if len(set(la) | set(lb)) > 1 else None,
        "confusion": {
            "labels": names,
            "matrix": [[sum(1 for u in units if u == (x, y)) for y in names] for x in names],
        },
        "onset_abs_ms": _summary([abs(a[i]["start_ms"] - b[j]["start_ms"]) for i, j in pairs]),
        "offset_abs_ms": _summary([abs(a[i]["end_ms"] - b[j]["end_ms"]) for i, j in pairs]),
    }


def resample_clusters(ref, hyps, coverage, recording_speakers, draw, legitimate=()) -> dict:
    """Resample whole coverage clusters, retaining negative-only copies and duration."""
    coverage = normalize_coverage(coverage)
    sampled, refs, predictions, functional = {}, [], [[] for _ in hyps], []
    for index, speaker in enumerate(draw):
        for rid, spans in coverage.items():
            if not spans or recording_speakers.get(rid) != speaker:
                continue
            copied_id = f"{rid}#bootstrap-copy-{index}"
            sampled[copied_id] = [list(span) for span in spans]
            def copy(events):
                return [{**e, "recording_id": copied_id, "speaker_id": f"{speaker}#copy-{index}",
                         "event_id": f"{copied_id}:{e.get('event_id') or position}"}
                        for position, e in enumerate(events) if e.get("recording_id") == rid]
            refs.extend(copy(ref))
            functional.extend(copy(legitimate))
            for target, events in zip(predictions, hyps):
                target.extend(copy(events))
    return {"reference": refs, "hypotheses": predictions, "coverage": sampled, "legitimate": functional}


def bootstrap_ci(ref: list[dict], hyps: list[list[dict]], stat, *, n: int = 1000, seed: int = 13,
                 coverage: dict | None = None, recording_speakers: dict | None = None, legitimate=()) -> dict:
    """Coverage-based speaker percentile CI; paired hypotheses share each draw.

    With coverage, stat receives the resampled coverage/legitimate/evaluable_ms.
    Without it, preserve the old event-only callable contract, explicitly labeled.
    """
    if type(n) is not int or not 0 <= n <= 10000:
        raise ValueError("bootstrap: requested replicates must be an integer in 0..10000")
    if coverage is not None:
        coverage = normalize_coverage(coverage)
        recording_speakers = recording_speakers or {}
        if any(not recording_speakers.get(rid) for rid, spans in coverage.items() if spans):
            raise ValueError("bootstrap: every covered recording requires a frozen speaker")
        speakers = sorted({recording_speakers[rid] for rid, spans in coverage.items() if spans}, key=str)
    else:
        speakers = sorted({e.get("speaker_id") for ev in (ref, *hyps) for e in ev}, key=str)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n):
        if not speakers:
            break
        draw = rng.choice(len(speakers), len(speakers))

        def pick(events):
            return [
                {**e, "recording_id": f"{e.get('recording_id')}#{k}"}
                for k, s in enumerate(draw)
                for e in events
                if e.get("speaker_id") == speakers[s]
            ]

        if coverage is None:
            v = stat(pick(ref), *map(pick, hyps))
        else:
            sampled = resample_clusters(ref, hyps, coverage, recording_speakers, [speakers[s] for s in draw], legitimate)
            v = stat(sampled["reference"], *sampled["hypotheses"], coverage=sampled["coverage"],
                     legitimate=sampled["legitimate"], evaluable_ms=coverage_duration(sampled["coverage"]))
        if v is not None and np.isfinite(v):
            values.append(float(v))
    lo, hi = np.percentile(values, [2.5, 97.5]) if values else (None, None)
    return {"low": float(lo) if lo is not None else None, "high": float(hi) if hi is not None else None,
            "n_boot": len(values), "n_speakers": len(speakers), "requested": n, "valid": len(values), "omitted": n - len(values),
            "seed": seed, "population": "covered_speakers_v2" if coverage is not None else "events_only_legacy",
            "reason": "no_evaluable_coverage" if coverage is not None and not speakers else "bootstrap_not_requested" if not n else "statistic_undefined_in_all_replicates" if not values else None}


def evaluate_runs(snapshot: dict, systems: dict, recordings: dict, *, protocol=None, bootstrap=1000) -> dict:
    """Pure own/paired evaluation on distinct, explicit eligibility universes."""
    if not systems or any(not runs for runs in systems.values()):
        raise ValueError("systems: select at least one run per named system")
    from core.cell_decode import require_legacy_consumer
    for runs in systems.values():
        for run in runs:
            require_legacy_consumer(run["data"].get("result", {}))
    parameters = {"iou_threshold": 0.5, "sensitivity": (0.3, 0.7), "tolerance_ms": 100, **(protocol or {})}
    if set(parameters) - {"iou_threshold", "sensitivity", "tolerance_ms", "classes"}:
        raise ValueError("protocol: unknown or protected protocol fields")
    if not isinstance(parameters["iou_threshold"], (int, float)) or not 0 < parameters["iou_threshold"] <= 1:
        raise ValueError("iou_threshold: expected 0 < IoU <= 1")
    if type(parameters["tolerance_ms"]) is not int or parameters["tolerance_ms"] < 0:
        raise ValueError("tolerance_ms: expected a nonnegative integer")
    if not isinstance(parameters["sensitivity"], (list, tuple)) or any(not isinstance(t, (int, float)) or not 0 < t <= 1 for t in parameters["sensitivity"]):
        raise ValueError("sensitivity: expected IoU thresholds in (0, 1]")
    if "classes" in parameters and (not isinstance(parameters["classes"], (list, tuple)) or not parameters["classes"] or
            len(set(parameters["classes"])) != len(parameters["classes"]) or not set(parameters["classes"]).issubset(TAXONOMY)):
        raise ValueError("classes: expected unique canonical taxonomy labels")
    seen, identities = set(), set()
    for name, runs in systems.items():
        system_names = {r["data"].get("system") for r in runs}
        if len(system_names) != 1 or None in system_names:
            raise ValueError("systems: one actual system identity is required per named selection")
        for run in runs:
            if run["id"] in seen:
                raise ValueError("duplicate run IDs are not independent evidence")
            identity = (run["data"]["system"], run["parent"])
            if identity in identities:
                raise ValueError("multiple selected runs for the same system and recording")
            seen.add(run["id"])
            identities.add(identity)
    view = reference_view(snapshot, {rid: rec["data"].get("speaker_id") for rid, rec in recordings.items()})
    hyps, universes, results, availability, cis, missing = {}, {}, {}, {}, {}, {}
    ref, legitimate = view["reference"], view["legitimate"]
    def score(hyp, coverage):
        return evaluate(ref, hyp, coverage=coverage, legitimate=legitimate, **parameters)
    def f1(r, h, **kw):
        return evaluate(r, h, **kw, **parameters)["micro"].get("disfluency", {}).get("f1")
    for name, runs in systems.items():
        available, hyps[name], availability[name] = {}, [], []
        for run in runs:
            rid = run["parent"]
            if rid not in recordings:
                raise ValueError("recording_id: selected run lacks recording metadata")
            spans = run_coverage(run, recordings[rid]["data"].get("segments", []))
            availability[name].append({"run_id": run["id"], "run_revision": run.get("rev"), "recording_id": rid,
                                       "recording_revision": recordings[rid].get("rev"), **spans})
            available.setdefault(rid, []).extend(spans["coverage"].get(rid, []))
            for event in run["data"].get("result", {}).get("events", []):
                hyps[name].append({**event, "recording_id": rid, "speaker_id": view["recording_speakers"].get(rid)})
        coverage = intersect_coverage(view["coverage"], available)
        universes[name] = coverage
        results[name] = score(hyps[name], coverage)
        results[name]["unavailable_ms"] = coverage_duration(view["coverage"]) - coverage_duration(coverage)
        missing[name] = sorted(rid for rid, spans in view["coverage"].items() if spans and not coverage.get(rid))
        cis[name] = bootstrap_ci(ref, [hyps[name]], f1, n=bootstrap, coverage=coverage,
                                 recording_speakers=view["recording_speakers"], legitimate=legitimate)
    paired, omitted_aliases = {}, []
    names = list(systems)
    for index, a in enumerate(names):
        for b in names[index + 1:]:
            key = f"{b}-{a}"
            if key in paired:
                raise ValueError("systems: ambiguous paired names; choose distinct unambiguous display names")
            coverage = intersect_coverage(universes[a], universes[b])
            pair_scores = {name: score(hyps[name], coverage) for name in (a, b)}
            fa, fb = (pair_scores[name]["micro"].get("disfluency", {}).get("f1") for name in (a, b))
            def delta(r, ha, hb, **kw):
                x, y = f1(r, ha, **kw), f1(r, hb, **kw)
                return y - x if x is not None and y is not None else None
            ci = bootstrap_ci(ref, [hyps[a], hyps[b]], delta, n=bootstrap, coverage=coverage,
                              recording_speakers=view["recording_speakers"], legitimate=legitimate)
            paired[key] = {"systems": [a, b], "coverage": coverage, "evaluable_ms": coverage_duration(coverage),
                           "results": pair_scores, "delta_f1": fb - fa if fa is not None and fb is not None else None,
                           "reason": "no_evaluable_coverage" if not coverage_duration(coverage) else "f1_undefined_for_pair" if fa is None or fb is None else None,
                           "bootstrap_ci": ci}
            if key not in cis:
                cis[key] = ci  # compatibility alias; never shadow an own-system CI
            else:
                omitted_aliases.append(key)
    return {"protocol": {**parameters, "version": PROTOCOL_VERSION, "boundary_policy": BOUNDARY_POLICY,
                         "bootstrap_population": "covered_speakers_v2"},
            "reference_view": view, "results": results, "paired_results": paired, "availability": availability,
            "missing_runs": missing, "bootstrap_ci": cis, "paired_ci_aliases_omitted": omitted_aliases}


def calibration(p: np.ndarray, y: np.ndarray, bins: int = 10) -> dict:
    """Binary reliability: `p` predicted probability of the event, `y` 0/1 outcome."""
    p, y = np.asarray(p, float), np.asarray(y, float)
    edges = np.minimum((p * bins).astype(int), bins - 1)
    table = [
        {"p_mean": float(p[edges == k].mean()), "freq": float(y[edges == k].mean()), "n": int((edges == k).sum())}
        for k in range(bins)
        if (edges == k).any()
    ]
    return {
        "brier": float(np.mean((p - y) ** 2)),
        "n": len(p),
        "bins": table,
        "ece": float(sum(t["n"] * abs(t["p_mean"] - t["freq"]) for t in table) / len(p)) if len(p) else None,
    }


def fit_temperature(probs: np.ndarray, y: np.ndarray) -> float:
    """Temperature for softmax(log p / T) minimizing dev NLL; never fit on test."""
    logp = np.log(np.clip(probs, 1e-12, 1))

    def nll(t):
        z = logp / t
        z -= z.max(1, keepdims=True)
        return -np.mean(z[np.arange(len(y)), y] - np.log(np.exp(z).sum(1)))

    return float(minimize_scalar(nll, bounds=(0.05, 20), method="bounded").x)


def coverage_risk(scores, correct, threshold: float) -> dict:
    """Fixed universe of units: coverage = share decided automatically, risk = error among them."""
    scores, correct = np.asarray(scores, float), np.asarray(correct, bool)
    decided = scores >= threshold
    return {
        "threshold": threshold,
        "n_units": len(scores),
        "coverage": float(decided.mean()) if len(scores) else None,
        "risk": float(1 - correct[decided].mean()) if decided.any() else None,
        "abstained": int((~decided).sum()),
    }


def pick_threshold(scores, correct, max_risk: float) -> float | None:
    """Lowest threshold whose dev risk stays within `max_risk` (maximum coverage)."""
    for t in sorted(set(np.asarray(scores, float))):
        r = coverage_risk(scores, correct, t)["risk"]
        if r is not None and r <= max_risk:
            return float(t)
    return None


def select_for_review(cases: list[dict], n: int, *, quotas=(0.4, 0.3, 0.3), seed: int = 13) -> list[dict]:
    """Development-only active selection: disagreement / uncertainty / random quotas,
    round-robin across speakers so one speaker cannot dominate. Never for test."""
    if any(c.get("split") == "test" for c in cases):
        raise ValueError("La selección activa no se usa sobre el conjunto de prueba.")
    rng = np.random.default_rng(seed)
    chosen, seen = [], set()
    for kind, share in zip(("disagreement", "uncertain", None), quotas):
        pool = [c for c in cases if (kind is None or c.get("kind") == kind) and id(c) not in seen]
        rng.shuffle(pool)
        seen_by_speaker: dict = {}
        rank = []
        for c in pool:  # k-th case of its speaker gets rank k -> round-robin order
            seen_by_speaker[c.get("speaker_id")] = seen_by_speaker.get(c.get("speaker_id"), 0) + 1
            rank.append(seen_by_speaker[c.get("speaker_id")])
        pool = [c for _, c in sorted(zip(rank, pool), key=lambda rc: rc[0])]
        for c in pool[: round(n * share)]:
            seen.add(id(c))
            chosen.append({**c, "selection": kind or "random"})
    return chosen


def sus_score(answers: list[int]) -> float:
    """System Usability Scale, 10 items on 1..5; odd items positive, even negative."""
    if len(answers) != 10 or not all(1 <= a <= 5 for a in answers):
        raise ValueError("SUS requiere 10 respuestas entre 1 y 5")
    return 2.5 * sum(a - 1 if i % 2 == 0 else 5 - a for i, a in enumerate(answers))
