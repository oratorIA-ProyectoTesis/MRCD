"""Inter-annotator agreement for the Phase 0.5 annotation campaign.

Input: two CSVs (one per annotator) with columns
    item_id, annotator, class, state, start_s, end_s[, note]
An item with no event carries one row with class=none, state=none and empty
times, so "looked and found nothing" is distinguishable from "not annotated".
Items present in only one file are reported and dropped from the statistics.

Event alignment (per item, never across items): candidate pairs are events of
A and B whose temporal IoU >= min_iou; pairs are taken greedily by descending
IoU, one-to-one. Unmatched events pair with the label NONE of the other side.
Units = matched pairs + unmatched events. Over these units we compute

* Cohen kappa and Krippendorff alpha (nominal) on the joint label class:state
  and on the class alone;
* per-class positive agreement (Dice over events: 2*both / (nA + nB));
* temporal agreement: mean IoU of matched pairs and the share of matched pairs
  whose start AND end differ by <= tol seconds;
* item-level kappa on "item contains at least one event".

Undefined values (zero variance, no pairs) are returned as None = not
available; they are never reported as 0.

Discourse-marker function layer (columns marker_token, marker_function; rows with
class=discourse_marker): ignored by the event statistics above and analysed by
`compute_markers` / `--field marker_function`. Sheets without those columns are
valid and simply have no marker spans.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from scripts.annotation.markers import (
    MARKER_CLASS,
    MIN_DETECTION_F1,
    MIN_MARKER_SPANS,
    check_marker,
)

NONE = "none"


def read_annotations(path: Path) -> dict[str, list[dict]]:
    items: dict[str, list[dict]] = defaultdict(list)
    with Path(path).open(encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            cls = (r["class"] or "").strip().lower()
            _ = items[r["item_id"]]  # register the item even when it has no event
            if cls == NONE or cls == MARKER_CLASS:
                continue
            items[r["item_id"]].append(
                {
                    "class": cls,
                    "state": (r["state"] or "").strip().lower(),
                    "start": float(r["start_s"]),
                    "end": float(r["end_s"]),
                }
            )
    return dict(items)


def iou(a: dict, b: dict) -> float:
    inter = min(a["end"], b["end"]) - max(a["start"], b["start"])
    if inter <= 0:
        return 0.0
    union = (a["end"] - a["start"]) + (b["end"] - b["start"]) - inter
    return inter / union if union > 0 else 0.0


def match_events(ev_a: list[dict], ev_b: list[dict], min_iou: float):
    """Greedy one-to-one matching. Returns (pairs, unmatched_a, unmatched_b)."""
    cands = sorted(
        ((iou(a, b), i, j) for i, a in enumerate(ev_a) for j, b in enumerate(ev_b)),
        reverse=True,
    )
    used_a, used_b, pairs = set(), set(), []
    for v, i, j in cands:
        if v < min_iou or v <= 0:
            break
        if i in used_a or j in used_b:
            continue
        used_a.add(i)
        used_b.add(j)
        pairs.append((ev_a[i], ev_b[j], v))
    return (
        pairs,
        [e for i, e in enumerate(ev_a) if i not in used_a],
        [e for j, e in enumerate(ev_b) if j not in used_b],
    )


def cohen_kappa(x: list, y: list) -> float | None:
    n = len(x)
    if n == 0:
        return None
    po = sum(a == b for a, b in zip(x, y)) / n
    cx, cy = Counter(x), Counter(y)
    pe = sum(cx[k] * cy[k] for k in set(cx) | set(cy)) / (n * n)
    if math.isclose(pe, 1.0):
        return None
    return (po - pe) / (1 - pe)


def krippendorff_alpha_nominal(x: list, y: list) -> float | None:
    """Two coders, no missing data: alpha = 1 - Do/De with nominal distance."""
    n = len(x)
    if n == 0:
        return None
    do = sum(a != b for a, b in zip(x, y)) / n
    counts = Counter(x) + Counter(y)
    total = 2 * n
    de = 1 - sum(c * (c - 1) for c in counts.values()) / (total * (total - 1))
    if math.isclose(de, 0.0):
        return None
    return 1 - do / de


MODES = ("inter", "intra", "subset")
MIN_SUBSET_ITEMS = 40


def annotator_ids(path: Path) -> set[str]:
    with Path(path).open(encoding="utf-8", newline="") as fh:
        return {r["annotator"].strip() for r in csv.DictReader(fh)}


def read_markers(path: Path) -> dict[str, list[dict]]:
    """Marker spans per item. Every item is registered; sheets lacking the
    marker columns yield items with no spans (backward compatible)."""
    items: dict[str, list[dict]] = defaultdict(list)
    with Path(path).open(encoding="utf-8", newline="") as fh:
        for n, r in enumerate(csv.DictReader(fh), start=2):
            _ = items[r["item_id"]]
            if (r["class"] or "").strip().lower() != MARKER_CLASS:
                continue
            try:
                tok, fn = check_marker(r.get("marker_token"), r.get("marker_function"))
                start, end = float(r["start_s"]), float(r["end_s"])
            except ValueError as exc:
                raise ValueError(
                    f"{Path(path).name} line {n} (item {r['item_id']}): {exc}"
                ) from None
            items[r["item_id"]].append(
                {"token": tok, "function": fn, "start": start, "end": end}
            )
    return dict(items)


def _check_mode(a_path, b_path, mode: str) -> None:
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    ids_a, ids_b = annotator_ids(a_path), annotator_ids(b_path)
    if mode == "intra" and ids_a != ids_b:
        raise ValueError(
            f"intra mode needs the same annotator in both files, got {ids_a} vs {ids_b}"
        )
    if mode in ("inter", "subset") and ids_a & ids_b:
        raise ValueError(
            f"{mode} mode needs different annotators, got overlap {ids_a & ids_b}"
        )


def compute_markers(
    a_path,
    b_path,
    min_iou: float = 0.1,
    mode: str = "inter",
    min_spans: int = MIN_MARKER_SPANS,
) -> dict:
    """Agreement on marker_function. Spans are matched per (item, token) with the
    same greedy IoU matcher as the event statistics; kappa and alpha use matched
    spans only. Fewer than `min_spans` matched spans -> kappa/alpha are None with a
    reason (never 0). Unmatched spans and the share of `uncertain` are reported.
    match_rate = matched / max(n_a, n_b); detection_f1 = 2 * matched / (n_a + n_b),
    both pooled over common items; the *_any_token variants ignore the token when
    matching and n_token_mismatches counts overlapping pairs with different tokens.
    kappa / alpha are always read together with them: detection F1 below
    MIN_DETECTION_F1 flags function_kappa_descriptive_only. Subset mode flags
    coverage below MIN_SUBSET_ITEMS like `compute`."""
    _check_mode(a_path, b_path, mode)
    A, B = read_markers(a_path), read_markers(b_path)
    common = sorted(set(A) & set(B))
    fa: list[str] = []
    fb: list[str] = []
    un_a = un_b = 0
    all_a: Counter = Counter()
    all_b: Counter = Counter()
    for item in common:
        for tok in sorted(
            {s["token"] for s in A[item]} | {s["token"] for s in B[item]}
        ):
            sa = [s for s in A[item] if s["token"] == tok]
            sb = [s for s in B[item] if s["token"] == tok]
            pairs, ua, ub = match_events(sa, sb, min_iou)
            for x, y, _v in pairs:
                fa.append(x["function"])
                fb.append(y["function"])
            un_a += len(ua)
            un_b += len(ub)
        all_a.update(s["function"] for s in A[item])
        all_b.update(s["function"] for s in B[item])
    n = len(fa)
    enough = n >= min_spans
    tot_a, tot_b = sum(all_a.values()), sum(all_b.values())
    # Detection agreement: token-aware (same token + IoU) and token-agnostic (IoU only).
    any_pairs = 0
    tok_mismatch = 0
    for item in common:
        pairs, _ua, _ub = match_events(A[item], B[item], min_iou)
        any_pairs += len(pairs)
        tok_mismatch += sum(x["token"] != y["token"] for x, y, _v in pairs)
    top = max(tot_a, tot_b)
    match_rate = (n / top) if top else None
    det_f1 = (2 * n / (tot_a + tot_b)) if tot_a + tot_b else None
    any_rate = (any_pairs / top) if top else None
    any_f1 = (2 * any_pairs / (tot_a + tot_b)) if tot_a + tot_b else None
    descriptive_only = det_f1 is None or det_f1 < MIN_DETECTION_F1
    return {
        "mode": mode,
        "field": "marker_function",
        "params": {"min_iou": min_iou, "min_matched_spans": min_spans},
        "n_items_common": len(common),
        "items_only_in_a": sorted(set(A) - set(B)),
        "items_only_in_b": sorted(set(B) - set(A)),
        "n_spans_a": tot_a,
        "n_spans_b": tot_b,
        "subset_below_minimum": (mode == "subset" and len(common) < MIN_SUBSET_ITEMS),
        "n_matched_spans": n,
        "n_matched_any_token": any_pairs,
        "n_token_mismatches": tok_mismatch,
        "match_rate": match_rate,
        "detection_f1": det_f1,
        "match_rate_any_token": any_rate,
        "detection_f1_any_token": any_f1,
        "min_detection_f1": MIN_DETECTION_F1,
        "function_kappa_descriptive_only": descriptive_only,
        "n_unmatched_a": un_a,
        "n_unmatched_b": un_b,
        "kappa": cohen_kappa(fa, fb) if enough else None,
        "alpha": krippendorff_alpha_nominal(fa, fb) if enough else None,
        "not_available_reason": (
            None if enough else f"fewer than {min_spans} matched marker spans (n = {n})"
        ),
        "proportion_uncertain": (
            (all_a["uncertain"] + all_b["uncertain"]) / (tot_a + tot_b)
            if tot_a + tot_b
            else None
        ),
        "proportion_uncertain_a": (all_a["uncertain"] / tot_a) if tot_a else None,
        "proportion_uncertain_b": (all_b["uncertain"] / tot_b) if tot_b else None,
        "counts_a": dict(all_a),
        "counts_b": dict(all_b),
    }


def compute(
    a_path,
    b_path,
    min_iou: float = 0.1,
    tol: float = 0.25,
    mode: str = "inter",
) -> dict:
    """mode: inter = two raters, same items; intra = same rater, two passes
    (item_id must already be the ORIGINAL id, see unblind.py); subset = rater 2
    labelled only part of the items, statistics use the intersection and the
    coverage is reported (fewer than MIN_SUBSET_ITEMS is flagged)."""
    _check_mode(a_path, b_path, mode)
    A, B = read_annotations(a_path), read_annotations(b_path)
    common = sorted(set(A) & set(B))
    units: list[tuple[str, str, str, str]] = []  # (label_a, label_b, cls_a, cls_b)
    ious: list[float] = []
    within_tol = 0
    n_a: Counter = Counter()
    n_b: Counter = Counter()
    both: Counter = Counter()
    for item in common:
        pairs, ua, ub = match_events(A[item], B[item], min_iou)
        for ea, eb, v in pairs:
            units.append(
                (
                    f"{ea['class']}:{ea['state']}",
                    f"{eb['class']}:{eb['state']}",
                    ea["class"],
                    eb["class"],
                )
            )
            ious.append(v)
            within_tol += (
                round(abs(ea["start"] - eb["start"]), 3) <= tol
                and round(abs(ea["end"] - eb["end"]), 3) <= tol
            )
            if ea["class"] == eb["class"]:
                both[ea["class"]] += 1
        for e in ua:
            units.append((f"{e['class']}:{e['state']}", NONE, e["class"], NONE))
        for e in ub:
            units.append((NONE, f"{e['class']}:{e['state']}", NONE, e["class"]))
        for e in A[item]:
            n_a[e["class"]] += 1
        for e in B[item]:
            n_b[e["class"]] += 1
    la, lb = [u[0] for u in units], [u[1] for u in units]
    ca, cb = [u[2] for u in units], [u[3] for u in units]
    ia = [bool(A[i]) for i in common]
    ib = [bool(B[i]) for i in common]
    per_class = {
        c: {
            "n_a": n_a[c],
            "n_b": n_b[c],
            "matched_same_class": both[c],
            "dice": (2 * both[c] / (n_a[c] + n_b[c])) if (n_a[c] + n_b[c]) else None,
        }
        for c in sorted(set(n_a) | set(n_b))
    }
    return {
        "mode": mode,
        "params": {"min_iou": min_iou, "boundary_tol_s": tol},
        "n_items_a": len(A),
        "n_items_b": len(B),
        "subset_below_minimum": (mode == "subset" and len(common) < MIN_SUBSET_ITEMS),
        "n_items_common": len(common),
        "items_only_in_a": sorted(set(A) - set(B)),
        "items_only_in_b": sorted(set(B) - set(A)),
        "n_units": len(units),
        "n_matched_pairs": len(ious),
        "n_unmatched_a": sum(u[1] == NONE for u in units),
        "n_unmatched_b": sum(u[0] == NONE for u in units),
        "kappa_label": cohen_kappa(la, lb),
        "alpha_label": krippendorff_alpha_nominal(la, lb),
        "kappa_class": cohen_kappa(ca, cb),
        "alpha_class": krippendorff_alpha_nominal(ca, cb),
        "kappa_item_has_event": cohen_kappa(ia, ib),
        "per_class": per_class,
        "mean_iou_matched": (sum(ious) / len(ious)) if ious else None,
        "boundary_agreement": (within_tol / len(ious)) if ious else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("annotator_a", type=Path)
    ap.add_argument("annotator_b", type=Path)
    ap.add_argument("--min-iou", type=float, default=0.1)
    ap.add_argument(
        "--tol", type=float, default=0.25, help="boundary tolerance in seconds"
    )
    ap.add_argument("--mode", choices=MODES, default="inter")
    ap.add_argument(
        "--field",
        choices=("events", "marker_function"),
        default="events",
        help="events = class/state statistics; marker_function = discourse-marker layer",
    )
    a = ap.parse_args()
    if a.field == "marker_function":
        out = compute_markers(a.annotator_a, a.annotator_b, a.min_iou, a.mode)
        print(json.dumps(out, indent=2, ensure_ascii=False))
        return
    out = compute(a.annotator_a, a.annotator_b, a.min_iou, a.tol, a.mode)
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
