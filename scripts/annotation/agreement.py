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
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

NONE = "none"


def read_annotations(path: Path) -> dict[str, list[dict]]:
    items: dict[str, list[dict]] = defaultdict(list)
    with Path(path).open(encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            cls = (r["class"] or "").strip().lower()
            _ = items[r["item_id"]]  # register the item even when it has no event
            if cls == NONE:
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
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    ids_a, ids_b = annotator_ids(a_path), annotator_ids(b_path)
    if mode == "intra" and ids_a != ids_b:
        raise ValueError(f"intra mode needs the same annotator in both files, got {ids_a} vs {ids_b}")
    if mode in ("inter", "subset") and ids_a & ids_b:
        raise ValueError(f"{mode} mode needs different annotators, got overlap {ids_a & ids_b}")
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
    a = ap.parse_args()
    out = compute(a.annotator_a, a.annotator_b, a.min_iou, a.tol, a.mode)
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
