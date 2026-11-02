"""Adjudication support: a working EAF that shows both annotators and pre-fills only what they agree on, plus an
automatic change log comparing the final adjudicated tiers with the two independent annotations.

Only human annotations are used. The pre-fill is the average of the boundaries of matched events (same tier, same
value, IoU >= primary threshold); every disagreement is left for the adjudicator to decide.
"""

from __future__ import annotations

import csv
from pathlib import Path

from mrcd.annotation_eval import metrics as M
from mrcd.annotation_eval.config import CONFIRMATORY as CFG
from mrcd.annotation_eval.eaf import Ev, build_eaf, parse_eaf, write_eaf
from mrcd.annotation_eval.schema import ALL_CLASSES, TIER_KEYS

IV = lambda e: (e.start_ms / 1000, e.end_ms / 1000)  # noqa: E731


def _match(evs_a: list[Ev], evs_b: list[Ev]):
    return M.match_events(
        [IV(e) for e in evs_a], [IV(e) for e in evs_b], CFG["primary_iou"]
    )


def build_adjudication(
    path_a1: Path,
    path_a2: Path,
    out_path: Path,
    media_name: str,
    guide_version: str | None = None,
):
    pa, pb = parse_eaf(path_a1), parse_eaf(path_a2)
    rec = pa.properties["recording_id"]
    if rec != pb.properties.get("recording_id") or pa.duration_ms != pb.duration_ms:
        raise ValueError("annotator files do not belong to the same recording")
    events = [e for e in pa.events if e.annotator == "A1"] + [
        e for e in pb.events if e.annotator == "A2"
    ]
    prefill: list[Ev] = []
    disagreements: list[dict] = []
    for key in TIER_KEYS:
        a = sorted(
            (e for e in pa.events if e.annotator == "A1" and e.tier == key),
            key=lambda e: e.start_ms,
        )
        b = sorted(
            (e for e in pb.events if e.annotator == "A2" and e.tier == key),
            key=lambda e: e.start_ms,
        )
        if key == "excluir":  # union of excluded regions
            for s, t in M.merge_intervals([IV(e) for e in a + b]):
                prefill.append(
                    Ev(
                        key,
                        "ADJ",
                        (a + b)[0].value,
                        int(round(s * 1000)),
                        int(round(t * 1000)),
                    )
                )
            continue
        pairs = _match(a, b)
        ma, mb = {i for i, _, _ in pairs}, {j for _, j, _ in pairs}
        for i, j, _ in pairs:
            if a[i].value == b[j].value:
                prefill.append(
                    Ev(
                        key,
                        "ADJ",
                        a[i].value,
                        round((a[i].start_ms + b[j].start_ms) / 2),
                        round((a[i].end_ms + b[j].end_ms) / 2),
                    )
                )
            else:
                disagreements.append(
                    {
                        "tier": key,
                        "type": "value",
                        "a1": a[i].value,
                        "a2": b[j].value,
                        "start_s": IV(a[i])[0],
                    }
                )
        disagreements += [
            {
                "tier": key,
                "type": "only_A1",
                "a1": a[i].value,
                "a2": "",
                "start_s": IV(a[i])[0],
            }
            for i in range(len(a))
            if i not in ma
        ]
        disagreements += [
            {
                "tier": key,
                "type": "only_A2",
                "a1": "",
                "a2": b[j].value,
                "start_s": IV(b[j])[0],
            }
            for j in range(len(b))
            if j not in mb
        ]
    cov = {
        c: "complete"
        for c in ALL_CLASSES
        if pa.coverage(c) == "complete" and pb.coverage(c) == "complete"
    }
    tree = build_eaf(
        rec,
        ["A1", "A2", "ADJ"],
        media_name,
        pa.duration_ms,
        events=events + prefill,
        coverage=cov,
        guide_version=guide_version or pa.properties.get("guide_version", ""),
    )
    write_eaf(tree, out_path)
    return sorted(disagreements, key=lambda d: (d["start_s"], d["tier"]))


def adjudication_log(path_adj: Path) -> list[dict]:
    """Rows describing, per tier, where every final ADJ event came from and which annotator events were not kept."""
    p = parse_eaf(path_adj)
    rows: list[dict] = []
    for key in TIER_KEYS:
        f = [e for e in p.events if e.annotator == "ADJ" and e.tier == key]
        a = [e for e in p.events if e.annotator == "A1" and e.tier == key]
        b = [e for e in p.events if e.annotator == "A2" and e.tier == key]
        pa, pb = _match(f, a), _match(f, b)
        ma, mb = {i: j for i, j, _ in pa}, {i: j for i, j, _ in pb}
        for i, e in enumerate(f):
            ia, ib = ma.get(i), mb.get(i)
            same_a = ia is not None and a[ia].value == e.value
            same_b = ib is not None and b[ib].value == e.value
            origin = (
                "A1+A2 agree"
                if same_a and same_b
                else "A1 only"
                if same_a
                else "A2 only"
                if same_b
                else "new/modified by adjudicator"
            )
            rows.append(
                {
                    "tier": key,
                    "decision": "kept",
                    "origin": origin,
                    "value": e.value,
                    "start_ms": e.start_ms,
                    "end_ms": e.end_ms,
                }
            )
        used_a, used_b = set(ma.values()), set(mb.values())
        for j, e in enumerate(a):
            if j not in used_a:
                rows.append(
                    {
                        "tier": key,
                        "decision": "discarded",
                        "origin": "A1",
                        "value": e.value,
                        "start_ms": e.start_ms,
                        "end_ms": e.end_ms,
                    }
                )
        for j, e in enumerate(b):
            if j not in used_b:
                rows.append(
                    {
                        "tier": key,
                        "decision": "discarded",
                        "origin": "A2",
                        "value": e.value,
                        "start_ms": e.start_ms,
                        "end_ms": e.end_ms,
                    }
                )
    return rows


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        list(rows[0])
        if rows
        else ["tier", "decision", "origin", "value", "start_ms", "end_ms"]
    )
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
