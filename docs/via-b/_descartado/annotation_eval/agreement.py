"""Inter-annotator agreement for two independent annotators, per class, with speaker-bootstrap intervals.

Three views, all computed on the independent (pre-adjudication) annotations:

* frame presence   : the audio is cut in ``agreement_frame_ms`` frames; each annotator marks a frame as event/no event.
* segment presence : non-overlapping ``segment_s`` segments, same presence rule as the confirmatory evaluation.
* event            : temporal matching (IoU >= ``primary_iou``), symmetric F1 between annotators, IoU of matched
                     pairs, and nominal kappa/alpha of the labels over the union of matched and unmatched events
                     (class-agnostic matching; true negatives are not units here, so it is a conservative figure).

Uncertain events and excluded regions are ignored (never counted as agreement or disagreement).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from mrcd.annotation_eval import metrics as M
from mrcd.annotation_eval.config import CONFIRMATORY as CFG
from mrcd.annotation_eval.eaf import Ev, parse_eaf
from mrcd.annotation_eval.schema import ALL_CLASSES, TIER_CLASSES, parse_recording_id


@dataclass
class RecPair:
    recording_id: str
    speaker: str
    duration_s: float
    a: list[Ev]
    b: list[Ev]
    classes: tuple[str, ...]  # classes declared complete by BOTH annotators
    phase: str = "T"


def segments(
    duration_s: float,
    seg_s: float = CFG["segment_s"],
    min_tail: float = CFG["segment_min_tail_s"],
):
    out, t = [], 0.0
    while t + seg_s <= duration_s + 1e-9:
        out.append((round(t, 6), round(t + seg_s, 6)))
        t += seg_s
    if duration_s - t >= min_tail:
        out.append((round(t, 6), round(duration_s, 6)))
    return out


def segment_positive(
    seg: tuple[float, float], events: Sequence[tuple[float, float]]
) -> bool:
    """Same rule as ``mrcd.dataset.labels_from_events``: overlap >= min(0.30 s, half of the event)."""
    rule = CFG["segment_overlap_rule"]
    for s, e in events:
        ov = M.overlap(seg, (s, e))
        if ov > 0 and ov >= min(
            rule["min_overlap_s"], rule["min_event_fraction"] * (e - s)
        ):
            return True
    return False


def _cls_events(
    evs: Sequence[Ev], cls: str, *, uncertain: bool = False
) -> list[tuple[float, float]]:
    out = []
    for e in evs:
        if e.tier == "excluir" or cls not in _classes_of(e):
            continue
        if e.uncertain == uncertain:
            out.append((e.start_ms / 1000, e.end_ms / 1000))
    return sorted(out)


def _classes_of(e: Ev) -> tuple[str, ...]:
    if e.uncertain:
        from mrcd.annotation_eval.schema import UNCERTAIN_CLASS

        return UNCERTAIN_CLASS[e.value]
    c = e.cls
    return (c,) if c else ()


def excluded_regions(
    evs_a: Sequence[Ev], evs_b: Sequence[Ev]
) -> list[tuple[float, float]]:
    return M.merge_intervals(
        [
            (e.start_ms / 1000, e.end_ms / 1000)
            for e in list(evs_a) + list(evs_b)
            if e.tier == "excluir"
        ]
    )


def _ignore_for_class(r: RecPair, cls: str) -> list[tuple[float, float]]:
    unc = _cls_events(r.a, cls, uncertain=True) + _cls_events(r.b, cls, uncertain=True)
    return M.merge_intervals(excluded_regions(r.a, r.b) + unc)


def _counts(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.array(
        [(x & y).sum(), (x & ~y).sum(), (~x & y).sum(), (~x & ~y).sum()], float
    )


def frame_counts(r: RecPair, cls: str) -> np.ndarray:
    ms = CFG["agreement_frame_ms"]
    a = M.presence_frames(_cls_events(r.a, cls), r.duration_s, ms)
    b = M.presence_frames(_cls_events(r.b, cls), r.duration_s, ms)
    keep = ~M.presence_frames(_ignore_for_class(r, cls), r.duration_s, ms)
    return _counts(a[keep], b[keep])


def segment_counts(r: RecPair, cls: str) -> np.ndarray:
    ea, eb, ign = (
        _cls_events(r.a, cls),
        _cls_events(r.b, cls),
        _ignore_for_class(r, cls),
    )
    xs, ys = [], []
    for seg in segments(r.duration_s):
        if M.fraction_inside(seg, ign) > CFG["segment_ignore_fraction"]:
            continue
        xs.append(segment_positive(seg, ea))
        ys.append(segment_positive(seg, eb))
    return _counts(np.array(xs, bool), np.array(ys, bool))


def event_counts(r: RecPair, cls: str) -> np.ndarray:
    """[matched, n_a, n_b, sum_iou] for certain events of ``cls``; events inside the other's uncertain region or an
    excluded region are dropped from the counts."""
    ea, eb = _cls_events(r.a, cls), _cls_events(r.b, cls)
    ex = excluded_regions(r.a, r.b)
    ua, ub = (
        _cls_events(r.a, cls, uncertain=True),
        _cls_events(r.b, cls, uncertain=True),
    )
    pairs = M.match_events(ea, eb, CFG["primary_iou"])
    ma, mb = {i for i, _, _ in pairs}, {j for _, j, _ in pairs}

    def dropped(iv, other_unc):
        return any(M.overlap(iv, x) > 0 for x in ex) or any(
            M.overlap(iv, x) > 0 for x in other_unc
        )

    na = sum(1 for i, iv in enumerate(ea) if i in ma or not dropped(iv, ub))
    nb = sum(1 for j, iv in enumerate(eb) if j in mb or not dropped(iv, ua))
    return np.array([len(pairs), na, nb, sum(p[2] for p in pairs)], float)


def event_units(r: RecPair, classes: Sequence[str]) -> list[tuple[str, str]]:
    """Class-agnostic one-to-one matching; unit = (label of A or 'none', label of B or 'none')."""
    ex = excluded_regions(r.a, r.b)
    ea = [
        (e, (e.start_ms / 1000, e.end_ms / 1000), c)
        for e in r.a
        if not e.uncertain and e.tier != "excluir"
        for c in [e.cls]
        if c in classes
    ]
    eb = [
        (e, (e.start_ms / 1000, e.end_ms / 1000), c)
        for e in r.b
        if not e.uncertain and e.tier != "excluir"
        for c in [e.cls]
        if c in classes
    ]
    pairs = M.match_events([x[1] for x in ea], [x[1] for x in eb], CFG["primary_iou"])
    ma, mb = {i for i, _, _ in pairs}, {j for _, j, _ in pairs}
    units = [(ea[i][2], eb[j][2]) for i, j, _ in pairs]
    units += [
        (c, "none")
        for i, (_, iv, c) in enumerate(ea)
        if i not in ma and not any(M.overlap(iv, x) > 0 for x in ex)
    ]
    units += [
        ("none", c)
        for j, (_, iv, c) in enumerate(eb)
        if j not in mb and not any(M.overlap(iv, x) > 0 for x in ex)
    ]
    return units


def _alpha_binary(c: np.ndarray) -> float | None:
    n11, n10, n01, n00 = c
    n = 2 * (n11 + n10 + n01 + n00)
    if n <= 1:
        return None
    d_o = 2 * (n10 + n01) / n
    n1, n0 = 2 * n11 + n10 + n01, 2 * n00 + n10 + n01
    d_e = 2 * n1 * n0 / (n * (n - 1))
    return None if d_e <= 0 else 1 - d_o / d_e


def _kappa(c: np.ndarray) -> float | None:
    return M.cohen_kappa(*c)["kappa"]


def _event_f1(c: np.ndarray) -> float | None:
    m, na, nb, _ = c
    return None if na + nb == 0 else 2 * m / (na + nb)


def _mean_iou(c: np.ndarray) -> float | None:
    return None if c[0] == 0 else c[3] / c[0]


def _by_speaker(pairs: Sequence[RecPair], fn, cls: str) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for r in pairs:
        if cls in r.classes:
            v = fn(r, cls)
            out[r.speaker] = out.get(r.speaker, 0) + v
    return out


def _ci(counts: dict, stat) -> dict:
    return M.speaker_bootstrap(
        counts, stat, B=CFG["bootstrap_B"], seed=CFG["seed"], level=CFG["ci_level"]
    )


def agreement_report(
    pairs: Sequence[RecPair], classes: Sequence[str] = ALL_CLASSES
) -> dict:
    """Per-class agreement with speaker-bootstrap CI and the quality gate decision."""
    out: dict = {
        "n_recordings": len(pairs),
        "n_speakers": len({p.speaker for p in pairs}),
        "classes": {},
    }
    for cls in classes:
        scope = [p for p in pairs if cls in p.classes]
        if not scope:
            out["classes"][cls] = {
                "status": "not_annotated_by_both",
                "passes_gate": None,
            }
            continue
        seg = _by_speaker(pairs, segment_counts, cls)
        frm = _by_speaker(pairs, frame_counts, cls)
        evc = _by_speaker(pairs, event_counts, cls)
        tot_seg, tot_evt = sum(seg.values()), sum(evc.values())
        row = {
            "n_recordings": len(scope),
            "n_speakers": len({p.speaker for p in scope}),
            "segment": {
                "counts_11_10_01_00": [int(x) for x in tot_seg],
                "cohen_kappa": _ci(seg, _kappa),
                "krippendorff_alpha": _ci(seg, _alpha_binary),
                "pabak": 2 * M.cohen_kappa(*tot_seg)["po"] - 1
                if tot_seg.sum()
                else None,
            },
            "frame": {
                "frame_ms": CFG["agreement_frame_ms"],
                "cohen_kappa": _ci(frm, _kappa),
                "krippendorff_alpha": _ci(frm, _alpha_binary),
            },
            "event": {
                "matched": int(tot_evt[0]),
                "n_annotator_a": int(tot_evt[1]),
                "n_annotator_b": int(tot_evt[2]),
                "iou_threshold": CFG["primary_iou"],
                "f1_between_annotators": _ci(evc, _event_f1),
                "mean_iou_matched": _ci(evc, _mean_iou),
            },
            "uncertain": {
                "a": sum(len(_cls_events(p.a, cls, uncertain=True)) for p in scope),
                "b": sum(len(_cls_events(p.b, cls, uncertain=True)) for p in scope),
            },
        }
        units = [u for p in scope for u in event_units(p, [cls])]
        lab = [[(0 if x == "none" else 1), (0 if y == "none" else 1)] for x, y in units]
        row["event"]["label_alpha_binary_over_events"] = M.krippendorff_alpha_nominal(
            lab
        )["alpha"]
        row["gate"] = quality_gate(row)
        row["passes_gate"] = row["gate"]["passes"]
        out["classes"][cls] = row
    all_units = [
        u for p in pairs for u in event_units(p, [c for c in classes if c in p.classes])
    ]
    cats = {c: i for i, c in enumerate(sorted({x for u in all_units for x in u}))}
    out["event_labels_all_classes"] = {
        "n_units": len(all_units),
        "krippendorff_alpha_nominal": M.krippendorff_alpha_nominal(
            [[cats[x], cats[y]] for x, y in all_units]
        )["alpha"],
        "note": "true negatives are not units; conservative; includes unmatched events as 'none'",
    }
    return out


def quality_gate(row: dict) -> dict:
    """Gate on POINT estimates of the independent annotations (CI reported alongside)."""
    k = row["segment"]["cohen_kappa"]["point"]
    a = row["segment"]["krippendorff_alpha"]["point"]
    f = row["event"]["f1_between_annotators"]["point"]
    n_events = max(row["event"]["n_annotator_a"], row["event"]["n_annotator_b"])
    checks = {
        "segment_kappa": (k, CFG["min_kappa"]),
        "segment_alpha": (a, CFG["min_alpha"]),
        "event_f1": (f, CFG["min_event_f1"]),
    }
    if any(v is None for v, _ in checks.values()) or n_events < 10:
        return {
            "passes": None,
            "reason": "insufficient_data (fewer than 10 events or undefined statistic)",
            "checks": checks,
        }
    failed = [name for name, (v, thr) in checks.items() if v < thr]
    return {
        "passes": not failed,
        "failed": failed,
        "reached_reference_0_70": all(
            v >= CFG["reference_agreement"] for v, _ in checks.values()
        ),
        "checks": {n: {"value": v, "threshold": t} for n, (v, t) in checks.items()},
        "action": "none"
        if not failed
        else "recalibrate guide on practice material; if still failing, class is NOT evaluated",
    }


def load_pair(path_a: str | Path, path_b: str | Path) -> RecPair:
    pa, pb = parse_eaf(path_a), parse_eaf(path_b)
    rec = pa.properties.get("recording_id", "")
    if rec != pb.properties.get("recording_id"):
        raise ValueError(f"recording_id differs between {path_a} and {path_b}")
    if pa.duration_ms != pb.duration_ms or pa.duration_ms is None:
        raise ValueError(f"audio_duration_ms differs or is missing for {rec}")
    info = parse_recording_id(rec)
    classes = tuple(
        c
        for c in ALL_CLASSES
        if pa.coverage(c) == "complete" and pb.coverage(c) == "complete"
    )
    return RecPair(
        rec,
        info["speaker"],
        pa.duration_ms / 1000,
        pa.events,
        pb.events,
        classes,
        info["phase"],
    )


def agreement_from_dirs(
    dir_a: str | Path, dir_b: str | Path, classes: Sequence[str] = ALL_CLASSES
) -> dict:
    fa = {p.stem: p for p in Path(dir_a).glob("*.eaf")}
    fb = {p.stem: p for p in Path(dir_b).glob("*.eaf")}
    missing = sorted(set(fa) ^ set(fb))
    pairs = [load_pair(fa[k], fb[k]) for k in sorted(set(fa) & set(fb))]
    rep = agreement_report(pairs, classes)
    rep["unpaired_files"] = missing
    rep["phases"] = sorted({p.phase for p in pairs})
    return rep


__all__ = [
    "RecPair",
    "agreement_report",
    "agreement_from_dirs",
    "load_pair",
    "segments",
    "segment_positive",
    "TIER_CLASSES",
]
