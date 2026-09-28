"""Event evaluation, human agreement, calibration and usability scoring.

Events are dicts with integer `start_ms`/`end_ms` ([start, end)), `label`, optional
`recording_id`, `speaker_id` and `decision` ("event" | "uncertain"). Protocol values
(IoU, tolerance, classes) are arguments fixed before evaluation, never tuned on test.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment, minimize_scalar
from sklearn.metrics import cohen_kappa_score

from core.constants import TAXONOMY

PAUSES = ("rhetorical_pause", "neutral_pause")
DISFLUENCIES = tuple(c for c in TAXONOMY if c not in PAUSES)
DURATION_BINS_MS = (0, 300, 1000, 10**9)


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
    mid = (ev["start_ms"] + ev["end_ms"]) / 2
    return any(a <= mid < b for a, b in coverage.get(ev.get("recording_id"), []))


def _prf(tp: int, fp: int, fn: int) -> dict:
    p = tp / (tp + fp) if tp + fp else None
    r = tp / (tp + fn) if tp + fn else None
    f = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else None
    return {"tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": r, "f1": f, "support": tp + fn}


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
    keep = lambda e: e["label"] in classes and in_coverage(e, coverage)  # noqa: E731
    uncertain = [h for h in hyp if h.get("decision") == "uncertain" and keep(h)]
    ref = [e for e in ref if e.get("decision", "event") == "event" and keep(e)]
    hyp = [e for e in hyp if e.get("decision", "event") == "event" and keep(e)]
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
        }
    tol_counts = _counts(ref, hyp, match(ref, hyp, by_tolerance(tolerance_ms)), classes)
    fp_events = [h for j, h in enumerate(hyp) if j not in {j for _, j in pairs}]
    return {
        "protocol": {
            "iou": iou_threshold,
            "tolerance_ms": tolerance_ms,
            "classes": list(classes),
            "matching": "one_to_one_same_class_max_cardinality",
        },
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


def bootstrap_ci(ref: list[dict], hyps: list[list[dict]], stat, *, n: int = 1000, seed: int = 13) -> dict:
    """Percentile CI resampling speakers with their events as one cluster.
    `stat(ref, *hyps)` returns a float; pass two hyps for a paired difference."""
    speakers = sorted({e.get("speaker_id") for ev in (ref, *hyps) for e in ev}, key=str)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n):
        draw = rng.choice(len(speakers), len(speakers))

        def pick(events):
            return [
                {**e, "recording_id": f"{e.get('recording_id')}#{k}"}
                for k, s in enumerate(draw)
                for e in events
                if e.get("speaker_id") == speakers[s]
            ]

        v = stat(pick(ref), *map(pick, hyps))
        if v is not None:
            values.append(v)
    lo, hi = np.percentile(values, [2.5, 97.5]) if values else (None, None)
    return {"low": lo, "high": hi, "n_boot": len(values), "n_speakers": len(speakers)}


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
