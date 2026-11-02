"""Pure metric helpers: temporal matching, kappa/alpha, speaker bootstrap, Holm. No I/O, no engine imports."""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

Interval = tuple[float, float]


def iou(a: Interval, b: Interval) -> float:
    inter = min(a[1], b[1]) - max(a[0], b[0])
    if inter <= 0:
        return 0.0
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return inter / union if union > 0 else 0.0


def overlap(a: Interval, b: Interval) -> float:
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))


def match_events(
    ref: Sequence[Interval],
    hyp: Sequence[Interval],
    thr: float = 0.5,
    *,
    any_overlap: bool = False,
) -> list[tuple[int, int, float]]:
    """One-to-one matching maximising the summed IoU among pairs with IoU >= thr (or any overlap).

    Returns (ref index, hyp index, iou).
    """
    if not len(ref) or not len(hyp):
        return []
    m = np.array([[iou(r, h) for h in hyp] for r in ref])
    ok = m > 0 if any_overlap else m >= thr - 1e-12
    if not ok.any():
        return []
    rows, cols = linear_sum_assignment(-(m * ok))
    return [(int(r), int(c), float(m[r, c])) for r, c in zip(rows, cols) if ok[r, c]]


def prf(
    tp: float, fp: float, fn: float
) -> tuple[float | None, float | None, float | None]:
    """Precision, recall, F1; None where the denominator is zero (absence is never reported as 0)."""
    p = tp / (tp + fp) if (tp + fp) > 0 else None
    r = tp / (tp + fn) if (tp + fn) > 0 else None
    f = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else None
    return p, r, f


def cohen_kappa(n11: float, n10: float, n01: float, n00: float) -> dict:
    """Cohen's kappa for two coders and a binary category from the 2x2 counts (coder A rows, coder B columns)."""
    n = n11 + n10 + n01 + n00
    if n <= 0:
        return {
            "kappa": None,
            "po": None,
            "pe": None,
            "pabak": None,
            "n": 0,
            "reason": "no_units",
        }
    po = (n11 + n00) / n
    pa1, pb1 = (n11 + n10) / n, (n11 + n01) / n
    pe = pa1 * pb1 + (1 - pa1) * (1 - pb1)
    kappa = None if pe >= 1 - 1e-12 else (po - pe) / (1 - pe)
    return {
        "kappa": kappa,
        "po": po,
        "pe": pe,
        "pabak": 2 * po - 1,
        "n": int(n),
        "reason": None if kappa is not None else "no_variation",
    }


def krippendorff_alpha_nominal(units: Sequence[Sequence[int | None]]) -> dict:
    """Krippendorff's alpha for nominal data; each unit lists the labels of its coders (None = missing).

    Units with fewer than two ratings are not pairable and are dropped, as in the original definition.
    """
    cats = sorted({v for u in units for v in u if v is not None})
    idx = {c: i for i, c in enumerate(cats)}
    k = len(cats)
    o = np.zeros((k, k))
    for u in units:
        vals = [idx[v] for v in u if v is not None]
        m = len(vals)
        if m < 2:
            continue
        counts = np.bincount(vals, minlength=k).astype(float)
        pair = np.outer(counts, counts) - np.diag(
            counts
        )  # ordered pairs of values within the unit
        o += pair / (m - 1)
    n = o.sum()
    if n <= 1 or k < 2:
        return {
            "alpha": None,
            "n_pairable": float(n),
            "reason": "no_variation" if n > 1 else "no_pairable_units",
        }
    nc = o.sum(1)
    d_o = (o.sum() - np.trace(o)) / n
    d_e = (np.outer(nc, nc).sum() - (nc**2).sum()) / (n * (n - 1))
    if d_e <= 0:
        return {"alpha": None, "n_pairable": float(n), "reason": "no_variation"}
    return {"alpha": float(1 - d_o / d_e), "n_pairable": float(n), "reason": None}


def _percentile_ci(vals: np.ndarray, level: float) -> tuple[float, float]:
    a = (1 - level) / 2
    lo, hi = np.percentile(vals, [100 * a, 100 * (1 - a)])
    return float(lo), float(hi)


def speaker_bootstrap(
    counts: dict[str, np.ndarray],
    statistic: Callable[[np.ndarray], float | None],
    *,
    B: int = 2000,
    seed: int = 0,
    level: float = 0.95,
) -> dict:
    """Percentile CI resampling SPEAKERS with replacement. ``counts[speaker]`` is an additive vector of sufficient
    statistics; ``statistic`` maps the summed vector to a number (or None when undefined)."""
    keys = sorted(counts)
    if not keys:
        return {"point": None, "lo": None, "hi": None, "n_speakers": 0, "n_valid": 0}
    S = np.array([np.asarray(counts[k], float) for k in keys])
    point = statistic(S.sum(0))
    rng = np.random.default_rng(seed)
    W = rng.multinomial(len(keys), np.full(len(keys), 1.0 / len(keys)), size=B).astype(
        float
    )
    sums = W @ S
    vals = np.array(
        [v for v in (statistic(s) for s in sums) if v is not None and np.isfinite(v)]
    )
    if len(vals) < max(20, B // 10):
        return {
            "point": point,
            "lo": None,
            "hi": None,
            "n_speakers": len(keys),
            "n_valid": int(len(vals)),
        }
    lo, hi = _percentile_ci(vals, level)
    return {
        "point": point,
        "lo": lo,
        "hi": hi,
        "n_speakers": len(keys),
        "n_valid": int(len(vals)),
    }


def paired_speaker_bootstrap(
    counts_a: dict[str, np.ndarray],
    counts_b: dict[str, np.ndarray],
    statistic: Callable[[np.ndarray], float | None],
    *,
    B: int = 2000,
    seed: int = 0,
    level: float = 0.95,
) -> dict:
    """Paired difference statistic(A) - statistic(B), resampling the SAME speakers for both systems.

    Two-sided p-value = 2 * min(P(d <= 0), P(d >= 0)) with the +1 correction (bootstrap sign test).
    """
    if set(counts_a) != set(counts_b):
        raise ValueError("paired comparison needs the same speakers in both systems")
    keys = sorted(counts_a)
    if not keys:
        return {"diff": None, "lo": None, "hi": None, "p": None, "n_speakers": 0}
    A = np.array([np.asarray(counts_a[k], float) for k in keys])
    Bm = np.array([np.asarray(counts_b[k], float) for k in keys])

    def d(sa, sb):
        x, y = statistic(sa), statistic(sb)
        return None if x is None or y is None else x - y

    point = d(A.sum(0), Bm.sum(0))
    rng = np.random.default_rng(seed)
    W = rng.multinomial(len(keys), np.full(len(keys), 1.0 / len(keys)), size=B).astype(
        float
    )
    sa, sb = W @ A, W @ Bm
    vals = np.array(
        [
            v
            for v in (d(x, y) for x, y in zip(sa, sb))
            if v is not None and np.isfinite(v)
        ]
    )
    if point is None or len(vals) < max(20, B // 10):
        return {
            "diff": point,
            "lo": None,
            "hi": None,
            "p": None,
            "n_speakers": len(keys),
        }
    lo, hi = _percentile_ci(vals, level)
    p_le = (1 + np.sum(vals <= 0)) / (len(vals) + 1)
    p_ge = (1 + np.sum(vals >= 0)) / (len(vals) + 1)
    return {
        "diff": float(point),
        "lo": lo,
        "hi": hi,
        "p": float(min(1.0, 2 * min(p_le, p_ge))),
        "n_speakers": len(keys),
    }


def holm(pvalues: Sequence[float | None]) -> list[float | None]:
    """Holm step-down adjusted p-values over the computable tests; None stays None."""
    idx = [i for i, p in enumerate(pvalues) if p is not None]
    m = len(idx)
    out: list[float | None] = [None] * len(pvalues)
    order = sorted(idx, key=lambda i: pvalues[i])
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvalues[i]))
        out[i] = running
    return out


def merge_intervals(ivs: Sequence[Interval]) -> list[Interval]:
    out: list[list[float]] = []
    for s, e in sorted(ivs):
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [(a, b) for a, b in out]


def fraction_inside(seg: Interval, regions: Sequence[Interval]) -> float:
    dur = seg[1] - seg[0]
    if dur <= 0:
        return 0.0
    return sum(overlap(seg, r) for r in merge_intervals(regions)) / dur


def presence_frames(
    events: Sequence[Interval], duration_s: float, frame_ms: int
) -> np.ndarray:
    n = int(round(duration_s * 1000 / frame_ms))
    v = np.zeros(n, bool)
    for s, e in events:
        a, b = int(round(s * 1000 / frame_ms)), int(round(e * 1000 / frame_ms))
        v[max(a, 0) : min(max(b, a + 1), n)] = True
    return v
