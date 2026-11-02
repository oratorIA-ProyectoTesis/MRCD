"""Small statistics helpers for the technical-validation kit: segment grid, Wilson interval, speaker bootstrap."""

from __future__ import annotations

import math
from statistics import NormalDist
from typing import Callable

import numpy as np

SEGMENT_S = 3.0
SEGMENT_MIN_TAIL_S = 1.5


def segments(
    duration_s: float, seg_s: float = SEGMENT_S, min_tail: float = SEGMENT_MIN_TAIL_S
) -> list[tuple[float, float]]:
    """Non-overlapping segments of ``seg_s`` seconds; a final shorter piece is kept only if >= ``min_tail`` seconds."""
    out, t = [], 0.0
    while t + seg_s <= duration_s + 1e-9:
        out.append((round(t, 6), round(t + seg_s, 6)))
        t += seg_s
    if duration_s - t >= min_tail:
        out.append((round(t, 6), round(duration_s, 6)))
    return out


def wilson(k: float, n: float, level: float = 0.95) -> dict:
    """Wilson interval for a proportion (no speaker clustering: use ``speaker_bootstrap`` for clustered data)."""
    if n <= 0:
        return {"point": None, "lo": None, "hi": None, "n": 0}
    z = NormalDist().inv_cdf(0.5 + level / 2)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return {"point": p, "lo": c - h, "hi": c + h, "n": int(n)}


def speaker_bootstrap(
    counts: dict[str, np.ndarray],
    statistic: Callable[[np.ndarray], float | None],
    *,
    B: int = 2000,
    seed: int = 0,
    level: float = 0.95,
) -> dict:
    """Percentile CI resampling SPEAKERS with replacement.

    ``counts[speaker]`` is an additive vector of sufficient statistics (e.g. [admitted, total]); ``statistic`` maps
    the summed vector to a number or None when undefined. With few speakers the interval is wide: that is the point.
    """
    keys = sorted(counts)
    if not keys:
        return {"point": None, "lo": None, "hi": None, "n_speakers": 0, "n_valid": 0}
    S = np.array([np.asarray(counts[k], float) for k in keys])
    point = statistic(S.sum(0))
    rng = np.random.default_rng(seed)
    W = rng.multinomial(len(keys), np.full(len(keys), 1.0 / len(keys)), size=B).astype(
        float
    )
    vals = np.array(
        [v for v in (statistic(s) for s in W @ S) if v is not None and np.isfinite(v)]
    )
    if len(vals) < max(20, B // 10):
        return {
            "point": point,
            "lo": None,
            "hi": None,
            "n_speakers": len(keys),
            "n_valid": int(len(vals)),
        }
    a = (1 - level) / 2
    lo, hi = np.percentile(vals, [100 * a, 100 * (1 - a)])
    return {
        "point": point,
        "lo": float(lo),
        "hi": float(hi),
        "n_speakers": len(keys),
        "n_valid": int(len(vals)),
    }


def quantiles(values, qs=(0.5, 0.95)) -> dict:
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)], float)
    if not v.size:
        return {"n": 0, **{f"p{int(q * 100)}": None for q in qs}}
    return {
        "n": int(v.size),
        **{f"p{int(q * 100)}": float(np.quantile(v, q)) for q in qs},
    }


def cluster_halfwidth(p: float, n_speakers: int, items_per_speaker: int, icc: float, level: float = 0.95) -> float:
    """Approximate CI half-width of a proportion measured on ``items_per_speaker`` items of ``n_speakers`` speakers,
    with intraclass correlation ``icc`` (design effect 1 + (m - 1) icc). Planning aid only."""
    n = n_speakers * items_per_speaker
    deff = 1 + (items_per_speaker - 1) * icc
    z = NormalDist().inv_cdf(0.5 + level / 2)
    return z * math.sqrt(p * (1 - p) * deff / n)
