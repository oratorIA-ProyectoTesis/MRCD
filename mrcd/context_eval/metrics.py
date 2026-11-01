"""Confusion metrics, speaker bootstrap, paired differences, Holm, Brier / ECE / reliability."""

from __future__ import annotations

import numpy as np


def prf(tp: int, fp: int, fn: int, tn: int) -> dict:
    nan = float("nan")
    p = tp / (tp + fp) if tp + fp else nan
    r = tp / (tp + fn) if tp + fn else nan
    f = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else nan
    return {
        "n": int(tp + fp + fn + tn),
        "support_pos": int(tp + fn),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "precision": p,
        "recall": r,
        "f1": f,
    }


def confusion(y, p) -> dict:
    y = np.asarray(y).astype(int)
    p = np.asarray(p).astype(int)
    if y.shape != p.shape:
        raise ValueError("y and p must have the same shape")
    return prf(
        int(((y == 1) & (p == 1)).sum()),
        int(((y == 0) & (p == 1)).sum()),
        int(((y == 1) & (p == 0)).sum()),
        int(((y == 0) & (p == 0)).sum()),
    )


def group_counts(y, p, g) -> np.ndarray:
    """(G, 3) array of tp, fp, fn per group (sorted group order)."""
    y = np.asarray(y).astype(int)
    p = np.asarray(p).astype(int)
    _, inv = np.unique(np.asarray(g).astype(str), return_inverse=True)
    C = np.zeros((int(inv.max()) + 1 if len(inv) else 0, 3))
    np.add.at(C[:, 0], inv, (y == 1) & (p == 1))
    np.add.at(C[:, 1], inv, (y == 0) & (p == 1))
    np.add.at(C[:, 2], inv, (y == 1) & (p == 0))
    return C


def f1_from_counts(c) -> np.ndarray:
    c = np.asarray(c, float)
    den = 2 * c[..., 0] + c[..., 1] + c[..., 2]
    return np.where(den > 0, 2 * c[..., 0] / np.maximum(den, 1), np.nan)


def boot_f1_ci(y, p, g, n_boot: int = 2000, seed: int = 13):
    """95% percentile CI of F1 resampling groups (speakers). NaN pair when undefined / one group."""
    C = group_counts(y, p, g)
    if len(C) < 2:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(C), (n_boot, len(C)))
    f = f1_from_counts(C[idx].sum(1))
    if not np.isfinite(f).any():
        return [float("nan"), float("nan")]
    return [float(np.nanpercentile(f, 2.5)), float(np.nanpercentile(f, 97.5))]


def paired_f1_delta(y, p1, p0, g, n_boot: int = 5000, seed: int = 13) -> dict:
    """F1(p1) - F1(p0) on the same units; speaker bootstrap; two-sided bootstrap p (floor 1/(B+1))."""
    y, p1, p0, g = (np.asarray(a) for a in (y, p1, p0, g))
    if not (len(y) == len(p1) == len(p0) == len(g)):
        raise ValueError("paired comparison needs identical units")
    C1, C0 = group_counts(y, p1, g), group_counts(y, p0, g)
    d0 = (
        float(f1_from_counts(C1.sum(0)) - f1_from_counts(C0.sum(0)))
        if len(C1)
        else float("nan")
    )
    nan2 = [float("nan")] * 2
    if len(C1) < 2:
        return {
            "delta": d0,
            "ci": nan2,
            "p_boot": float("nan"),
            "n_groups": int(len(C1)),
            "n_boot": 0,
        }
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(C1), (n_boot, len(C1)))
    d = f1_from_counts(C1[idx].sum(1)) - f1_from_counts(C0[idx].sum(1))
    d = d[np.isfinite(d)]
    if len(d) == 0:
        return {
            "delta": d0,
            "ci": nan2,
            "p_boot": float("nan"),
            "n_groups": int(len(C1)),
            "n_boot": 0,
        }
    B = len(d)
    p = min(
        1.0, 2 * min((np.sum(d <= 0) + 1) / (B + 1), (np.sum(d >= 0) + 1) / (B + 1))
    )
    return {
        "delta": d0,
        "ci": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
        "p_boot": float(p),
        "n_groups": int(len(C1)),
        "n_boot": int(B),
    }


def holm(pvals) -> list[float]:
    """Holm step-down adjusted p-values in input order (NaN stays NaN and is not counted)."""
    p = np.asarray(pvals, float)
    ok = np.flatnonzero(np.isfinite(p))
    out = np.full(len(p), np.nan)
    m = len(ok)
    order = ok[np.argsort(p[ok], kind="stable")]
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (m - rank) * p[i]))
        out[i] = run
    return out.tolist()


def brier(y, s) -> float:
    y = np.asarray(y, float)
    s = np.asarray(s, float)
    return float(np.mean((s - y) ** 2)) if len(y) else float("nan")


def reliability(y, s, bins: int = 10):
    """Equal-width bins on [0,1]: list of (lo, hi, n, mean_score, frac_pos)."""
    y = np.asarray(y, float)
    s = np.clip(np.asarray(s, float), 0, 1)
    edges = np.linspace(0, 1, bins + 1)
    b = np.minimum(np.digitize(s, edges[1:-1]), bins - 1)
    out = []
    for i in range(bins):
        m = b == i
        n = int(m.sum())
        out.append(
            (
                float(edges[i]),
                float(edges[i + 1]),
                n,
                float(s[m].mean()) if n else float("nan"),
                float(y[m].mean()) if n else float("nan"),
            )
        )
    return out


def ece(y, s, bins: int = 10) -> float:
    rel = reliability(y, s, bins)
    N = sum(r[2] for r in rel)
    if N == 0:
        return float("nan")
    return float(sum(r[2] * abs(r[3] - r[4]) for r in rel if r[2]) / N)


def boot_diff(fn, y, a, b, g, n_boot: int = 2000, seed: int = 13):
    """Speaker-bootstrap CI of fn(y,a) - fn(y,b) (e.g. Brier or ECE raw vs calibrated)."""
    y, a, b, g = (np.asarray(x) for x in (y, a, b, g))
    ug, inv = np.unique(g.astype(str), return_inverse=True)
    members = [np.flatnonzero(inv == i) for i in range(len(ug))]
    d0 = fn(y, a) - fn(y, b)
    if len(ug) < 2:
        return {"delta": float(d0), "ci": [float("nan")] * 2}
    rng = np.random.default_rng(seed)
    ds = []
    for _ in range(n_boot):
        ii = np.concatenate([members[k] for k in rng.integers(0, len(ug), len(ug))])
        ds.append(fn(y[ii], a[ii]) - fn(y[ii], b[ii]))
    ds = np.array(ds)
    return {
        "delta": float(d0),
        "ci": [float(np.nanpercentile(ds, 2.5)), float(np.nanpercentile(ds, 97.5))],
    }
