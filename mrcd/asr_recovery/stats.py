"""Weighted confusion, speaker bootstrap, paired deltas, WER, survival-rate CIs."""

from __future__ import annotations

import numpy as np

from mrcd.context_eval.metrics import prf
from mrcd.context_eval.reference import _norm_tokens as norm_tokens


def wcounts(y, p, w):
    y = np.asarray(y).astype(int)
    p = np.asarray(p).astype(int)
    w = np.asarray(w, float)
    return (
        float(w[(y == 1) & (p == 1)].sum()),
        float(w[(y == 0) & (p == 1)].sum()),
        float(w[(y == 1) & (p == 0)].sum()),
        float(w[(y == 0) & (p == 0)].sum()),
    )


def wconfusion(y, p, w=None) -> dict:
    """Raw counts and weighted P/R/F1 (weights = inverse sampling fractions; ones give the plain metrics)."""
    y = np.asarray(y).astype(int)
    p = np.asarray(p).astype(int)
    w = np.ones(len(y)) if w is None else np.asarray(w, float)
    out = prf(
        int(((y == 1) & (p == 1)).sum()),
        int(((y == 0) & (p == 1)).sum()),
        int(((y == 1) & (p == 0)).sum()),
        int(((y == 0) & (p == 0)).sum()),
    )
    tp, fp, fn, tn = wcounts(y, p, w)
    nan = float("nan")
    out["w_tp"], out["w_fp"], out["w_fn"], out["w_tn"] = tp, fp, fn, tn
    out["w_precision"] = tp / (tp + fp) if tp + fp else nan
    out["w_recall"] = tp / (tp + fn) if tp + fn else nan
    out["w_f1"] = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else nan
    return out


def _gcounts(y, p, g, w):
    y = np.asarray(y).astype(int)
    p = np.asarray(p).astype(int)
    w = np.asarray(w, float)
    _, inv = np.unique(np.asarray(g).astype(str), return_inverse=True)
    C = np.zeros((int(inv.max()) + 1 if len(inv) else 0, 3))
    np.add.at(C[:, 0], inv, w * ((y == 1) & (p == 1)))
    np.add.at(C[:, 1], inv, w * ((y == 0) & (p == 1)))
    np.add.at(C[:, 2], inv, w * ((y == 1) & (p == 0)))
    return C


def _f1(c):
    c = np.asarray(c, float)
    den = 2 * c[..., 0] + c[..., 1] + c[..., 2]
    return np.where(den > 0, 2 * c[..., 0] / np.where(den > 0, den, 1), np.nan)


def boot_wf1_ci(y, p, g, w=None, n_boot: int = 2000, seed: int = 13):
    w = np.ones(len(y)) if w is None else w
    C = _gcounts(y, p, g, w)
    if len(C) < 2:
        return [float("nan")] * 2
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(C), (n_boot, len(C)))
    f = _f1(C[idx].sum(1))
    if not np.isfinite(f).any():
        return [float("nan")] * 2
    return [float(np.nanpercentile(f, 2.5)), float(np.nanpercentile(f, 97.5))]


def paired_wf1_delta(y, p1, p0, g, w=None, n_boot: int = 5000, seed: int = 13) -> dict:
    """Weighted F1(p1) - F1(p0) on the same units; speaker bootstrap; two-sided bootstrap p (floor 1/(B+1))."""
    y, p1, p0, g = (np.asarray(a) for a in (y, p1, p0, g))
    if not (len(y) == len(p1) == len(p0) == len(g)):
        raise ValueError("paired comparison needs identical units")
    w = np.ones(len(y)) if w is None else np.asarray(w, float)
    C1, C0 = _gcounts(y, p1, g, w), _gcounts(y, p0, g, w)
    nan2 = [float("nan")] * 2
    d0 = float(_f1(C1.sum(0)) - _f1(C0.sum(0))) if len(C1) else float("nan")
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
    d = _f1(C1[idx].sum(1)) - _f1(C0[idx].sum(1))
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


def rate_ci(s, g, n_boot: int = 2000, seed: int = 13) -> dict:
    """Mean of a numeric vector with speaker-cluster bootstrap CI."""
    s = np.asarray(s, float)
    g = np.asarray(g).astype(str)
    _, inv = np.unique(g, return_inverse=True)
    k = int(inv.max()) + 1 if len(inv) else 0
    num = np.zeros(k)
    den = np.zeros(k)
    np.add.at(num, inv, s)
    np.add.at(den, inv, 1.0)
    n = float(den.sum())
    rate = float(num.sum() / n) if n else float("nan")
    if k < 2:
        return {
            "n": int(n),
            "k": float(s.sum()),
            "rate": rate,
            "ci": [float("nan")] * 2,
        }
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, k, (n_boot, k))
    r = num[idx].sum(1) / np.maximum(den[idx].sum(1), 1)
    return {
        "n": int(n),
        "k": float(s.sum()),
        "rate": rate,
        "ci": [float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))],
    }


def paired_rate_delta(s1, s0, g, n_boot: int = 2000, seed: int = 13) -> dict:
    """mean(s1) - mean(s0) on the same units, with gained / lost counts and speaker-cluster CI."""
    s1 = np.asarray(s1, float)
    s0 = np.asarray(s0, float)
    d = rate_ci(s1 - s0, g, n_boot, seed)
    return {
        "n": d["n"],
        "gained": int(((s1 == 1) & (s0 == 0)).sum()),
        "lost": int(((s1 == 0) & (s0 == 1)).sum()),
        "delta": d["rate"],
        "ci": d["ci"],
    }


def edit_distance(a: list, b: list) -> int:
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, yv in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != yv)))
        prev = cur
    return prev[-1]


def wer(ref: str, hyp: str) -> tuple[int, int]:
    """(edit distance, reference length) over normalised tokens."""
    r, h = norm_tokens(ref), norm_tokens(hyp)
    return edit_distance(r, h), len(r)


def corpus_wer(pairs) -> float:
    e = n = 0
    for ref, hyp in pairs:
        a, b = wer(ref, hyp)
        e += a
        n += b
    return e / n if n else float("nan")


def prompt_echo(text: str, prompt: str, n: int = 4) -> bool:
    t, p = norm_tokens(text), norm_tokens(prompt)
    grams = {tuple(p[i : i + n]) for i in range(len(p) - n + 1)}
    return any(tuple(t[i : i + n]) in grams for i in range(len(t) - n + 1))
