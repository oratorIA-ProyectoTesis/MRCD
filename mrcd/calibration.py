"""Score calibration (Platt / isotonic), cross-fitting by groups, quantile-bin Brier/ECE and a three-state decision.

Honesty: a calibrator maps a score to an estimate learned on development data. The word "calibrated" may be used
for a score only when a data-separated check (cross-fitted, grouped) shows it; see ``S3/PROTOCOL.md`` section 4.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from mrcd.context_eval.partition import inner_splits

PRESENT, ABSENT, UNCERTAIN = "present", "absent", "uncertain"
EPS = 1e-4


def _logit(s):
    s = np.clip(np.asarray(s, float), EPS, 1 - EPS)
    return np.log(s / (1 - s))


@dataclass
class Calibrator:
    method: str  # "platt" | "isotonic" | "constant"
    params: dict

    def apply(self, s) -> np.ndarray:
        s = np.asarray(s, float)
        if self.method == "constant":
            return np.full(len(s), self.params["c"])
        if self.method == "platt":
            z = self.params["a"] * _logit(s) + self.params["b"]
            return 1 / (1 + np.exp(-z))
        return np.interp(s, self.params["x"], self.params["y"])

    def to_dict(self) -> dict:
        return {"method": self.method, "params": self.params}

    @classmethod
    def from_dict(cls, d: dict) -> "Calibrator":
        return cls(d["method"], d["params"])


def fit_calibrator(s, y, method: str, w=None) -> Calibrator:
    s = np.asarray(s, float)
    y = np.asarray(y).astype(int)
    w = None if w is None else np.asarray(w, float)
    if len(set(y.tolist())) < 2:
        return Calibrator(
            "constant", {"c": float(np.average(y, weights=w)) if len(y) else 0.0}
        )
    if method == "platt":
        lr = LogisticRegression(C=100.0, solver="lbfgs").fit(
            _logit(s).reshape(-1, 1), y, sample_weight=w
        )
        return Calibrator(
            "platt", {"a": float(lr.coef_[0, 0]), "b": float(lr.intercept_[0])}
        )
    if method == "isotonic":
        iso = IsotonicRegression(
            increasing=True, out_of_bounds="clip", y_min=0.0, y_max=1.0
        ).fit(s, y, sample_weight=w)
        return Calibrator(
            "isotonic",
            {
                "x": [float(v) for v in iso.X_thresholds_],
                "y": [float(v) for v in iso.y_thresholds_],
            },
        )
    raise ValueError(method)


def crossfit(
    s, y, groups, method: str, k: int = 3, seed: int = 13, w=None
) -> np.ndarray:
    """Out-of-group calibrated scores: each row is calibrated by a calibrator fitted on other groups only."""
    s = np.asarray(s, float)
    y = np.asarray(y).astype(int)
    g = np.asarray([str(a) for a in groups])
    out = np.full(len(s), np.nan)
    for _, tr, te in inner_splits(g, k, seed - 1):
        assert not (set(g[tr]) & set(g[te]))
        out[te] = fit_calibrator(
            s[tr], y[tr], method, None if w is None else np.asarray(w)[tr]
        ).apply(s[te])
    return out


# ------------------------------------------------------------------ metrics
def brier(y, p, w=None) -> float:
    y = np.asarray(y, float)
    return (
        float(np.average((np.asarray(p, float) - y) ** 2, weights=w))
        if len(y)
        else float("nan")
    )


def reliability_quantile(y, p, bins: int = 10, w=None):
    """Equal-mass bins: list of (n, weight, mean_score, frac_pos)."""
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    w = np.ones(len(y)) if w is None else np.asarray(w, float)
    order = np.argsort(p, kind="stable")
    out = []
    for ch in np.array_split(order, min(bins, max(len(y), 1))):
        if len(ch) == 0:
            continue
        ww = w[ch]
        out.append(
            (
                int(len(ch)),
                float(ww.sum()),
                float(np.average(p[ch], weights=ww)),
                float(np.average(y[ch], weights=ww)),
            )
        )
    return out


def ece_quantile(y, p, bins: int = 10, w=None) -> float:
    rel = reliability_quantile(y, p, bins, w)
    tot = sum(r[1] for r in rel)
    return (
        float(sum(r[1] * abs(r[2] - r[3]) for r in rel) / tot) if tot else float("nan")
    )


def paired_metric_delta(
    fn, y, a, b, g, w=None, n_boot: int = 2000, seed: int = 13
) -> dict:
    """fn(y,a,w)-fn(y,b,w) with speaker-bootstrap CI and two-sided p (floor 1/(B+1))."""
    y, a, b = np.asarray(y), np.asarray(a), np.asarray(b)
    w = np.ones(len(y)) if w is None else np.asarray(w, float)
    ug, inv = np.unique(np.asarray(g).astype(str), return_inverse=True)
    mem = [np.flatnonzero(inv == i) for i in range(len(ug))]
    d0 = fn(y, a, w) - fn(y, b, w)
    if len(ug) < 2:
        return {"delta": float(d0), "ci": [float("nan")] * 2, "p_boot": float("nan")}
    rng = np.random.default_rng(seed)
    ds = []
    for _ in range(n_boot):
        ii = np.concatenate([mem[k] for k in rng.integers(0, len(ug), len(ug))])
        ds.append(fn(y[ii], a[ii], w[ii]) - fn(y[ii], b[ii], w[ii]))
    ds = np.array(ds)
    B = len(ds)
    p = min(
        1.0, 2 * min((np.sum(ds <= 0) + 1) / (B + 1), (np.sum(ds >= 0) + 1) / (B + 1))
    )
    return {
        "delta": float(d0),
        "ci": [float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5))],
        "p_boot": float(p),
    }


# ------------------------------------------------------------------ three states
def three_state(p: float, lo: float, hi: float) -> str:
    if p != p:
        return UNCERTAIN
    if p >= hi:
        return PRESENT
    if p < lo:
        return ABSENT
    return UNCERTAIN


def states(p, lo: float, hi: float) -> np.ndarray:
    return np.array([three_state(v, lo, hi) for v in np.asarray(p, float)])


def selective_report(y, st, w=None) -> dict:
    """Coverage / risk of a three-state decision. Accepted = present or absent. Error = 0-1 error among accepted."""
    y = np.asarray(y).astype(int)
    st = np.asarray(st)
    w = np.ones(len(y)) if w is None else np.asarray(w, float)
    acc = st != UNCERTAIN
    pred = (st == PRESENT).astype(int)
    W, Wa = w.sum(), w[acc].sum()
    ap, aa = st == PRESENT, st == ABSENT
    tp = w[ap & (y == 1)].sum()
    fp = w[ap & (y == 0)].sum()
    fn_all = w[
        (pred == 0) & (y == 1)
    ].sum()  # uncertain and absent both count as "not present"
    wpos = w[y == 1].sum()
    return {
        "n": int(len(y)),
        "coverage": float(Wa / W) if W else float("nan"),
        "error_accepted": float(w[acc & (pred != y)].sum() / Wa)
        if Wa
        else float("nan"),
        "precision_present": float(tp / (tp + fp)) if tp + fp else float("nan"),
        "npv_absent": float(w[aa & (y == 0)].sum() / w[aa].sum())
        if w[aa].sum()
        else float("nan"),
        "pos_share_uncertain": float(w[(~acc) & (y == 1)].sum() / wpos)
        if wpos
        else float("nan"),
        "f1_all_cases_uncertain_as_absent": float(2 * tp / (2 * tp + fp + fn_all))
        if 2 * tp + fp + fn_all
        else float("nan"),
        "n_present": int(ap.sum()),
        "n_absent": int(aa.sum()),
        "n_uncertain": int((~acc).sum()),
    }


def choose_band(
    p, y, thr: float, w=None, rel: float = 0.75, step: float = 0.05
) -> dict:
    """Maximise coverage s.t. accepted error <= rel * baseline error (baseline: p >= thr, no abstention).

    Grid: lo / hi at 5% quantile steps of the scores below / above ``thr``. Development scores only.
    """
    p = np.asarray(p, float)
    y = np.asarray(y).astype(int)
    w = np.ones(len(y)) if w is None else np.asarray(w, float)
    base = selective_report(y, states(p, thr, thr), w)["error_accepted"]
    qs = np.arange(0.0, 1.0 + 1e-9, step)
    lo_set = {float(thr)} | (
        {float(v) for v in np.quantile(p[p < thr], qs)} if (p < thr).any() else set()
    )
    hi_set = {float(thr)} | (
        {float(v) for v in np.quantile(p[p >= thr], qs)} if (p >= thr).any() else set()
    )
    best = None
    for lo in sorted(lo_set):
        for hi in sorted(hi_set):
            if lo > thr or hi < thr:
                continue
            r = selective_report(y, states(p, lo, hi), w)
            if r["coverage"] > 0 and r["error_accepted"] <= rel * base + 1e-12:
                if best is None or r["coverage"] > best[0]:
                    best = (r["coverage"], lo, hi)
    if best is None:
        return {
            "lo": float(thr),
            "hi": float(thr),
            "feasible": False,
            "baseline_error": base,
        }
    return {
        "lo": best[1],
        "hi": best[2],
        "feasible": True,
        "baseline_error": base,
        "coverage_dev": best[0],
    }
