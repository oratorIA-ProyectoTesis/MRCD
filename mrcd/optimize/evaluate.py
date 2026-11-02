"""Pooled outer-fold metrics, paired comparisons with Holm, coverage/risk with speaker-bootstrap CI, promotion rule."""

from __future__ import annotations

import numpy as np

from mrcd.asr_recovery import stats as S
from mrcd.calibration import selective_report
from mrcd.context_eval.metrics import holm

PRESENT, ABSENT, UNCERTAIN = "present", "absent", "uncertain"


def describe(y, p, g, w=None, n_boot: int = 2000, seed: int = 13) -> dict:
    """Weighted confusion (weights default to 1) + speaker-bootstrap CI of the weighted F1."""
    y, p, g = np.asarray(y), np.asarray(p), np.asarray(g)
    m = S.wconfusion(y, p, w)
    m["n_groups"] = int(len(set(g.tolist())))
    m["n_pos_groups"] = int(len({a for a, b in zip(g.tolist(), y.tolist()) if b}))
    m["f1_ci95"] = S.boot_wf1_ci(y, p, g, w, n_boot, seed)
    return m


def paired(y, p1, p0, g, w=None, n_boot: int = 5000, seed: int = 13) -> dict:
    return S.paired_wf1_delta(y, p1, p0, g, w, n_boot, seed)


def compare_family(
    y, cand, comparators: dict, g, w=None, n_boot: int = 5000, seed: int = 13
) -> list[dict]:
    """cand minus each comparator, same units; Holm over the comparators of this call."""
    rows = []
    for name, p in comparators.items():
        d = paired(y, cand, p, g, w, n_boot, seed)
        d["comparator"] = name
        rows.append(d)
    adj = holm([r["p_boot"] for r in rows])
    for r, a in zip(rows, adj):
        r["p_holm"] = a
        r["verdict"] = (
            "supported after Holm"
            if (a < 0.05 and r["ci"][0] > 0)
            else "not demonstrated"
        )
    return rows


def coverage_risk(y, states, g, w=None, n_boot: int = 2000, seed: int = 13) -> dict:
    """Coverage / error-among-accepted with speaker-bootstrap CI. Uncertain = abstention (never a negative)."""
    y, st, g = np.asarray(y), np.asarray(states), np.asarray(g).astype(str)
    w = np.ones(len(y)) if w is None else np.asarray(w, float)
    out = selective_report(y, st, w)
    ug, inv = np.unique(g, return_inverse=True)
    acc = st != UNCERTAIN
    pred = (st == PRESENT).astype(int)
    A = np.zeros((len(ug), 3))
    np.add.at(A[:, 0], inv, w)
    np.add.at(A[:, 1], inv, w * acc)
    np.add.at(A[:, 2], inv, w * (acc & (pred != y)))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(ug), (n_boot, len(ug)))
    T = A[idx].sum(1)
    cov = T[:, 1] / np.where(T[:, 0] > 0, T[:, 0], np.nan)
    risk = T[:, 2] / np.where(T[:, 1] > 0, T[:, 1], np.nan)
    out["coverage_ci95"] = [
        float(np.nanpercentile(cov, 2.5)),
        float(np.nanpercentile(cov, 97.5)),
    ]
    out["risk_ci95"] = [
        float(np.nanpercentile(risk, 2.5)),
        float(np.nanpercentile(risk, 97.5)),
    ]
    return out


def promotion(vs_default: dict, alpha: float = 0.05) -> dict:
    """Promotion rule (frozen in PROTOCOL.md section 7).

    ``vs_default``: class -> paired result (delta, ci, p_boot) of the optimised pipeline minus the current default.
    Promote iff, over the Holm family of the classes: (a) at least one class has adjusted p < alpha and CI lower > 0,
    and (b) in NO class the CI lies entirely below 0 (not worse). A class that cannot be evaluated (NaN CI) blocks
    promotion: absence of evidence is not a pass."""
    cls = list(vs_default)
    adj = holm([vs_default[c]["p_boot"] for c in cls])
    per = {}
    for c, a in zip(cls, adj):
        r = vs_default[c]
        lo, hi = r["ci"]
        ok = bool(np.isfinite(lo) and np.isfinite(hi))
        per[c] = {
            "delta": r["delta"],
            "ci": r["ci"],
            "p_holm": a,
            "superior": bool(ok and a < alpha and lo > 0),
            "not_worse": bool(ok and hi >= 0),
            "evaluable": ok,
        }
    sup = [c for c in cls if per[c]["superior"]]
    worse = [c for c in cls if not per[c]["not_worse"]]
    promote = bool(sup) and not worse
    return {
        "promote": promote,
        "superior_in": sup,
        "worse_or_unevaluable_in": worse,
        "per_class": per,
        "reason": (
            "promoted"
            if promote
            else (
                "no class superior after Holm"
                if not sup
                else f"worse/unevaluable in {worse}"
            )
        ),
    }
