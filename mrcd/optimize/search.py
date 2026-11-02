"""Nested search. Everything chosen (configuration, threshold, OR-threshold, calibration, band) comes from the inner
loop of the OUTER-TRAIN rows; the outer test rows are only scored."""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field

import numpy as np

from mrcd import calibration as CAL
from mrcd.context_eval.partition import GroupCrossingError, inner_splits
from mrcd.fusion_text import apply_or, best_threshold_w, fit_or, wf1
from mrcd.optimize import space as SP
from mrcd.optimize.blocks import FeatureStore
from mrcd.optimize.models import fit_model

K_INNER = 3


@dataclass
class Data:
    """One class' dataset: aligned arrays + the feature store (records)."""

    cls: str
    ids: list
    y: np.ndarray
    w: np.ndarray
    g: np.ndarray
    outer: np.ndarray
    store: FeatureStore
    rule: dict = field(default_factory=dict)  # profile -> int rule vector

    def rule_vec(self, profile: str) -> np.ndarray:
        if profile not in self.rule:
            if self.cls == "prolongation":
                self.rule[profile] = np.zeros(len(self.y), int)
            else:
                self.rule[profile] = self.store.stateless("rule", profile)[:, 0].astype(
                    int
                )
        return self.rule[profile]


def _assert_disjoint(g, a, b, what):
    if set(g[a]) & set(g[b]):
        raise GroupCrossingError(f"{what}: train/test share a speaker")


def inner_oof(
    cfg: SP.Config, d: Data, tr: np.ndarray, seed: int = SP.SEED
) -> np.ndarray:
    """Pooled inner out-of-fold scores of ``cfg`` over rows ``tr`` (global indices); normalisers fitted on inner-train."""
    s = np.full(len(tr), np.nan)
    g = d.g[tr]
    for _, itr, ite in inner_splits(g, K_INNER, seed):
        _assert_disjoint(g, itr, ite, "inner")
        gtr = tr[itr]
        X = d.store.matrix(cfg.blocks, cfg.profile, gtr)
        m = fit_model(cfg.model, dict(cfg.hp), X[gtr], d.y[gtr], d.w[gtr])
        s[ite] = m.score(X[tr[ite]])
    return s


def objective(cfg: SP.Config, d: Data, tr: np.ndarray, s: np.ndarray) -> dict:
    """Weighted F1 at the threshold chosen on the SAME inner-OOF scores (equally optimistic for every config)."""
    y, w = d.y[tr], d.w[tr]
    thr = best_threshold_w(s, y, w)
    out = {"thr": thr, "t_or": None}
    if cfg.fusion == "or_rule":
        r = d.rule_vec(cfg.profile)[tr]
        t_or = fit_or(r, s, y, w)
        out["t_or"] = t_or
        out["f1"] = wf1(y, apply_or(r, s, t_or), w)
    else:
        out["f1"] = wf1(y, s >= thr, w)
    return out


def search(
    d: Data,
    tr: np.ndarray,
    seed: int,
    budget: int = SP.BUDGET,
    patience: int = SP.PATIENCE,
    eps: float = SP.EPS,
    evaluate=None,
) -> dict:
    """Seeded random search with a stop rule: budget exhausted, or ``patience`` consecutive configurations
    without a gain larger than ``eps`` over the best inner objective. ``evaluate(cfg) -> (scores, objective)`` is
    injectable (tests)."""
    order = SP.sample_order(d.cls, seed, budget)
    trials, best, since, stop = [], None, 0, "budget"
    t0 = time.time()
    for i, cfg in enumerate(order):
        if evaluate is None:
            s = inner_oof(cfg, d, tr, seed)
            ob = (
                objective(cfg, d, tr, s)
                if np.isfinite(s).all()
                else {"f1": float("-inf"), "thr": 0.5, "t_or": None}
            )
        else:
            s, ob = evaluate(cfg)
        trials.append(
            {
                "i": i,
                "config": cfg.to_dict(),
                "inner_f1": ob["f1"],
                "thr": ob["thr"],
                "t_or": ob["t_or"],
            }
        )
        gain = best is None or ob["f1"] > best["inner_f1"] + eps
        if best is None or ob["f1"] > best["inner_f1"]:
            best = {
                "inner_f1": ob["f1"],
                "config": cfg,
                "thr": ob["thr"],
                "t_or": ob["t_or"],
                "inner_oof": s,
            }
        since = 0 if gain else since + 1
        if evaluate is None:
            print(f"  cfg {i + 1}/{len(order)} {cfg.label()} inner_f1={ob['f1']:.4f} best={best['inner_f1']:.4f}", flush=True)
        if since >= patience:
            stop = "patience"
            break
    return {
        "best": best,
        "trials": trials,
        "stop": stop,
        "n_evaluated": len(trials),
        "seconds": round(time.time() - t0, 1),
    }


def choose_calibration(s_inner, y, w, g, thr):
    """Method in {none, platt, isotonic} with the lowest cross-fitted inner Brier, plus the abstention band."""
    s_inner = np.asarray(s_inner, float)
    cands = {"none": float(np.average((s_inner - y) ** 2, weights=w))}
    for m in ("platt", "isotonic"):
        c = CAL.crossfit(s_inner, y, g, m, K_INNER, SP.SEED, w)
        cands[m] = (
            float(np.average((c - y) ** 2, weights=w))
            if np.isfinite(c).all()
            else float("inf")
        )
    method = min(
        cands, key=lambda k: (cands[k], ("none", "platt", "isotonic").index(k))
    )
    if method == "none":
        cal, p_in = None, s_inner
    else:
        p_in = CAL.crossfit(s_inner, y, g, method, K_INNER, SP.SEED, w)
        cal = CAL.fit_calibrator(s_inner, y, method, w)
    thr_c = float(cal.apply([thr])[0]) if cal else float(thr)
    band = CAL.choose_band(p_in, y, thr_c, w)
    return method, cal, band, thr_c, cands


def fit_config(d: Data, tr: np.ndarray, cfg: SP.Config):
    X = d.store.matrix(cfg.blocks, cfg.profile, tr)
    return fit_model(cfg.model, dict(cfg.hp), X[tr], d.y[tr], d.w[tr]), X


def decide(d: Data, cfg: SP.Config, rows: np.ndarray, s, thr, t_or) -> np.ndarray:
    if cfg.fusion == "or_rule":
        return apply_or(d.rule_vec(cfg.profile)[rows], s, t_or)
    return (np.asarray(s) >= thr).astype(int)


def outer_fold(
    d: Data,
    f: int,
    seed: int = SP.SEED,
    budget: int = SP.BUDGET,
    patience: int = SP.PATIENCE,
) -> dict:
    te = np.flatnonzero(d.outer == f)
    tr = np.flatnonzero(d.outer != f)
    _assert_disjoint(d.g, tr, te, "outer")
    res = search(d, tr, seed + 1000 * (f + 1), budget, patience)
    b = res["best"]
    cfg = b["config"]
    m, X = fit_config(d, tr, cfg)
    s_te = m.score(X[te])
    dec = decide(d, cfg, te, s_te, b["thr"], b["t_or"])
    method, cal, band, thr_c, cands = choose_calibration(
        b["inner_oof"], d.y[tr], d.w[tr], d.g[tr], b["thr"]
    )
    p_te = cal.apply(s_te) if cal else s_te
    return {
        "fold": int(f),
        "te": te,
        "score": s_te,
        "decision": np.asarray(dec, int),
        "cal": p_te,
        "state": CAL.states(p_te, band["lo"], band["hi"]),
        "band": band,
        "calibration": method,
        "brier_cands": cands,
        "config": cfg.to_dict(),
        "inner_f1": b["inner_f1"],
        "thr": b["thr"],
        "t_or": b["t_or"],
        "stop": res["stop"],
        "n_evaluated": res["n_evaluated"],
        "trials": res["trials"],
        "seconds": res["seconds"],
        "n_train": int(len(tr)),
        "n_test": int(len(te)),
    }


def nested(d: Data, folds=None, **kw) -> dict:
    folds = sorted(set(d.outer.tolist())) if folds is None else folds
    n = len(d.y)
    out = {
        "score": np.full(n, np.nan),
        "decision": np.full(n, -1),
        "cal": np.full(n, np.nan),
        "state": np.full(n, "", object),
        "per_fold": [],
    }
    for f in folds:
        r = outer_fold(d, f, **kw)
        te = r.pop("te")
        out["score"][te], out["decision"][te] = r.pop("score"), r.pop("decision")
        out["cal"][te], out["state"][te] = r.pop("cal"), r.pop("state")
        out["per_fold"].append(r)
    return out


def learning_curve(
    d: Data, f: int, cfg: SP.Config, fracs=(0.25, 0.5, 0.75), seed: int = SP.SEED
) -> dict:
    """Outer-fold decisions when the already selected configuration is retrained on a fraction of the TRAIN speakers
    (deterministic sha order). The threshold is re-selected by inner CV on the subsample."""
    te = np.flatnonzero(d.outer == f)
    tr_all = np.flatnonzero(d.outer != f)
    spk = sorted(
        set(d.g[tr_all].tolist()),
        key=lambda x: hashlib.sha256(f"{seed}|lc|{x}".encode()).hexdigest(),
    )
    out = {}
    for fr in fracs:
        keep = set(spk[: max(int(round(fr * len(spk))), K_INNER)])
        tr = tr_all[np.isin(d.g[tr_all], list(keep))]
        _assert_disjoint(d.g, tr, te, "learning-curve")
        s = inner_oof(cfg, d, tr, seed)
        ob = objective(cfg, d, tr, s)
        m, X = fit_config(d, tr, cfg)
        dec = decide(d, cfg, te, m.score(X[te]), ob["thr"], ob["t_or"])
        out[str(fr)] = {
            "te": te,
            "decision": np.asarray(dec, int),
            "n_train": int(len(tr)),
            "n_speakers": int(len(keep)),
            "inner_f1": ob["f1"],
        }
    return out
