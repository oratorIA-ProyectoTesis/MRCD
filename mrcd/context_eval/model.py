"""Regularised logistic regression with nested speaker-grouped selection, threshold and cross-fitted calibration.

Everything learned (standardisation, C, threshold, calibrator) comes from the TRAIN rows of each outer fold only.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

from mrcd.context_eval.partition import GroupCrossingError, inner_splits
from mrcd.decision import best_threshold

CS = (0.03, 0.3, 3.0)
SEED = 13


@dataclass
class FittedLR:
    mu: np.ndarray
    sd: np.ndarray
    clf: LogisticRegression | None
    const: float = 0.0  # score returned when the train set has a single class

    def score(self, X) -> np.ndarray:
        X = np.asarray(X, float)
        if self.clf is None:
            return np.full(len(X), self.const)
        return self.clf.predict_proba((X - self.mu) / self.sd)[:, 1]


def fit_lr(X, y, C: float) -> FittedLR:
    X = np.asarray(X, float)
    y = np.asarray(y).astype(int)
    if len(set(y.tolist())) < 2:
        return FittedLR(
            np.zeros(X.shape[1]),
            np.ones(X.shape[1]),
            None,
            float(y.mean()) if len(y) else 0.0,
        )
    mu = X.mean(0)
    sd = X.std(0)
    sd[sd < 1e-12] = 1.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf = LogisticRegression(C=C, solver="lbfgs", max_iter=2000).fit(
            (X - mu) / sd, y
        )
    return FittedLR(mu, sd, clf)


def _ap(y, s) -> float:
    return (
        float(average_precision_score(y, s))
        if 0 < int(np.sum(y)) < len(y)
        else float("nan")
    )


def select_fit(X, y, groups, Cs=CS, k_inner: int = 3, seed: int = SEED) -> dict:
    """Choose C (inner-OOF average precision), threshold (inner-OOF argmax F1), refit on all given rows.

    Returns dict(model, C, threshold, inner_oof (aligned to rows), inner_ap {C: ap}).
    """
    X = np.asarray(X, float)
    y = np.asarray(y).astype(int)
    g = np.asarray([str(a) for a in groups])
    oof = {C: np.full(len(y), np.nan) for C in Cs}
    ok_inner = len(set(g.tolist())) >= k_inner and 0 < y.sum() < len(y)
    if ok_inner:
        for _, tr, te in inner_splits(g, k_inner, seed):
            if set(g[tr]) & set(g[te]):
                raise GroupCrossingError("inner train/test share a group")
            for C in Cs:
                oof[C][te] = fit_lr(X[tr], y[tr], C).score(X[te])
        aps = {C: _ap(y, oof[C]) for C in Cs}
    else:
        aps = {C: float("nan") for C in Cs}
    valid = {C: a for C, a in aps.items() if np.isfinite(a)}
    C_best = max(valid, key=lambda c: (valid[c], -c)) if valid else Cs[len(Cs) // 2]
    s_oof = oof[C_best]
    if ok_inner and np.isfinite(s_oof).all():
        thr, _ = best_threshold(s_oof, y)
    else:
        thr = 0.5
        s_oof = np.full(len(y), np.nan)
    return {
        "model": fit_lr(X, y, C_best),
        "C": C_best,
        "threshold": float(thr),
        "inner_oof": s_oof,
        "inner_ap": {str(c): a for c, a in aps.items()},
    }


def platt(s_train, y_train):
    """Platt scaling on the logit of an uncalibrated score; returns a function score -> probability."""
    s = np.clip(np.asarray(s_train, float), 1e-4, 1 - 1e-4)
    y = np.asarray(y_train).astype(int)
    if len(set(y.tolist())) < 2:
        c = float(y.mean()) if len(y) else 0.0
        return lambda x: np.full(len(np.asarray(x)), c)
    lo = np.log(s / (1 - s)).reshape(-1, 1)
    lr = LogisticRegression(C=100.0, solver="lbfgs").fit(lo, y)

    def f(x):
        x = np.clip(np.asarray(x, float), 1e-4, 1 - 1e-4)
        return lr.predict_proba(np.log(x / (1 - x)).reshape(-1, 1))[:, 1]

    return f


def nested_oof(
    X,
    y,
    groups,
    outer_fold,
    calibrate: bool = False,
    Cs=CS,
    k_inner: int = 3,
    seed: int = SEED,
):
    """Pooled out-of-fold scores / decisions with everything learned inside the outer train.

    ``outer_fold``: integer fold id per row (speaker-grouped, built once for the whole pool).
    Returns dict(score, decision, calibrated (or None), per_fold list).
    """
    X = np.asarray(X, float)
    y = np.asarray(y).astype(int)
    g = np.asarray([str(a) for a in groups])
    fold = np.asarray(outer_fold).astype(int)
    score = np.full(len(y), np.nan)
    dec = np.zeros(len(y), int)
    cal = np.full(len(y), np.nan) if calibrate else None
    per = []
    for f in sorted(set(fold.tolist())):
        te = np.flatnonzero(fold == f)
        tr = np.flatnonzero(fold != f)
        if set(g[tr]) & set(g[te]):
            raise GroupCrossingError("outer train/test share a group")
        sel = select_fit(X[tr], y[tr], g[tr], Cs, k_inner, seed)
        s = sel["model"].score(X[te])
        score[te] = s
        dec[te] = (s >= sel["threshold"]).astype(int)
        if calibrate:
            m = np.isfinite(sel["inner_oof"])
            cal[te] = platt(sel["inner_oof"][m], y[tr][m])(s) if m.any() else s
        per.append(
            {
                "fold": int(f),
                "n_train": int(len(tr)),
                "n_test": int(len(te)),
                "pos_train": int(y[tr].sum()),
                "pos_test": int(y[te].sum()),
                "C": sel["C"],
                "threshold": sel["threshold"],
                "inner_ap": sel["inner_ap"],
            }
        )
    return {"score": score, "decision": dec, "calibrated": cal, "per_fold": per}
