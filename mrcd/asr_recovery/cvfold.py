"""Nested speaker-grouped OOF where anything learned from the data (including feature statistics) comes from the TRAIN rows only."""

from __future__ import annotations

import numpy as np

from mrcd.context_eval.model import CS, SEED, select_fit
from mrcd.context_eval.partition import GroupCrossingError


def nested_oof_fold_aware(
    make_X, y, groups, outer_fold, Cs=CS, k_inner: int = 3, seed: int = SEED
):
    """``make_X(train_idx) -> (n, d) matrix``: called once per outer fold with the TRAIN row indices only.

    Anything it learns (medians, z-score statistics) must use only those rows; the test rows are transformed with them.
    Returns dict(score, decision, per_fold) with per_fold items carrying the fitted ``sel`` (model, C, threshold).
    """
    y = np.asarray(y).astype(int)
    g = np.asarray([str(a) for a in groups])
    fold = np.asarray(outer_fold).astype(int)
    score = np.full(len(y), np.nan)
    dec = np.zeros(len(y), int)
    per = []
    for f in sorted(set(fold.tolist())):
        te = np.flatnonzero(fold == f)
        tr = np.flatnonzero(fold != f)
        if set(g[tr]) & set(g[te]):
            raise GroupCrossingError("outer train/test share a group")
        X = np.asarray(make_X(tr), float)
        if X.shape[0] != len(y):
            raise ValueError("make_X must return one row per unit")
        sel = select_fit(X[tr], y[tr], g[tr], Cs, k_inner, seed)
        s = sel["model"].score(X[te])
        score[te] = s
        dec[te] = (s >= sel["threshold"]).astype(int)
        per.append(
            {
                "fold": int(f),
                "n_train": int(len(tr)),
                "n_test": int(len(te)),
                "pos_train": int(y[tr].sum()),
                "pos_test": int(y[te].sum()),
                "C": sel["C"],
                "threshold": sel["threshold"],
                "sel": sel,
            }
        )
    return {"score": score, "decision": dec, "per_fold": per}


class NanFill:
    """Column medians learned on given rows; NaN / inf are replaced by them."""

    def fit(self, X):
        X = np.asarray(X, float)
        Xf = np.where(np.isfinite(X), X, np.nan)
        with np.errstate(all="ignore"):
            import warnings

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                med = np.nanmedian(Xf, axis=0)
        self.med = np.where(np.isfinite(med), med, 0.0)
        return self

    def transform(self, X):
        X = np.asarray(X, float)
        return np.where(np.isfinite(X), X, self.med[None, :])
