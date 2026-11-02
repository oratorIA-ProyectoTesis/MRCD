"""Model families (regularised LR, histogram gradient boosting) with sample weights."""

from __future__ import annotations

import warnings

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from mrcd.fusion_text import fit_lr_w

SEED = 13


class HGB:
    def __init__(self, clf, const=None):
        self.clf, self.const = clf, const

    def score(self, X):
        if self.clf is None:
            return np.full(len(X), self.const)
        return self.clf.predict_proba(np.asarray(X, float))[:, 1]


def fit_model(model: str, hp: dict, X, y, w):
    """Returns an object with ``.score(X) -> P(y=1)``. Single-class train sets give a constant."""
    X = np.asarray(X, float)
    y = np.asarray(y).astype(int)
    w = np.ones(len(y)) if w is None else np.asarray(w, float)
    if model == "lr":
        return fit_lr_w(X, y, w, float(hp["C"]))
    if model == "hgb":
        if len(set(y.tolist())) < 2:
            return HGB(None, float(y.mean()) if len(y) else 0.0)
        clf = HistGradientBoostingClassifier(
            max_depth=int(hp["max_depth"]),
            max_iter=int(hp["max_iter"]),
            learning_rate=0.05,
            l2_regularization=1.0,
            min_samples_leaf=20,
            early_stopping=False,
            random_state=SEED,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf.fit(X, y, sample_weight=w)
        return HGB(clf)
    raise ValueError(model)
