"""Contextual occurrence classifier: L2 logistic regression with nested, speaker-grouped selection."""

from __future__ import annotations

import hashlib
import pickle
from collections.abc import Sequence

import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

from mrcd.context_ambiguity.features import ACOUSTIC_KEYS
from mrcd.context_eval.partition import group_folds

C_GRID = (0.03, 0.1, 0.3, 1.0, 3.0)
INNER_K = 3
SEED = 13


class CtxModel:
    """DictVectorizer + standardised acoustic columns + logistic regression + F1 threshold."""

    def __init__(self, C: float = 1.0, threshold: float = 0.5):
        self.C, self.threshold = C, threshold
        self.vec = DictVectorizer(sparse=True)
        self.clf = LogisticRegression(C=C, max_iter=2000, solver="lbfgs")
        self.mu: dict[str, float] = {}
        self.sd: dict[str, float] = {}
        self.const: float | None = None

    def _scale(self, feats: Sequence[dict]) -> list[dict]:
        out = []
        for f in feats:
            g = dict(f)
            for k in ACOUSTIC_KEYS:
                if k in g:
                    g[k] = (g[k] - self.mu[k]) / self.sd[k]
            out.append(g)
        return out

    def fit(self, feats: Sequence[dict], y: Sequence[int]) -> "CtxModel":
        for k in ACOUSTIC_KEYS:
            v = np.array([f[k] for f in feats if k in f], float)
            if len(v):
                self.mu[k], self.sd[k] = float(v.mean()), float(v.std() or 1.0)
        X = self.vec.fit_transform(self._scale(feats))
        yy = np.asarray(y, int)
        if yy.min() == yy.max():  # degenerate fold: constant score
            self.const = float(yy[0])
        else:
            self.const = None
            self.clf.fit(X, yy)
        return self

    def score(self, feats: Sequence[dict]) -> np.ndarray:
        if self.const is not None:
            return np.full(len(feats), self.const)
        return self.clf.predict_proba(self.vec.transform(self._scale(feats)))[:, 1]

    def predict(self, feats: Sequence[dict]) -> np.ndarray:
        return (self.score(feats) >= self.threshold).astype(int)


def best_threshold(y: np.ndarray, s: np.ndarray) -> tuple[float, float]:
    """Threshold maximising F1 on (y, s); ties go to the largest threshold. Returns (thr, f1)."""
    best = (0.5, -1.0)
    for t in np.unique(np.round(s, 6)):
        p = s >= t
        tp = int((p & (y == 1)).sum())
        fp = int((p & (y == 0)).sum())
        fn = int((~p & (y == 1)).sum())
        f = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
        if f >= best[1]:
            best = (float(t), f)
    return best


def select_and_fit(
    feats: Sequence[dict], y: Sequence[int], groups: Sequence[str]
) -> tuple[CtxModel, dict]:
    """Inner speaker-grouped CV picks C and the threshold; refits on all given rows."""
    y = np.asarray(y, int)
    fold = group_folds(list(groups), min(INNER_K, len(set(groups))), SEED)
    info = {}
    best = None
    for C in C_GRID:
        oof = np.zeros(len(y))
        for k in sorted(set(fold)):
            tr, te = np.flatnonzero(fold != k), np.flatnonzero(fold == k)
            m = CtxModel(C).fit([feats[i] for i in tr], y[tr])
            oof[te] = m.score([feats[i] for i in te])
        thr, f1 = best_threshold(y, oof)
        info[C] = {"inner_f1": f1, "thr": thr}
        if best is None or f1 > best[1] + 1e-9:
            best = (C, f1, thr)
    C, f1, thr = best
    final = CtxModel(C, thr).fit(feats, y)
    return final, {"C": C, "threshold": thr, "inner_f1": f1, "grid": info}


def save_artifact(model: CtxModel, vocab, path, meta: dict) -> str:
    blob = pickle.dumps({"model": model, "vocab": vocab, "meta": meta})
    with open(path, "wb") as f:
        f.write(blob)
    return hashlib.sha256(blob).hexdigest()
