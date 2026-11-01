"""Weighted LR text route (ctx_LR on any ASR text), rule+model fusion and a serialisable route artifact.

All thresholds / hyper-parameters are learned from the rows given (callers pass TRAIN rows only).
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

from mrcd.context_eval.model import CS, FittedLR
from mrcd.context_eval.partition import inner_splits
from mrcd.decision import GRID

ROUTE_VERSION = "s3-rep-ctx-p1/1"
NEVER = 1.01  # threshold that is never reached ("use the rule alone")


def wf1(y, pred, w) -> float:
    y = np.asarray(y).astype(int)
    pred = np.asarray(pred).astype(int)
    w = np.asarray(w, float)
    tp = w[(y == 1) & (pred == 1)].sum()
    fp = w[(y == 0) & (pred == 1)].sum()
    fn = w[(y == 1) & (pred == 0)].sum()
    return float(2 * tp / (2 * tp + fp + fn)) if 2 * tp + fp + fn else 0.0


def best_threshold_w(scores, y, w, grid=GRID) -> float:
    s = np.asarray(scores, float)
    fs = np.array([wf1(y, s >= t, w) for t in grid])
    top = np.flatnonzero(fs >= fs.max() - 1e-12)
    return float(grid[top[len(top) // 2]])


def fit_lr_w(X, y, w, C: float) -> FittedLR:
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
            (X - mu) / sd, y, sample_weight=np.asarray(w, float)
        )
    return FittedLR(mu, sd, clf)


def select_fit_w(X, y, groups, w=None, Cs=CS, k_inner: int = 3, seed: int = 13) -> dict:
    """Weighted analogue of ``context_eval.model.select_fit`` (C by weighted inner-OOF AP, threshold by weighted F1)."""
    X = np.asarray(X, float)
    y = np.asarray(y).astype(int)
    w = np.ones(len(y)) if w is None else np.asarray(w, float)
    g = np.asarray([str(a) for a in groups])
    oof = {C: np.full(len(y), np.nan) for C in Cs}
    ok = len(set(g.tolist())) >= k_inner and 0 < y.sum() < len(y)
    if ok:
        for _, tr, te in inner_splits(g, k_inner, seed):
            assert not (set(g[tr]) & set(g[te]))
            for C in Cs:
                oof[C][te] = fit_lr_w(X[tr], y[tr], w[tr], C).score(X[te])
        aps = {
            C: (
                float(average_precision_score(y, oof[C], sample_weight=w))
                if np.isfinite(oof[C]).all()
                else float("nan")
            )
            for C in Cs
        }
    else:
        aps = {C: float("nan") for C in Cs}
    valid = {C: a for C, a in aps.items() if np.isfinite(a)}
    C_best = max(valid, key=lambda c: (valid[c], -c)) if valid else Cs[len(Cs) // 2]
    s = oof[C_best]
    if ok and np.isfinite(s).all():
        thr = best_threshold_w(s, y, w)
    else:
        thr, s = 0.5, np.full(len(y), np.nan)
    return {
        "model": fit_lr_w(X, y, w, C_best),
        "C": C_best,
        "threshold": float(thr),
        "inner_oof": s,
        "inner_ap": {str(c): a for c, a in aps.items()},
    }


# ------------------------------------------------------------------ fusion
def fit_or(rule, score, y, w=None, grid=GRID) -> float:
    """t_or maximising (weighted) F1 of ``rule OR score >= t``; NEVER (rule alone) is a candidate (wins ties)."""
    rule = np.asarray(rule).astype(int)
    score = np.asarray(score, float)
    w = np.ones(len(rule)) if w is None else np.asarray(w, float)
    cands = [float(t) for t in grid] + [NEVER]
    fs = np.array([wf1(y, rule | (score >= t), w) for t in cands])
    top = np.flatnonzero(fs >= fs.max() - 1e-12)
    if len(cands) - 1 in top:
        return NEVER
    return float(cands[top[len(top) // 2]])


def apply_or(rule, score, t) -> np.ndarray:
    return (np.asarray(rule).astype(int) | (np.asarray(score, float) >= t)).astype(int)


def stack_features(rule, score) -> np.ndarray:
    s = np.clip(np.asarray(score, float), 1e-4, 1 - 1e-4)
    return np.column_stack([np.asarray(rule, float), np.log(s / (1 - s))])


# ------------------------------------------------------------------ serialisable route
@dataclass
class TextRoute:
    """ctx_LR trained on ASR text of a given profile; optional OR-fusion with the repetition rule."""

    task: str
    asr_profile: str
    mu: list
    sd: list
    coef: list
    intercept: float
    threshold: float
    t_or: float
    n_features: int
    train_note: str

    def score_texts(self, texts) -> np.ndarray:
        from mrcd.context_eval.textfeat import featurize

        X = featurize(list(texts), self.task)
        if X.shape[1] != self.n_features:
            raise ValueError("feature dimension differs from the trained route")
        z = ((X - np.array(self.mu)) / np.array(self.sd)) @ np.array(
            self.coef
        ) + self.intercept
        return 1 / (1 + np.exp(-z))

    def decide(self, texts, fused: bool = False):
        from mrcd.context_eval.textfeat import rule_rep

        s = self.score_texts(texts)
        m = (s >= self.threshold).astype(int)
        if fused:
            m = apply_or([rule_rep(t) for t in texts], s, self.t_or)
        return s, m

    def to_json(self) -> str:
        d = dict(self.__dict__)
        d["version"] = ROUTE_VERSION
        d["experimental"] = True
        d["note"] = (
            "score is a logistic-regression output on text features; NOT a calibrated probability"
        )
        return json.dumps(d, indent=1)

    @classmethod
    def from_json(cls, s: str) -> "TextRoute":
        d = json.loads(s)
        for k in ("version", "experimental", "note"):
            d.pop(k, None)
        return cls(**d)

    @classmethod
    def load(cls, path) -> "TextRoute":
        return cls.from_json(Path(path).read_text("utf8"))


def route_from_fit(
    sel: dict, task: str, asr_profile: str, t_or: float, note: str
) -> TextRoute:
    m: FittedLR = sel["model"]
    return TextRoute(
        task,
        asr_profile,
        m.mu.tolist(),
        m.sd.tolist(),
        m.clf.coef_[0].tolist(),
        float(m.clf.intercept_[0]),
        float(sel["threshold"]),
        float(t_or),
        int(len(m.mu)),
        note,
    )
