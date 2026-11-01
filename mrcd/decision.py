"""Decision layer on top of window probabilities (new module, 2026-10-30).

V0 rule ("any window argmax == class") ignores how confident the window was and lets every other class
compete. This module offers a per-class SCORE and a per-class threshold on it.

Honesty: the checkpoint probabilities are NOT calibrated. A window softmax value is a score used for
ranking and thresholding, never a probability of presence. Thresholds must be fitted on development data
only (speaker-grouped cross-validation, ``cv_threshold``) and frozen before any held-out evaluation.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

import numpy as np

AGGREGATIONS = ("max", "mean", "top2")
UNCERTAIN, PRESENT, ABSENT = "uncertain", "present", "absent"
GRID = np.round(np.arange(0.02, 0.981, 0.02), 4)


def any_window_argmax(probs: np.ndarray, labels: list[str], cls: str) -> int:
    """V0 rule: 1 if any window's argmax equals ``cls``."""
    return int((probs.argmax(1) == labels.index(cls)).any()) if len(probs) else 0


def unit_score(
    probs: np.ndarray, labels: list[str], cls: str, agg: str = "max"
) -> float:
    """Aggregate the window probabilities of ``cls`` into one unit score (nan when there is no window)."""
    if agg not in AGGREGATIONS:
        raise ValueError(f"agg must be one of {AGGREGATIONS}")
    if len(probs) == 0:
        return float("nan")
    p = np.asarray(probs)[:, labels.index(cls)]
    if agg == "max":
        return float(p.max())
    if agg == "mean":
        return float(p.mean())
    return float(np.sort(p)[-2:].mean())


def f1_binary(y, pred) -> float:
    y = np.asarray(y).astype(int)
    pred = np.asarray(pred).astype(int)
    tp = int(((y == 1) & (pred == 1)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    return 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0


def best_threshold(scores, y, grid=GRID) -> tuple[float, float]:
    """Threshold maximizing F1 (``score >= t`` is positive). Ties -> median of the tied thresholds."""
    s = np.asarray(scores, float)
    fs = np.array([f1_binary(y, s >= t) for t in grid])
    top = np.flatnonzero(fs >= fs.max() - 1e-12)
    t = float(grid[top[len(top) // 2]])
    return t, float(fs.max())


def group_folds(groups, k: int = 5, seed: int = 13) -> np.ndarray:
    """Deterministic fold id per item so that every group (speaker) falls in exactly ONE fold."""
    g = np.asarray(groups)
    ug = sorted(
        set(g.tolist()),
        key=lambda x: hashlib.sha256(f"{seed}|{x}".encode()).hexdigest(),
    )
    fold_of = {x: i % k for i, x in enumerate(ug)}
    return np.array([fold_of[x] for x in g.tolist()])


def cv_threshold(scores, y, groups, k: int = 5, seed: int = 13, grid=GRID) -> dict:
    """Speaker-grouped CV for a threshold: out-of-fold predictions + the final threshold on ALL data.

    Returns ``{"threshold", "dev_f1_in_sample", "oof_pred", "oof_f1", "fold_thresholds"}``. The out-of-fold F1
    is the honest development estimate; the in-sample F1 is optimistic by construction.
    """
    s = np.asarray(scores, float)
    y = np.asarray(y).astype(int)
    fold = group_folds(groups, k, seed)
    oof = np.zeros(len(s), int)
    fts = []
    for f in range(k):
        te = fold == f
        if not te.any() or (~te).sum() == 0:
            continue
        t, _ = best_threshold(s[~te], y[~te], grid)
        fts.append(t)
        oof[te] = (s[te] >= t).astype(int)
    t_all, f_all = best_threshold(s, y, grid)
    return {
        "threshold": t_all,
        "dev_f1_in_sample": f_all,
        "oof_pred": oof,
        "oof_f1": f1_binary(y, oof),
        "fold_thresholds": fts,
    }


def three_state(score: float, threshold: float, margin: float = 0.0) -> str:
    """present if score >= threshold + margin, absent if score < threshold - margin, else uncertain.

    NaN scores (no window) are always ``uncertain``: absence of evidence is not evidence of absence.
    """
    if score != score:
        return UNCERTAIN
    if score >= threshold + margin:
        return PRESENT
    if score < threshold - margin:
        return ABSENT
    return UNCERTAIN


@dataclass
class ThresholdRule:
    """Per-class aggregation + threshold. JSON round-trippable so it can be frozen and hashed."""

    agg: dict[str, str] = field(default_factory=dict)
    threshold: dict[str, float] = field(default_factory=dict)
    note: str = "scores are NOT calibrated probabilities"

    def score(self, probs, labels, cls) -> float:
        return unit_score(probs, labels, cls, self.agg[cls])

    def predict(self, probs, labels, cls, margin: float = 0.0) -> str:
        return three_state(self.score(probs, labels, cls), self.threshold[cls], margin)

    def to_json(self) -> str:
        return json.dumps(
            {"agg": self.agg, "threshold": self.threshold, "note": self.note},
            indent=1,
            sort_keys=True,
        )

    @classmethod
    def from_json(cls, s: str) -> "ThresholdRule":
        d = json.loads(s)
        return cls(
            d["agg"],
            {k: float(v) for k, v in d["threshold"].items()},
            d.get("note", ""),
        )
