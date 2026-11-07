"""Pickle-free export of the packaged per-class pipelines (logistic-regression families only).

The campaign artifacts (``models/o_campaign_20261101/*.pkl``) are pickles. Unpickling executes arbitrary code, so
the product never reads them: ``export_pipeline`` turns a fitted logistic-regression ``ClassPipeline`` into plain JSON
numbers and ``load_pipeline`` rebuilds an equivalent object from that JSON. The pickles are read only by the export
script, after their sha256 matched the campaign manifest.

The JSON holds exactly what ``ClassPipeline.predict`` uses: block list, feature profile, duration normaliser,
standardiser (mean / sd), coefficients, intercept, thresholds, optional calibrator and abstention band.
"""

from __future__ import annotations

import json

import numpy as np

from mrcd.optimize import space as SP
from mrcd.optimize.class_pipeline import ClassPipeline

FORMAT = "mrcd.o_pipeline.json/1"


class JsonLR:
    """``FittedLR.score`` rebuilt from plain numbers: ``sigmoid(((x - mu) / sd) @ coef + intercept)``."""

    def __init__(self, mu, sd, coef, intercept):
        self.mu = np.asarray(mu, float)
        self.sd = np.asarray(sd, float)
        self.coef = np.asarray(coef, float)
        self.intercept = float(intercept)

    def score(self, X) -> np.ndarray:
        z = ((np.asarray(X, float) - self.mu) / self.sd) @ self.coef + self.intercept
        return 1.0 / (1.0 + np.exp(-z))


def export_pipeline(p: ClassPipeline) -> dict:
    m = p.model
    if p.cfg.model != "lr" or getattr(m, "clf", None) is None:
        raise ValueError(
            f"only fitted logistic-regression pipelines are exportable (got {p.cfg.model!r})"
        )
    clf = m.clf
    if clf.coef_.shape[0] != 1:
        raise ValueError("binary classifier expected")
    return {
        "format": FORMAT,
        "config": p.cfg.to_dict(),
        "stats": {"med": float(p.stats["med"]), "scale": float(p.stats["scale"])},
        "lr": {
            "mu": [float(v) for v in np.asarray(m.mu, float)],
            "sd": [float(v) for v in np.asarray(m.sd, float)],
            "coef": [float(v) for v in clf.coef_[0]],
            "intercept": float(clf.intercept_[0]),
        },
        "thr": float(p.thr),
        "t_or": None if p.t_or is None else float(p.t_or),
        "calibrator": p.calibrator,
        "band": {"lo": float(p.band["lo"]), "hi": float(p.band["hi"])},
        "train_note": p.train_note,
        "calibration_claim": "none",
    }


def dumps(doc: dict) -> bytes:
    return (json.dumps(doc, indent=1, sort_keys=True, allow_nan=False) + "\n").encode(
        "utf8"
    )


def load_pipeline(doc: dict) -> ClassPipeline:
    if doc.get("format") != FORMAT:
        raise ValueError(f"unsupported pipeline format {doc.get('format')!r}")
    lr = doc["lr"]
    return ClassPipeline(
        SP.Config.from_dict(doc["config"]),
        dict(doc["stats"]),
        JsonLR(lr["mu"], lr["sd"], lr["coef"], lr["intercept"]),
        float(doc["thr"]),
        None if doc["t_or"] is None else float(doc["t_or"]),
        doc["calibrator"],
        dict(doc["band"]),
        doc["train_note"],
    )
