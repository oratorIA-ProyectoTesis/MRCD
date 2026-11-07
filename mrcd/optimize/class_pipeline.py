"""The per-class pipeline object (no serialisation code on purpose).

Split out of ``product_pipeline`` so the service decision path (``safe_pipeline``) does not transitively import the
module that can ``pickle.loads``. ``product_pipeline`` re-exports ``ClassPipeline`` so existing pickles still resolve."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mrcd import calibration as CAL
from mrcd.asr_recovery.signals import SignalStats
from mrcd.fusion_text import apply_or
from mrcd.optimize import blocks as B
from mrcd.optimize import space as SP


@dataclass
class ClassPipeline:
    cfg: SP.Config
    stats: dict  # {"med": float, "scale": float} duration normaliser fitted on the training rows
    model: object
    thr: float
    t_or: float | None
    calibrator: dict | None
    band: dict
    train_note: str

    def matrix(self, recs: list[dict]) -> np.ndarray:
        st = SignalStats(self.stats["med"], self.stats["scale"])
        parts = []
        for b in self.cfg.blocks:
            if b in B.STATELESS:
                parts.append(
                    B._san(B.STATELESS[b](recs, self.cfg.profile, self.cfg.cls))
                )
            else:
                parts.append(B._san(B.stateful_block(b, recs, self.cfg.profile, st)))
        return np.hstack(parts)

    def predict(self, recs: list[dict]) -> dict:
        s = np.asarray(self.model.score(self.matrix(recs)), float)
        if self.cfg.fusion == "or_rule":
            r = B._san(B.b_rule(recs, self.cfg.profile, self.cfg.cls))[:, 0].astype(int)
            dec = apply_or(r, s, self.t_or)
        else:
            dec = (s >= self.thr).astype(int)
        cal = (
            CAL.Calibrator.from_dict(self.calibrator).apply(s) if self.calibrator else s
        )
        return {
            "score": s,
            "decision": dec,
            "calibrated": cal,
            "state": CAL.states(cal, self.band["lo"], self.band["hi"]),
        }
