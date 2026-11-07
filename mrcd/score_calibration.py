"""Opt-in, versioned score calibration and abstention policy (calibracion_20261107; EXPLORATORY).

Maps the unit score of the V0 head (max softmax of a class over the windows of a turn) to a frequency estimated on
DIHANA folds 1-4 (speaker-grouped cross-validation). It is NOT wired into the engine or the service: ``default`` is
always False. The word "probability" is never returned; the estimate is labelled as estimated on development data.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

from mrcd.calibration import ABSENT, PRESENT, UNCERTAIN, Calibrator

VERSION = "score-calibration/1"
WORDING = "estimated frequency on development data (DIHANA folds 1-4, exploratory); not a validated probability"
CALIBRATORS_FILE = "calibrators.json"
MANIFEST_FILE = "MANIFEST.json"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class ManifestError(ValueError):
    pass


@dataclass
class ClassPolicy:
    calibrator: Calibrator
    threshold: float  # F1-optimal cut on the calibrated score (train rows)
    band: (
        float | None
    )  # RF-09 half-width d; None = abstention not enabled for this class

    def estimate(self, score: float) -> float:
        return float(self.calibrator.apply([score])[0])

    def state(self, score: float) -> str:
        p = self.estimate(score)
        if p != p:
            return UNCERTAIN
        if self.band is None:
            return PRESENT if p >= self.threshold else ABSENT
        if p >= self.threshold + self.band:
            return PRESENT
        if p < self.threshold - self.band:
            return ABSENT
        return UNCERTAIN


@dataclass
class ScoreCalibration:
    version: str
    default: bool
    status: str
    wording: str
    classes: dict[str, ClassPolicy] = field(default_factory=dict)

    def estimate(self, cls: str, score: float) -> dict:
        pol = self.classes[cls]
        return {"estimate": pol.estimate(score), "note": self.wording}

    def state(self, cls: str, score: float) -> str:
        return self.classes[cls].state(score)


def save(out_dir, classes: dict[str, dict], meta: dict) -> dict:
    """``classes``: {cls: {"calibrator": Calibrator, "threshold": float, "band": float|None, ...extra}}."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    body = {
        c: {
            "calibrator": v["calibrator"].to_dict(),
            "threshold": float(v["threshold"]),
            "band": None if v.get("band") is None else float(v["band"]),
        }
        for c, v in classes.items()
    }
    cpath = out / CALIBRATORS_FILE
    cpath.write_text(json.dumps(body, indent=1, sort_keys=True), "utf8")
    manifest = {
        "version": VERSION,
        "default": False,
        "status": "exploratory",
        "wording": WORDING,
        "calibrators_sha256": sha256_file(cpath),
        **meta,
    }
    (out / MANIFEST_FILE).write_text(
        json.dumps(manifest, indent=1, sort_keys=True), "utf8"
    )
    return manifest


def load(out_dir) -> ScoreCalibration:
    out = Path(out_dir)
    man = json.loads((out / MANIFEST_FILE).read_text("utf8"))
    if man.get("version") != VERSION:
        raise ManifestError(f"unsupported version {man.get('version')!r}")
    if man.get("default") is not False:
        raise ManifestError("calibration must be opt-in (default false)")
    cpath = out / CALIBRATORS_FILE
    if sha256_file(cpath) != man["calibrators_sha256"]:
        raise ManifestError("calibrators.json does not match the manifest sha256")
    body = json.loads(cpath.read_text("utf8"))
    return ScoreCalibration(
        man["version"],
        False,
        man["status"],
        man["wording"],
        {
            c: ClassPolicy(
                Calibrator.from_dict(v["calibrator"]), v["threshold"], v["band"]
            )
            for c, v in body.items()
        },
    )
