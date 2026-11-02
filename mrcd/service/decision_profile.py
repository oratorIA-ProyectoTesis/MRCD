"""Decision profiles of the service (contract rev 1.4).

* ``v0``       - the engine's historical decision for every class (window argmax events). Reproduces every document the
                 service produced before rev 1.4 (events, scores, summary). Always available, explicit opt-in.
* ``o2026_11`` - PRODUCT DEFAULT of the service. ``filler_word`` and ``prolongation`` come from the per-class pipelines
                 of the O-campaign (2026-11-01) applied to pause-delimited ASR segments; ``repetition`` and every other class
                 keep the engine decision (the campaign did not beat the one-line rule / S3 route for repetition).

The library (``DisfluencyEngine.analyze``) is NOT affected: it always returns the V0 events. Only the service chooses a profile.

Honest limits (also emitted as a warning in every document): exploratory result (DIHANA folds 1-4, nested CV by speaker),
configuration selection unstable across folds, the V0 comparator is a weak baseline, scores are model outputs and NOT
calibrated probabilities (``calibration_claim: none``), and the pipelines were trained on DIHANA *turns* while the service
applies them to ASR segments split at pauses >= 0.6 s (the same unit convention as the S3 text route), a domain shift that
was not measured outside DIHANA (EPIC filler: no demonstrated gain; AMERESCO prolongation: uninformative).

Artifacts are plain JSON (no pickle). Every file is read once, its sha256 is compared with the value PINNED IN THIS SOURCE
FILE and only then parsed from those same bytes; a changed file is refused (``ProfileArtifactError``).
"""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from mrcd.optimize import safe_pipeline as SAFE
from mrcd.service import s3_features as S3F
from mrcd.service.contract import SCORE_KIND_UNIT_PIPELINE  # noqa: F401  (re-exported)

PROFILE_V0 = "v0"
PROFILE_O = "o2026_11"
DECISION_PROFILES = (PROFILE_V0, PROFILE_O)
DEFAULT_DECISION_PROFILE = (
    PROFILE_O  # service / app default; the library default stays V0
)
O_CLASSES = ("filler_word", "prolongation")

PROFILE_DIR = Path(__file__).resolve().parent / "profiles" / PROFILE_O
#: sha256 of the exported pipelines, pinned here (a second manifest next to the files would not protect against an
#: attacker who can edit both). Regenerate with ``scripts/o_integration/export_json_pipelines.py`` and update on purpose.
PINNED_SHA256 = {
    "filler_word.json": "2a07b50c156bd8e064f6703b38c78dd85b1b853a6ad4a5e0e3be354a80858c1e",
    "prolongation.json": "d6230ed2e8b930e1cd930c89b1d74769310f7068c65c3070f75faee83ab7a96a",
}
#: sha256 of the campaign pickles the JSON files were exported from (provenance only; the pickles are never read here)
SOURCE_PICKLE_SHA256 = {
    "filler_word": "787828317c87b188469f09bec03d6fd3c633e8b1af2a7d2dff84725936064c91",
    "prolongation": "ba2f253648717a0671cf7f15d833bd466327b8f07db34ad43f471975b38f3f21",
}
MARGIN_S = (
    0.15  # context kept on each side of a segment clip (ASR word times are approximate)
)
MODALITIES = ["acoustic", "linguistic"]
VALIDATION_TEXT = (
    "exploratory unit-level decision (O-campaign pipeline) on an ASR segment split at pauses; "
    "interval = segment span; trained on DIHANA turns, not validated outside DIHANA"
)


class ProfileArtifactError(ValueError):
    """A profile artifact is missing, altered or malformed."""


def sha256_hex(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def load_o_pipelines(directory: Path | None = None) -> dict:
    """Verified pipelines of ``o2026_11``: ``{cls: ClassPipeline}``. Not cached (see ``o_pipelines``)."""
    d = Path(directory) if directory is not None else PROFILE_DIR
    out = {}
    for cls in O_CLASSES:
        name = f"{cls}.json"
        try:
            blob = (d / name).read_bytes()
        except OSError as exc:
            raise ProfileArtifactError(
                f"{name}: artifact missing ({type(exc).__name__})"
            ) from exc
        if sha256_hex(blob) != PINNED_SHA256[name]:
            raise ProfileArtifactError(f"{name}: sha256 mismatch (artifact altered)")
        try:
            out[cls] = SAFE.load_pipeline(json.loads(blob.decode("utf8")))
        except (ValueError, KeyError, TypeError) as exc:
            raise ProfileArtifactError(
                f"{name}: malformed artifact ({type(exc).__name__})"
            ) from exc
    return out


@lru_cache(maxsize=1)
def o_pipelines() -> dict:
    return load_o_pipelines()


def profile_meta(profile: str, *, applied: bool | None = True) -> dict:
    """The ``provenance.decision_profile`` object of a document."""
    if profile == PROFILE_V0:
        return {
            "name": PROFILE_V0,
            "status": "historical",
            "classes": {
                c: {"route": "engine_v0_window_events"}
                for c in ("filler_word", "prolongation", "repetition")
            },
            "calibration_claim": "none",
            "pipelines_verified": None,
            "note": "engine decision of every class (window argmax events); identical to contract revisions <= 1.3",
        }
    classes = {
        c: {
            "route": "o_campaign_unit_pipeline",
            "unit": f"ASR segment split at pauses >= {S3F.GAP_S} s (max {S3F.MAX_WORDS} words)",
            "artifact": f"{c}.json",
            "artifact_sha256": PINNED_SHA256[f"{c}.json"],
            "source_pickle_sha256": SOURCE_PICKLE_SHA256[c],
            "pipeline_version": "o-campaign/1",
            "asr_profile_trained_on": "p0",
        }
        for c in O_CLASSES
    }
    classes["repetition"] = {"route": "engine_v0_window_events"}
    return {
        "name": PROFILE_O,
        "status": "exploratory",
        "classes": classes,
        "calibration_claim": "none",
        "pipelines_verified": None if applied is None else bool(applied),
        "note": "product default of the service; the library default stays v0. Repetition keeps the engine decision.",
    }


def warning_for(profile: str) -> dict | None:
    if profile != PROFILE_O:
        return None
    return {
        "code": "decision_profile_exploratory",
        "message": "Decision profile o2026_11 (service default): filler_word and prolongation come from exploratory "
        "pipelines selected by nested cross-validation on DIHANA folds 1-4 (configuration unstable across folds, weak V0 "
        "baseline, no external gain demonstrated). Scores are model outputs, not calibrated probabilities. Units are ASR "
        "segments split at pauses, whereas training used DIHANA turns. Repetition keeps the engine decision. "
        "Use decision_profile=v0 to reproduce the historical output.",
    }


# ----------------------------------------------------------------------------------------------- unit records
def _clip_range(seg: list[dict], n: int, sr: int) -> tuple[int, int]:
    a = max(0, int(round((seg[0]["start"] - MARGIN_S) * sr)))
    b = min(n, int(round((seg[-1]["end"] + MARGIN_S) * sr)))
    return a, max(b, a + 1)


def unit_records(
    engine, audio: np.ndarray, sample_rate: int, words: list[dict]
) -> tuple[list[dict], list[dict]]:
    """One record (``optimize.data`` layout) per pause-delimited ASR segment, plus its ``{start, end}`` span.

    Each segment becomes a standalone clip (like a DIHANA turn): words are re-timed to the clip, the engine builds the
    window probabilities / pooled vector / pause descriptors for that clip (``DisfluencyEngine.unit_features``)."""
    recs, spans = [], []
    for seg in S3F.segments_from_words(words):
        a, b = _clip_range(seg, len(audio), sample_rate)
        t0 = a / sample_rate
        rel = []
        for w in seg:
            w2 = dict(w)
            w2["start"] = max(0.0, float(w["start"]) - t0)
            w2["end"] = max(w2["start"], float(w["end"]) - t0)
            rel.append(w2)
        f = engine.unit_features(audio[a:b], rel)
        recs.append(
            {
                "probs": f["probs"],
                "pooled": f["pooled"],
                "pause": f["pause"],
                "n_samples": f["n_samples"],
                "words": {"p0": rel},
                "text": {"p0": " ".join(w["text"] for w in seg)},
                "acoustic": None,
            }
        )
        spans.append({"start": float(seg[0]["start"]), "end": float(seg[-1]["end"])})
    return recs, spans


def profile_events(pipelines: dict, recs: list[dict], spans: list[dict]) -> list[dict]:
    """Positive units of each pipelined class as raw event dicts (analyzer layout, plus ``_decision``)."""
    out = []
    for cls in O_CLASSES:
        if not recs:
            break
        pipe = pipelines[cls]
        pred = pipe.predict(recs)
        for i, (r, sp) in enumerate(zip(recs, spans)):
            if not int(pred["decision"][i]):
                continue
            score = float(np.clip(pred["score"][i], 0.0, 1.0))
            out.append(
                {
                    "class": cls,
                    "start_s": round(sp["start"], 3),
                    "end_s": round(sp["end"], 3),
                    "score": score,
                    "n_windows": int(len(r["probs"])),
                    "text": r["text"]["p0"],
                    "decision": {
                        "state": str(pred["state"][i]),
                        "estimate": round(
                            float(np.clip(pred["calibrated"][i], 0.0, 1.0)), 6
                        ),
                        "lo": round(float(pipe.band["lo"]), 6),
                        "hi": round(float(pipe.band["hi"]), 6),
                        "basis": "o_campaign_unit_pipeline_inner_band_uncalibrated",
                        "calibration_claim": "none",
                    },
                }
            )
    return out


def apply_profile(
    engine, audio: np.ndarray, sample_rate: int, words: list[dict] | None
) -> list[dict]:
    """Events of the O pipelines for one recording. Raises ``ProfileArtifactError`` if an artifact was altered."""
    pipes = o_pipelines()
    recs, spans = unit_records(engine, audio, sample_rate, words or [])
    return profile_events(pipes, recs, spans)
