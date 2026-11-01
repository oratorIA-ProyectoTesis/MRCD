"""Opt-in S3 service features: experimental P1 text route and the optional three-state decision.

Everything here is EXPERIMENTAL and off by default. Scores are never called calibrated probabilities: the
``decision.estimate`` is a development-fitted Platt mapping with ``calibration_claim: "none"``.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from mrcd import calibration as C
from mrcd.fusion_text import TextRoute

DATA = Path(__file__).resolve().parent / "experimental"
ROUTE_FILE = DATA / "rep_ctx_p1_route.json"
BANDS_FILE = DATA / "decision_bands_s3.json"
GAP_S = 0.6  # a pause of at least this long closes a text segment
MAX_WORDS = 30


@lru_cache(maxsize=1)
def load_route() -> TextRoute:
    return TextRoute.load(ROUTE_FILE)


@lru_cache(maxsize=1)
def load_bands() -> dict:
    return json.loads(BANDS_FILE.read_text("utf8"))


def route_info() -> dict:
    r = load_route()
    return {
        "name": "rep_ctx_p1",
        "experimental": True,
        "task": r.task,
        "trained_on_asr_profile": r.asr_profile,
        "note": r.train_note,
        "unit": "turn-level model applied to ASR segments split at pauses >= %.1f s"
        % GAP_S,
        "validated_outside_dihana": False,
    }


def segments_from_words(
    words: list[dict], gap_s: float = GAP_S, max_words: int = MAX_WORDS
) -> list[list[dict]]:
    segs: list[list[dict]] = []
    cur: list[dict] = []
    for w in sorted(words, key=lambda x: (x["start"], x["end"])):
        if cur and (w["start"] - cur[-1]["end"] >= gap_s or len(cur) >= max_words):
            segs.append(cur)
            cur = []
        cur.append(w)
    if cur:
        segs.append(cur)
    return segs


def text_route_events(words: list[dict], fused: bool = False) -> list[dict]:
    """Positive segments of the P1 text route: dicts(start, end, score, text, n_words)."""
    route = load_route()
    segs = segments_from_words(words)
    if not segs:
        return []
    texts = [" ".join(w["text"] for w in s) for s in segs]
    scores, dec = route.decide(texts, fused=fused)
    return [
        {
            "start": s[0]["start"],
            "end": s[-1]["end"],
            "score": float(sc),
            "text": t,
            "n_words": len(s),
        }
        for s, sc, t, d in zip(segs, scores, texts, dec)
        if d
    ]


def decision_for(cls_key: str, score: float) -> dict | None:
    """Three-state decision for a score of class ``cls_key`` ('filler_word', ..., 'text_route_repetition')."""
    entry = load_bands()["classes"].get(cls_key)
    if entry is None:
        return None
    cal = C.Calibrator.from_dict(entry["calibrator"])
    p = float(cal.apply([score])[0])
    return {
        "state": C.three_state(p, entry["lo"], entry["hi"]),
        "estimate": round(min(max(p, 0.0), 1.0), 6),
        "lo": round(float(entry["lo"]), 6),
        "hi": round(float(entry["hi"]), 6),
        "basis": "platt_fit_on_dihana_folds_1_4_cross_fitted_band_experimental",
        "calibration_claim": "none",
    }
