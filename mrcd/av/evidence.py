"""Opt-in, EXPERIMENTAL gated video evidence for spans of a recording ("av_gated").

Nothing in the default path imports or calls this module. For every span it returns either gated observable
features or `available: False` with a reason (never zeros). It does not classify anything and never infers a
psychological state; the features are lip aperture, jaw opening and head-movement summaries.
"""

from __future__ import annotations

import numpy as np

from mrcd.av.config import ANALYSIS, config_hash
from mrcd.av.faces import SHAPE_DIM, FaceSeries, extract_video, to_face_series  # noqa: F401
from mrcd.av.features import V_NAMES, speaker_baseline, visual_features
from mrcd.av.gate import analyze_video, speech_envelope

EXPERIMENTAL_NOTE = (
    "experimental: audio-visual evidence gated by a technical synchrony and active-speaker check "
    f"(config {config_hash()[:12]}); observable geometry only; not validated as a detector; not a default"
)
_SIGNAL_COLS = {"ap": 0, "jaw": 1, "pucker": 2, "pitch": 3, "yaw": 4, "roll": 5}


def evidence_from_tracks(
    tracks: dict,
    audio: np.ndarray,
    sr: int,
    spans: list[tuple[float, float]],
    seed: int = 0,
):
    """Gate and describe spans given stored face tracks (`extract_video` output) and the recording's audio."""
    fps, n = float(tracks["fps"]), int(tracks["n"])
    fs = to_face_series(tracks, speech_envelope(audio, sr, fps, n))
    rows, vrow = analyze_video(
        fs, [(f"span{i}", a, b) for i, (a, b) in enumerate(spans)], seed=seed
    )
    out, sigs, adm = [], [], []
    for r, (a, b) in zip(rows, spans):
        if not r["admitted"]:
            out.append(
                {
                    "t0": a,
                    "t1": b,
                    "available": False,
                    "reason": r["primary_reason"],
                    "reasons": r["reasons"],
                }
            )
            sigs.append(None)
            adm.append(False)
            continue
        i0, i1 = max(int(a * fps), 0), min(int(np.ceil(b * fps)), n)
        v, ok = tracks["vec"][r["track"], i0:i1], tracks["present"][r["track"], i0:i1]
        sigs.append({k: np.where(ok, v[:, j], np.nan) for k, j in _SIGNAL_COLS.items()})
        out.append({"t0": a, "t1": b})
        adm.append(True)
    pooled = {
        k: np.concatenate(
            [s[k][np.isfinite(s["ap"])] for s in sigs if s is not None]
            or [np.array([])]
        )
        for k in _SIGNAL_COLS
    }
    ang = [
        np.stack([s[k] for k in ("pitch", "yaw", "roll")], 1)
        for s in sigs
        if s is not None
    ]
    pooled["speed"] = np.concatenate(
        [np.sqrt(np.nansum(np.diff(x, axis=0) ** 2, 1)) * fps for x in ang]
        or [np.array([])]
    )
    base = (
        speaker_baseline(pooled, ANALYSIS["speaker_baseline_min_frames"])
        if len(pooled["ap"])
        else None
    )
    for i, o in enumerate(out):
        if not adm[i]:
            continue
        r = rows[i]
        if base is None:
            o.update(available=False, reason="speaker_baseline_insufficient")
            continue
        f = visual_features(sigs[i], fps, base)
        if f is None:
            o.update(available=False, reason="visual_features_unavailable")
            continue
        o.update(
            available=True,
            features=dict(zip(V_NAMES, [float(x) for x in f])),
            gate={
                k: r[k]
                for k in (
                    "face_coverage",
                    "identity_drift",
                    "local_offset_ms",
                    "r_at_video_offset",
                    "surrogate_p",
                )
            },
        )
    for o in out:
        o["experimental"] = True
        o["note"] = EXPERIMENTAL_NOTE
    return out, vrow


def evidence_from_video(
    video_path: str, audio: np.ndarray, sr: int, spans, face_model: str, seed: int = 0
):
    """Convenience wrapper: track faces in `video_path` then gate. A missing file is 'not available: reason'."""
    import os

    if not os.path.exists(video_path):
        return [
            {
                "t0": a,
                "t1": b,
                "available": False,
                "reason": "video_file_not_found",
                "experimental": True,
                "note": EXPERIMENTAL_NOTE,
            }
            for a, b in spans
        ], {"reason": "video_file_not_found"}
    return evidence_from_tracks(
        extract_video(video_path, face_model), audio, sr, spans, seed
    )
