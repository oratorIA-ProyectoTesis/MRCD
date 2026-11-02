"""Frozen constants of the technical-validation protocol (their hash goes into the pre-registration).

This track produces NO accuracy evidence: there are no human labels, so no precision, recall or F1 can be computed
or claimed from these recordings.
"""

from __future__ import annotations

import hashlib
import json

CONFIG_VERSION = "viab-technical-1.0"

TECHNICAL = {
    # recording design
    "min_speakers": 10,
    "min_total_speech_minutes": 30.0,
    "target_total_speech_minutes": 45.0,
    "min_recording_s": 120.0,
    # statistics
    "bootstrap_B": 2000,
    "ci_level": 0.95,
    "seed": 20261102,
    # pre-registered technical criteria (decision rules in PREREGISTRO_TECNICO.md)
    "T1_min_admitted_fraction": 0.80,  # share of 3 s segments admitted by the FROZEN S6 gate
    "T2_min_sync_ok_fraction": 0.80,  # recordings whose AV offset is estimable and within the gate limits
    "T3_min_median_face_present_fraction": 0.90,
    "T4_min_audio_ok_fraction": 0.90,  # recordings passing level / clipping / noise-floor checks
    "T5_min_analysis_success_fraction": 0.95,  # engine run completes with status ok on unseen speakers
}

SEGMENT_S = 3.0


def config_hash() -> str:
    return hashlib.sha256(
        json.dumps({"v": CONFIG_VERSION, "t": TECHNICAL}, sort_keys=True).encode()
    ).hexdigest()
