"""Frozen S6 gate and analysis constants.

Fixed BEFORE any metric was computed (see results/mejora_objetivos_20261101/S6/PROTOCOL.md). The values are
a-priori engineering choices with the justification given in the protocol; they were not tuned on labels or on
classifier results. Changing any value requires a new, dated protocol version.
"""

from __future__ import annotations

import hashlib
import json

CONFIG_VERSION = "s6-gate-1.1-A1"

GATE = {
    # unit eligibility
    "min_unit_s": 1.5,
    # C1: the dominant face of the unit is present and roughly frontal in >= 80 % of the frames
    "min_face_coverage": 0.80,
    "frontal_yaw_deg": 45.0,
    "frontal_pitch_deg": 35.0,
    # C2: a second face present in >= 25 % of the unit's frames means the active speaker is ambiguous
    "max_second_face_share": 0.25,
    # C3: identity consistency (landmark-ratio descriptor drift between halves, in robust SDs)
    "max_identity_drift": 3.0,
    # C4: audio-visual synchrony
    "lag_search_frames": 10,  # +-400 ms at 25 fps
    "video_max_abs_offset_ms": 120.0,  # ITU-R BT.1359 style acceptability, tightened
    "video_max_ci_width_ms": 200.0,
    "video_min_units_for_offset": 3,  # amendment A1 (was 5)
    "video_min_pooled_frames": 150,  # amendment A1: pooled valid frames (6 s at 25 fps)
    "unit_min_r": 0.30,
    "unit_surrogate_alpha": 0.05,
    "n_surrogates": 100,
    "n_offset_bootstrap": 500,
    "offset_ci": 0.90,
    "min_overlap_frames": 20,
}

ANALYSIS = {
    "negatives_per_positive_cap": 3.0,  # deterministic sha1 subsample of admitted negatives if above
    "cv_folds": 5,
    "cv_repeats": 5,
    "bootstrap_B": 2000,
    "equivalence_margin_f1": 0.05,
    "min_admitted_positives": 40,
    "min_positive_speakers": 10,
    "dominant_speakers": ["epic_actor_010", "epic_actor_173"],
    "logreg_C": 1.0,
    "speaker_baseline_min_frames": 250,  # frames of admitted units needed to estimate a speaker baseline (10 s at 25 fps)
    "seed": 20261101,
}


def config_hash() -> str:
    blob = json.dumps({"v": CONFIG_VERSION, "gate": GATE, "analysis": ANALYSIS}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()
