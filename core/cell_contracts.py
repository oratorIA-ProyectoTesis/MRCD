"""Strict additive v2 feature/checkpoint contracts; no legacy version inference."""
from copy import deepcopy
import numpy as np
from core.constants import TAXONOMY, TAXONOMY_VERSION, KINESIC_FIELDS
from core.temporal import TEMPORAL_CONTRACT
from core.targets import TARGET_CONTRACT

CELL_FEATURE_VERSION = "owned-cell-features-v2"
CELL_MODEL_IMPLEMENTATION = "cell-fusion-multilabel-v2"
CELL_MODES = ("audio_only", "audio_text", "trimodal")
NUMERIC = ("ac", "pros", "ling", "kin")
SHAPES = {"ac": (300, 5), "ac_mask": (300, 5), "ac_support": (300,), "ac_target": (300,), "ac_pos": (300,),
          "pros": (8,), "pros_mask": (8,), "ling": (40, 12), "ling_mask": (40, 12),
          "lmask": (40,), "lpos": (40,), "ltarget": (40,), "word_indices": (40,), "word_intervals": (40, 2), "word_source_count": (),
          "kin": (30, 12), "kin_mask": (30, 12), "kin_support": (30,), "kpos": (30,), "ktarget": (30,),
          "real_frame_count": (), "has_video": (), "media_duration_ms": (), "owned_start_ms": (),
          "owned_end_ms": (), "nominal_center_ms": ()}


def cell_feature_contract():
    return dict(version=CELL_FEATURE_VERSION, temporal=TEMPORAL_CONTRACT, target=TARGET_CONTRACT,
                taxonomy_version=TAXONOMY_VERSION, taxonomy=list(TAXONOMY),
                shapes={k: list(v) for k, v in SHAPES.items()},
                acoustic_hop_ms=10, visual_hop_ms=100, actual_duration_authoritative=True,
                linguistic_interval="[nominal_center-5000,nominal_center+5000)_ms_plus_owned_overlap",
                truncation="owned_first_then_distance_start_end_original_index_chronological_v2",
                overflow="target_word_overflow_error", masks="per_measure_before_arithmetic_v2",
                acoustic_fields=["log_rms", "f0_st_100hz", "voiced_prob", "vad", "adjacent_df0"],
                prosody_fields=["sil_dur", "sil_start_relative_owned_s", "f0_slope_pre", "f0_range_pre",
                                "f0_level_pre", "f0_reset", "rms_slope_pre", "final_lengthen"],
                linguistic_fields=["is_filler", "is_function_word", "is_edit_term", "repeats_prev",
                                   "is_fragment", "clause_end", "log_duration", "asr_prob", "gap_before",
                                   "gap_after", "owned_zone", "dangling_gap_250ms"],
                kinesic_fields=list(KINESIC_FIELDS), visual_min_real_frames=24,
                positions="actual_seconds_relative_nominal_center", live_supported=False)


def validate_cell_features(data):
    if data.get("feature_contract") != cell_feature_contract():
        raise ValueError("v2 feature contract mismatch; legacy/same-width data is not compatible")
    if "ac" not in data or np.asarray(data["ac"]).ndim != 3:
        raise ValueError("missing v2 acoustic batch")
    n = len(data["ac"])
    for key, shape in SHAPES.items():
        if key not in data or np.asarray(data[key]).shape != (n, *shape):
            raise ValueError(f"{key} must have shape {(n, *shape)}")
        if key.endswith("_mask") or key in ("ac_support", "kin_support", "ac_target", "lmask", "ltarget", "ktarget", "has_video"):
            if np.asarray(data[key]).dtype != np.bool_:
                raise ValueError(f"{key} must be boolean")
    for key in NUMERIC:
        if not np.isfinite(data[key][data[key + "_mask"]]).all():
            raise ValueError(f"{key}: nonfinite observed measurement")
    for key in ("ac_pos", "lpos", "kpos", "word_intervals"):
        if not np.isfinite(data[key]).all():
            raise ValueError(f"{key}: nonfinite position")
    if np.any(data["ac_mask"] & ~data["ac_support"][..., None]) or np.any(data["ac_target"] & ~data["ac_support"]):
        raise ValueError("acoustic masks exceed actual time support")
    if np.any(data["ling_mask"] & ~data["lmask"][..., None]) or np.any(data["ltarget"] & ~data["lmask"]):
        raise ValueError("linguistic masks exceed retained words")
    if not np.array_equal(data["has_video"], data["real_frame_count"] >= 24):
        raise ValueError("has_video disagrees with actual detected frame count")
    if (np.any(data["kin_mask"] & ~data["kin_support"][..., None])
            or not np.array_equal(data["kin_support"].sum(1), data["real_frame_count"])):
        raise ValueError("visual support disagrees with actual detections")
    start, end, duration, center = (data[key] for key in ("owned_start_ms", "owned_end_ms", "media_duration_ms", "nominal_center_ms"))
    if (np.any(start < 0) or np.any(start % 500) or np.any(duration < 0)
            or not np.array_equal(end, np.minimum(start + 500, duration))
            or np.any(end <= start) or not np.array_equal(center, start + 250)):
        raise ValueError("invalid actual-duration/ownership metadata")
    _validate_words(data, start, end, duration, center)
    for pos, support, target in (("ac_pos", "ac_support", "ac_target"), ("kpos", "kin_support", "ktarget")):
        absolute = np.round(data[pos] * 1000 + center[:, None], 6)
        actual = data[support]
        if np.any(actual & ((absolute < np.maximum(0, center - 1500)[:, None])
                            | (absolute >= np.minimum(duration, center + 1500)[:, None]))):
            raise ValueError(f"{pos}: observed support exceeds real media/evidence")
        expected = actual & (absolute >= start[:, None]) & (absolute < end[:, None])
        if not np.array_equal(data[target], expected):
            raise ValueError(f"{target}: ownership disagrees with actual timestamps")
    return n


def _validate_words(data, start, end, duration, center):
    """Only active slots carry words; neutral masked storage is not an observation."""
    active = data["lmask"]
    indices, count = data["word_indices"], data["word_source_count"]
    if (not np.issubdtype(indices.dtype, np.integer) or not np.issubdtype(count.dtype, np.integer)
            or np.any(count < 0) or np.any(active & ((indices < 0) | (indices >= count[:, None])))):
        raise ValueError("active word identity is outside retained source words")
    left, right = data["word_intervals"][..., 0], data["word_intervals"][..., 1]
    midpoint = (left + right) / 2
    if np.any(active & ((left < 0) | (right <= left) | (right > duration[:, None]))):
        raise ValueError("active word interval outside real media")
    expected_position = (midpoint - center[:, None]) / 1000
    if not np.allclose(data["lpos"][active], expected_position[active], rtol=0, atol=1e-8):
        raise ValueError("active word position disagrees with actual interval/nominal center")
    target = active & (np.maximum(left, start[:, None]) < np.minimum(right, end[:, None]))
    if not np.array_equal(data["ltarget"], target):
        raise ValueError("ltarget disagrees with owned positive interval overlap")
    in_context = (midpoint >= center[:, None] - 5000) & (midpoint < center[:, None] + 5000)
    if np.any(active & ~in_context & ~target):
        raise ValueError("active word lies outside centered context and owned interval")
    for row in range(len(active)):
        kept = np.flatnonzero(active[row])
        ids = indices[row, kept].tolist()
        order = [(left[row, slot], right[row, slot], int(indices[row, slot])) for slot in kept]
        if len(set(ids)) != len(ids) or order != sorted(order):
            raise ValueError("active word identities must be unique and chronologically ordered")


def effective_masks(data, mode="trimodal", modalities=None):
    if mode not in CELL_MODES:
        raise ValueError("unknown cell mode")
    enabled = dict(ac=True, pros=True, ling=mode != "audio_only", kin=mode == "trimodal")
    if modalities is not None:
        enabled = {key: value and bool(modalities.get(key, False)) for key, value in enabled.items()}
    masks = {key: np.asarray(data[key + "_mask"], bool).copy() & enabled[key] for key in NUMERIC}
    masks["kin"] &= np.asarray(data["has_video"])[:, None, None]
    return masks


def input_evaluable(data, mode="trimodal", modalities=None):
    masks = effective_masks(data, mode, modalities)
    return ((masks["ac"].any(2) & data["ac_target"]).any(1)
            | (masks["ling"].any(2) & data["ltarget"]).any(1)
            | (masks["kin"].any(2) & data["ktarget"]).any(1))
