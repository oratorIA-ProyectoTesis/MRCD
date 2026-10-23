"""Acquisition != effective model input != assessed quality (additive contract)."""
import numpy as np
from core.cell_contracts import NUMERIC, effective_masks, validate_cell_features
from core.cell_normalization import validate_cell_stats

EVIDENCE_SCHEMA = "mrcd-cell-evidence-v1"


def cell_evidence(features, *, mode, capabilities, stats=None):
    n = validate_cell_features(features)
    modalities = capabilities.get("modalities", {})
    masks = effective_masks(features, mode, modalities)
    if stats is not None:
        validate_cell_stats(stats)
        masks = {key: mask & (np.asarray(stats[key]["count"]) > 0) for key, mask in masks.items()}
    rows = []
    for i in range(n):
        acquired, effective, quality = {}, {}, {}
        for key in NUMERIC:
            raw = features[key + "_mask"][i]
            mask = masks[key][i]
            reasons = []
            if not raw.any(): reasons.append("no_acquired_measurements")
            if key == "ling" and mode == "audio_only" or key == "kin" and mode != "trimodal": reasons.append("mode_disabled")
            if not modalities.get(key, False): reasons.append("modality_not_trained")
            if key == "kin" and not features["has_video"][i]: reasons.append("insufficient_real_visual_frames")
            if raw.any() and stats is not None and not (raw & (np.asarray(stats[key]["count"]) > 0)).any():
                reasons.append("no_trained_measurements")
            owned = {"ac":"ac_target", "ling":"ltarget", "kin":"ktarget"}.get(key)
            own = bool((mask.any(-1) & features[owned][i]).any()) if owned else None
            acquired[key] = dict(available=bool(raw.any()), observed_measurements=int(raw.sum()), reasons=[] if raw.any() else ["no_acquired_measurements"])
            effective[key] = dict(available=bool(mask.any()), owned_available=own, observed_measurements=int(mask.sum()), reasons=reasons)
            quality[key] = dict(value=None, reason="quality_not_assessed")
        rows.append(dict(schema=EVIDENCE_SCHEMA, acquired=acquired, effective=effective, quality=quality))
    return rows
