"""Versioned, dependency-light boundaries for extracted features and evidence.

Arrays remain NPZ/JSON so research artifacts stay inspectable without a server.
Validation happens when artifacts enter training or inference, not in the DSP loop.
"""
from __future__ import annotations

import json
from collections.abc import Mapping

import numpy as np

from core.constants import (ACOUSTIC_HOP_S, KINESIC_DIM, KINESIC_FIELDS, LABELS, STRIDE_S,
                            TAXONOMY, VIDEO_FPS, WINDOW_S)

FEATURE_SCHEMA_VERSION = "1.2.0"
RAW_FEATURE_SCHEMA_VERSION = "1.0.0"
FUSION_IMPLEMENTATION = "reconstructed-v1"
MODES = ("audio_only", "audio_text", "trimodal")
EVIDENCE_SOURCES = ("auto", "human", "gold_llm")
NUMERIC_FEATURES = ("ac", "pros", "ling", "kin")
BATCH_FEATURES = ("ac", "pros", "ling", "lpos", "lmask", "kin", "kmask", "has_video")


def raw_feature_contract() -> dict:
    """Contract for the extracted signals before window assembly."""
    return {"version": RAW_FEATURE_SCHEMA_VERSION,
            "acoustic_hop_s": ACOUSTIC_HOP_S, "acoustic_width": 5,
            "video_fps": VIDEO_FPS, "kinesic_fields": list(KINESIC_FIELDS),
            "word_feature_base_width": 10}


def validate_raw_features(data: Mapping, *, path: str = "raw feature NPZ") -> dict:
    """Reject unversioned/stale extraction rather than stamping it as current."""
    required = ("raw_feature_schema", "raw_contract", "raw_extractor", "raw_extractor_config",
                "ac_frames", "word_feats", "k_times", "k_vectors")
    missing = [key for key in required if key not in data]
    if missing:
        raise ValueError(f"{path}: missing raw provenance {missing}; re-extract with "
                         "scripts/extract_and_label.py --force or scripts/ingest_single_video.py --force")
    if _scalar(data["raw_feature_schema"]) != RAW_FEATURE_SCHEMA_VERSION:
        raise ValueError(f"{path}: unsupported raw feature schema; re-extract with --force")
    try:
        contract = json.loads(_scalar(data["raw_contract"]))
        config = json.loads(_scalar(data["raw_extractor_config"]))
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{path}: invalid raw extractor provenance; re-extract with --force") from exc
    if contract != raw_feature_contract():
        raise ValueError(f"{path}: raw extractor contract differs from current code; re-extract with --force")
    producer = _scalar(data["raw_extractor"])
    if producer not in ("extract_and_label", "ingest_single_video") or not isinstance(config, dict):
        raise ValueError(f"{path}: unknown raw extractor provenance; re-extract with --force")
    if not {"asr_model", "variety", "vad_backend", "face_tracking_method"}.issubset(config):
        raise ValueError(f"{path}: incomplete raw extractor configuration; re-extract with --force")
    ac, words = np.asarray(data["ac_frames"]), np.asarray(data["word_feats"])
    kt, kv = np.asarray(data["k_times"]), np.asarray(data["k_vectors"])
    if ac.ndim != 2 or ac.shape[1] != 5 or words.ndim != 2 or words.shape[1] < 10:
        raise ValueError(f"{path}: raw acoustic/linguistic dimensions differ from contract")
    if kt.ndim != 1 or kv.shape != (len(kt), KINESIC_DIM):
        raise ValueError(f"{path}: raw kinesic dimensions differ from contract")
    return {"schema": RAW_FEATURE_SCHEMA_VERSION, "extractor": producer, "config": config}


def feature_contract() -> dict:
    """Serializable shape/time/taxonomy contract stored with new artifacts."""
    return {"version": FEATURE_SCHEMA_VERSION, "labels": list(LABELS),
            "taxonomy": list(TAXONOMY), "window_s": WINDOW_S, "stride_s": STRIDE_S,
            "acoustic_hop_s": ACOUSTIC_HOP_S, "video_fps": VIDEO_FPS,
            "kinesic_dim": KINESIC_DIM, "evidence_sources": list(EVIDENCE_SOURCES),
            "auto_labels_are_ground_truth": False,
            "missingness": {"linguistic": "lmask", "visual": "kmask",
                            "visual_window": "has_video", "prosodic_reset": "reset_valid"}}


def _scalar(value) -> str:
    return str(np.asarray(value).item())


def validate_features(data: Mapping, *, expected_dims: Mapping[str, int] | None = None,
                      require_labels: bool = False, mode: str = "trimodal") -> int:
    """Fail early on a malformed feature batch, including mask/position shape."""
    if mode not in MODES:
        raise ValueError(f"unknown MRCD mode {mode!r}")
    required = ["ac"]
    if mode != "audio_only":
        required += ["ling", "lpos", "lmask"]
    if mode == "trimodal":
        required += ["kin", "kmask"]
        if "has_video" not in data:
            raise ValueError("trimodal features require has_video from >=24 real visual frames; "
                             "rebuild windows from versioned raw extraction (legacy kmask cannot establish coverage)")
    if expected_dims and int(expected_dims.get("pros", 0)):
        required.append("pros")
    missing = [key for key in required if key not in data]
    if missing:
        raise ValueError(f"missing MRCD feature keys: {missing}")
    ac = np.asarray(data["ac"])
    if ac.ndim != 3 or ac.shape[1:] != (int(round(WINDOW_S / ACOUSTIC_HOP_S)), 5):
        raise ValueError("ac must have shape [N, 300, 5]")
    n = ac.shape[0]
    if "pros" in data:
        pros = np.asarray(data["pros"])
        if pros.ndim != 2 or pros.shape[0] != n:
            raise ValueError("pros must have shape [N, features]")
    if "ling" in data:
        ling = np.asarray(data["ling"])
        if ling.ndim != 3 or ling.shape[0] != n:
            raise ValueError("ling must have shape [N, words, features]")
        for key in ("lmask", "lpos"):
            if key in data and np.asarray(data[key]).shape != ling.shape[:2]:
                raise ValueError(f"{key} must have shape {ling.shape[:2]}")
    if "kin" in data:
        kin = np.asarray(data["kin"])
        if kin.shape != (n, int(round(WINDOW_S * VIDEO_FPS)), KINESIC_DIM):
            raise ValueError("kin must have shape [N, 30, 12]")
        if "kmask" in data and np.asarray(data["kmask"]).shape != kin.shape[:2]:
            raise ValueError(f"kmask must have shape {kin.shape[:2]}")
    if "has_video" in data and np.asarray(data["has_video"]).shape != (n,):
        raise ValueError("has_video must have shape [N]")
    for key in NUMERIC_FEATURES:
        if key not in data:
            continue
        values = np.asarray(data[key])
        mask_name = {"ling": "lmask", "kin": "kmask"}.get(key)
        observed = values[np.asarray(data[mask_name], bool)] if mask_name and mask_name in data else values
        if not np.isfinite(observed).all():
            raise ValueError(f"{key} has nonfinite observed features")
    if expected_dims:
        for key, width in expected_dims.items():
            if key in data and key in NUMERIC_FEATURES and np.asarray(data[key]).shape[-1] != width:
                raise ValueError(f"{key} width differs from checkpoint: expected {width}")
    if require_labels:
        if "labels" not in data:
            raise ValueError("training batch needs labels")
        labels = tuple(x.decode() if isinstance(x, bytes) else str(x) for x in data["labels"])
        if not labels or len(set(labels)) != len(labels) or not set(labels).issubset(LABELS):
            raise ValueError("labels differ from the canonical MRCD taxonomy")
        if "feature_schema" in data and labels != LABELS:
            raise ValueError("versioned labels differ from the canonical MRCD taxonomy")
        if "y" not in data or np.asarray(data["y"]).shape != (n,):
            raise ValueError("training batch needs y with shape [N]")
        if np.any((np.asarray(data["y"]) < 0) | (np.asarray(data["y"]) >= len(labels))):
            raise ValueError("y contains a class outside the canonical taxonomy")
        if "label_source" not in data or _scalar(data["label_source"]) not in EVIDENCE_SOURCES:
            raise ValueError(f"label_source must be one of {EVIDENCE_SOURCES}")
        if "feature_schema" in data and _scalar(data["label_source"]) == "gold_llm":
            judge_fields = ("judge_n_frames", "judge_visual_evidence_used", "judge_provider",
                            "judge_model", "judge_input_fingerprint", "judge_fallback_note")
            absent = [key for key in judge_fields if key not in data]
            if absent:
                raise ValueError(f"versioned GOLD is missing judge provenance {absent}; rebuild windows")
            if any(np.asarray(data[key]).shape != (n,) for key in judge_fields):
                raise ValueError("judge provenance fields must have shape [N]")
            frame_counts = np.asarray(data["judge_n_frames"])
            if np.any(frame_counts < 0) or not np.array_equal(
                    np.asarray(data["judge_visual_evidence_used"], bool), frame_counts > 0):
                raise ValueError("judge visual provenance conflicts with n_frames")
            if any(np.any(np.asarray(data[key]).astype(str) == "") for key in
                   ("judge_provider", "judge_model", "judge_input_fingerprint")):
                raise ValueError("versioned GOLD needs judge provider, model and input fingerprint")
    if "feature_schema" in data and _scalar(data["feature_schema"]) != FEATURE_SCHEMA_VERSION:
        raise ValueError(f"unsupported feature schema {_scalar(data['feature_schema'])!r}")
    return n


def validate_checkpoint(checkpoint: Mapping) -> dict:
    """Validate versioned checkpoints and explicitly identify legacy envelopes.

    Legacy metadata is accepted for the existing round-trip test, not a promise
    that a historical state_dict matches this newly reconstructed architecture.
    """
    for key in ("state_dict", "mode", "labels", "dims", "norm"):
        if key not in checkpoint:
            raise ValueError(f"checkpoint missing {key}")
    if checkpoint["mode"] not in MODES:
        raise ValueError(f"unknown checkpoint mode {checkpoint['mode']!r}")
    labels = list(checkpoint["labels"])
    if not labels or len(set(labels)) != len(labels):
        raise ValueError("checkpoint labels must be nonempty and unique")
    dims = checkpoint["dims"]
    if any(int(dims.get(key, 0)) <= 0 for key in ("ac", "ling", "d")):
        raise ValueError("checkpoint dimensions ac, ling, d must be positive")
    for key, pair in checkpoint["norm"].items():
        if key not in NUMERIC_FEATURES or len(pair) != 2:
            raise ValueError(f"invalid normalization entry {key!r}")
        if len(pair[0]) != len(pair[1]):
            raise ValueError(f"normalization width mismatch for {key}")
        if key in dims and len(pair[0]) != int(dims[key]):
            raise ValueError(f"normalization does not match {key} feature width")
    contract = checkpoint.get("feature_contract")
    if contract is None:
        return {"contract_status": "legacy_unversioned"}
    if contract.get("version") != FEATURE_SCHEMA_VERSION:
        raise ValueError(f"unsupported checkpoint feature schema {contract.get('version')!r}")
    if checkpoint.get("fusion_implementation") != FUSION_IMPLEMENTATION:
        raise ValueError("checkpoint fusion implementation is not compatible with this model")
    if labels != list(LABELS) or contract.get("labels") != list(LABELS):
        raise ValueError("checkpoint taxonomy differs from canonical MRCD labels")
    required_norm = {"ac", "ling", "kin"}
    if int(dims.get("pros", 0)):
        required_norm.add("pros")
    if not required_norm.issubset(checkpoint["norm"]):
        raise ValueError(f"checkpoint missing normalization for {sorted(required_norm - checkpoint['norm'].keys())}")
    if checkpoint.get("provenance", {}).get("label_source") not in EVIDENCE_SOURCES:
        raise ValueError("checkpoint needs a recognized evidence source")
    if any(contract.get(key) != value for key, value in feature_contract().items()
           if key not in ("version", "labels")):
        raise ValueError("checkpoint feature contract differs from current extractor")
    return {"contract_status": "versioned", "feature_schema": FEATURE_SCHEMA_VERSION}
