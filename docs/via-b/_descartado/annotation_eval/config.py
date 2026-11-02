"""Frozen constants of the confirmatory evaluation (hash goes into the pre-registration)."""

from __future__ import annotations

import hashlib
import json

CONFIG_VERSION = "viab-confirmatory-1.0"

CONFIRMATORY = {
    # --- matching / units
    "primary_iou": 0.5,
    "sensitivity_iou": [0.3],
    "sensitivity_any_overlap": True,
    "segment_s": 3.0,  # presence unit (non-overlapping; tail shorter than 1.5 s dropped)
    "segment_min_tail_s": 1.5,
    "segment_overlap_rule": {"min_overlap_s": 0.30, "min_event_fraction": 0.5},  # min(0.30, 0.5*dur), as in dataset.py
    "segment_ignore_fraction": 0.5,  # segment dropped when >50 % lies in an ignore region
    # --- statistics
    "bootstrap_B": 2000,
    "ci_level": 0.95,
    "seed": 20261102,
    "alpha": 0.05,
    "equivalence_margin_f1": 0.05,
    # --- adequacy (a class below these is reported as exploratory / not concluyente)
    "min_positives_per_class": 40,
    "min_speakers": 10,
    "min_positive_speakers": 10,
    "min_gate_admitted_fraction": 0.80,
    # --- annotation quality gate (computed on the independent, pre-adjudication annotations)
    "min_kappa": 0.60,
    "min_alpha": 0.60,
    "min_event_f1": 0.60,
    "reference_agreement": 0.70,  # original OE4 indicator, reported as reference only
    "agreement_frame_ms": 20,
    # --- original OE4 targets, REFERENCE only (no decision depends on them)
    "reference_f1": {"filler_word": 0.80, "prolongation": 0.80, "repetition": 0.80},
    "reference_coverage_risk": {"coverage": 0.90, "risk": 0.10},
}

#: Systems of the pre-registration. ``rule_lexical`` is the baseline of every paired comparison.
SYSTEMS = ("rule_lexical", "v0", "vc", "ctx_lr_p1", "p1_noloop")
BASELINE_SYSTEM = "rule_lexical"
ABLATION_SYSTEMS = ("at", "atv")  # audio+text vs audio+text+video (gate frozen in mrcd/av/config.py)


def config_hash() -> str:
    blob = json.dumps({"v": CONFIG_VERSION, "c": CONFIRMATORY, "s": SYSTEMS, "a": ABLATION_SYSTEMS}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()
