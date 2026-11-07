"""Pins the single neutral-pause / candidate-silence placeholder (0.60 s). It is NOT calibrated: no human pause
labels exist. Changing it requires an explicit decision and a doc update (docs/contracts/DISAMBIGUATION_SPEC.md)."""

import re
from pathlib import Path

from mrcd.constants import H
from mrcd.service import s3_features

ROOT = Path(__file__).resolve().parents[1]


def test_silence_min_is_pinned():
    assert H["silence_min_s"] == 0.60


def test_service_segmentation_uses_same_value():
    assert s3_features.GAP_S == H["silence_min_s"]


def test_spec_declares_one_uncalibrated_value_and_no_400ms():
    t = (ROOT / "docs" / "contracts" / "DISAMBIGUATION_SPEC.md").read_text("utf8")
    assert "uncalibrated placeholder" in t and "0.60 s" in t
    assert not re.search(r"\b400 ?ms\b", t)
