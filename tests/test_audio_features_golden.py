"""Automated audio->features golden (REV-INT finding 1): ``DisfluencyEngine.unit_features`` on a synthetic clip.

The o2026_11 regression golden is built from stored features; this one covers the path that starts from AUDIO
(acoustic extraction, Silero, windowing, linguistic alignment, head inputs). Regenerate with
``scripts/hardening/make_features_golden.py`` only on purpose.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from tests.golden_features_audio import WORDS, synthetic_clip

ROOT = Path(__file__).resolve().parents[1]
CKPT = ROOT / "models" / "ckpt_audio_text.pt"
GOLDEN = ROOT / "tests" / "data" / "audio_features_golden.json"

pytestmark = pytest.mark.skipif(not CKPT.is_file(), reason="checkpoint not present")


@pytest.fixture(scope="module")
def features():
    from mrcd.engine import DisfluencyEngine

    return DisfluencyEngine.load(CKPT, device="cpu").unit_features(synthetic_clip(), WORDS)


def test_synthetic_clip_is_deterministic():
    a, b = synthetic_clip(), synthetic_clip()
    assert a.dtype == np.float32 and np.array_equal(a, b) and np.isfinite(a).all()


def test_audio_to_features_matches_the_frozen_golden(features):
    g = json.loads(GOLDEN.read_text("utf8"))
    atol = g["atol"]
    assert features["n_samples"] == g["n_samples"]
    for key in ("probs", "pooled"):
        got, want = np.asarray(features[key]), np.asarray(g[key])
        assert got.shape == want.shape, key
        assert np.abs(got - want).max() <= atol, key
    assert set(features["pause"]) == set(g["pause"])
    for k, v in g["pause"].items():
        assert abs(features["pause"][k] - v) <= atol, k


def test_golden_detects_a_changed_input(features):
    g = json.loads(GOLDEN.read_text("utf8"))
    from mrcd.engine import DisfluencyEngine

    other = DisfluencyEngine.load(CKPT, device="cpu").unit_features(synthetic_clip() * 0.5, WORDS)
    assert np.abs(np.asarray(other["pooled"]) - np.asarray(g["pooled"])).max() > g["atol"]
