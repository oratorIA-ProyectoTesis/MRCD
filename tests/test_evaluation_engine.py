"""Evaluator, decoding and extraction-refactor guarantees (no ASR/model downloads)."""

import numpy as np
import pytest

from core import evaluation as ev
from core.constants import LABELS, SAMPLE_RATE
from core.engine import decode, window_tensors
from core.extractors.acoustic import extract_acoustic
from core.extractors.linguistic import Word
from core.live.window_builder import build_window


def e(a, b, label="filler_word", **kw):
    return {"start_ms": a, "end_ms": b, "label": label, **kw}


def test_long_event_cannot_match_five_references():
    ref = [e(i * 1000, i * 1000 + 400) for i in range(5)]
    r = ev.evaluate(ref, [e(0, 5000)])
    assert r["per_class"]["filler_word"]["tp"] == 0 and r["per_class"]["filler_word"]["fn"] == 5
    r = ev.evaluate(ref, [e(0, 5000)], iou_threshold=0.01)
    assert r["per_class"]["filler_word"]["tp"] == 1  # one-to-one even when every pair overlaps


def test_class_and_boundary_errors_are_not_hidden():
    ref = [e(0, 1000), e(5000, 6000, "repetition")]
    hyp = [e(0, 1000, "revision"), e(5600, 7000, "repetition")]
    r = ev.evaluate(ref, hyp)
    assert r["errors"] == {"class": 1, "interval": 1, "unsupported": 0, "missed": 2}
    assert r["boundaries"]["matched_proportion"] == 0


def test_pauses_are_reported_apart_from_disfluencies():
    ref = [e(0, 500), e(1000, 2000, "neutral_pause")]
    hyp = [e(0, 500), e(1000, 2000, "neutral_pause"), e(3000, 4000, "rhetorical_pause")]
    r = ev.evaluate(ref, hyp, evaluable_ms=60000)
    assert r["micro"]["disfluency"]["f1"] == 1.0 and r["micro"]["pause"]["fp"] == 1
    assert r["false_alarms_per_min"] == 0  # pause false alarms are not disfluency false alarms


def test_unreviewed_audio_is_not_negative_and_uncertain_is_not_an_event():
    cov = {"r": [(0, 10000)]}
    hyp = [e(20000, 20500, recording_id="r"), e(100, 600, recording_id="r", decision="uncertain")]
    r = ev.evaluate([], hyp, coverage=cov)
    assert r["n_hyp"] == 0 and r["n_uncertain"] == 1


def test_tolerance_metric_is_secondary_and_separate():
    r = ev.evaluate([e(0, 200)], [e(60, 150)])
    assert r["micro"]["disfluency"]["tp"] == 0 and r["tolerance"]["disfluency"]["tp"] == 1


def test_agreement_counts_omissions_in_kappa_units():
    a = [e(0, 500), e(1000, 1500, "repetition")]
    b = [e(0, 520)]
    g = ev.agreement(a, b)
    assert g["matched"] == 1 and g["existence_agreement"] == pytest.approx(2 / 3)
    assert ("repetition", "none") in {
        (x, y)
        for x in g["confusion"]["labels"]
        for y in g["confusion"]["labels"]
        if g["confusion"]["matrix"][g["confusion"]["labels"].index(x)][g["confusion"]["labels"].index(y)]
    }


def test_speaker_bootstrap_and_paired_difference():
    ref = [e(i * 1000, i * 1000 + 500, speaker_id=f"s{i % 3}") for i in range(9)]
    good, bad = list(ref), ref[:3]
    f1 = lambda r, h: ev.evaluate(r, h)["micro"]["disfluency"]["f1"]  # noqa: E731
    ci = ev.bootstrap_ci(ref, [good], f1, n=50)
    assert ci["low"] == ci["high"] == 1.0 and ci["n_speakers"] == 3
    diff = ev.bootstrap_ci(ref, [good, bad], lambda r, a, b: f1(r, a) - f1(r, b), n=50)
    assert diff["low"] > 0


def test_calibration_threshold_and_coverage_use_fixed_units():
    scores, correct = [0.9, 0.8, 0.6, 0.4], [1, 1, 0, 0]
    assert ev.pick_threshold(scores, correct, max_risk=0.0) == 0.8
    cr = ev.coverage_risk(scores, correct, 0.8)
    assert cr == {"threshold": 0.8, "n_units": 4, "coverage": 0.5, "risk": 0.0, "abstained": 2}
    assert ev.calibration(np.array([0.9, 0.1]), np.array([1, 0]))["brier"] == pytest.approx(0.01)
    probs = np.array([[0.9, 0.1], [0.8, 0.2], [0.7, 0.3], [0.6, 0.4]])
    assert ev.fit_temperature(probs, np.array([0, 1, 0, 1])) > 1  # overconfident -> soften


def test_active_selection_refuses_test_and_sus_scoring():
    with pytest.raises(ValueError):
        ev.select_for_review([{"split": "test", "kind": "uncertain"}], 1)
    picked = ev.select_for_review([{"kind": "disagreement", "speaker_id": s} for s in "aab"], 2, quotas=(1, 0, 0))
    assert sorted(c["speaker_id"] for c in picked) == ["a", "b"]  # round-robin across speakers
    assert ev.sus_score([5, 1] * 5) == 100 and ev.sus_score([3] * 10) == 50


def test_decode_tiles_by_stride_and_abstains_below_threshold():
    labels = list(LABELS)
    probs = np.zeros((5, len(labels)))
    probs[[0, 1], labels.index("filler_word")] = 0.9
    probs[2, labels.index("fluent")] = 1
    probs[[3, 4], labels.index("block")] = 0.4
    events = decode(probs, np.arange(5) * 0.5, labels, thresholds={"block": 0.5})
    assert [(x["label"], x["decision"], x["start_ms"], x["end_ms"]) for x in events] == [
        ("filler_word", "event", 0, 1000),
        ("block", "uncertain", 1500, 2500),
    ]


def test_vectorized_windows_equal_per_window_builder():
    """Refactor is computational only: batch tensors equal the old per-window path."""
    rng = np.random.default_rng(0)
    t = np.arange(16 * SAMPLE_RATE) / SAMPLE_RATE
    audio = (
        0.2 * np.sin(2 * np.pi * 140 * t) * (np.sin(2 * np.pi * 0.4 * t) > 0) + 0.01 * rng.standard_normal(len(t))
    ).astype(np.float32)
    words = [
        Word("eh", "eh", 1.0, 1.3, 0.9, ","),
        Word("bueno", "bueno", 4.0, 4.4, 0.9, ""),
        Word("casa", "casa", 9.0, 9.5, 0.9, "."),
    ]
    starts = [0.0, 2.5, 6.0]
    frames = extract_acoustic(audio, use_silero=False).frame_matrix()
    batch = window_tensors(frames, words, starts)
    for i, s in enumerate(starts):
        single = build_window(audio, words, s, use_silero=False)
        for key, value in single.items():
            np.testing.assert_array_equal(value[0], batch[key][i])
