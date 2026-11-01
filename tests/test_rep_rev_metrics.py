"""Metrics, speaker bootstrap, paired difference, Holm, Brier / ECE."""

import numpy as np
import pytest

from mrcd.context_eval import metrics as M


def test_confusion_counts_and_edge_cases():
    c = M.confusion([1, 1, 0, 0, 1], [1, 0, 1, 0, 1])
    assert (c["tp"], c["fp"], c["fn"], c["tn"]) == (2, 1, 1, 1)
    assert c["f1"] == pytest.approx(2 * 2 / (2 * 2 + 1 + 1))
    # no positives at all: recall and F1 are undefined (NaN), never 0
    c0 = M.confusion([0, 0, 0], [0, 0, 0])
    assert np.isnan(c0["recall"]) and np.isnan(c0["f1"]) and np.isnan(c0["precision"])
    # positives exist but nothing predicted: precision undefined, recall 0, F1 0
    c1 = M.confusion([1, 0], [0, 0])
    assert np.isnan(c1["precision"]) and c1["recall"] == 0 and c1["f1"] == 0
    # empty input does not crash
    ce = M.confusion([], [])
    assert ce["n"] == 0 and np.isnan(ce["f1"])
    with pytest.raises(ValueError):
        M.confusion([1, 0], [1])


def test_bootstrap_resamples_speakers_not_units():
    # speaker A: 50 units predicted perfectly; speaker B: 50 units all wrong (FN/FP). A unit-level bootstrap would give
    # a narrow CI around 0.5; a speaker-level one must cover very different values (down to ~0 and up to 1).
    y = np.array([1, 0] * 25 + [1, 0] * 25)
    p = np.concatenate([y[:50], 1 - y[50:]])
    g = np.array(["A"] * 50 + ["B"] * 50)
    lo, hi = M.boot_f1_ci(y, p, g, 2000, 13)
    assert lo < 0.05 and hi > 0.95


def test_bootstrap_with_one_group_or_no_positives_is_nan_not_a_number():
    lo, hi = M.boot_f1_ci([1, 0, 1], [1, 0, 0], ["a", "a", "a"])
    assert np.isnan(lo) and np.isnan(hi)
    lo, hi = M.boot_f1_ci([0, 0, 0, 0], [0, 0, 0, 0], ["a", "a", "b", "b"])
    assert np.isnan(lo) and np.isnan(hi)


def test_bootstrap_is_deterministic():
    y = np.array([1, 0, 1, 0, 1, 0, 1, 1])
    p = np.array([1, 0, 0, 0, 1, 1, 1, 0])
    g = np.array(list("aabbccdd"))
    assert M.boot_f1_ci(y, p, g, 500, 7) == M.boot_f1_ci(y, p, g, 500, 7)


def test_paired_delta_identical_predictions_is_zero_with_p_one():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    p = rng.integers(0, 2, 200)
    g = np.repeat(np.arange(20), 10)
    d = M.paired_f1_delta(y, p, p, g, 1000, 13)
    assert d["delta"] == 0 and d["p_boot"] == 1.0 and d["ci"] == [0.0, 0.0]


def test_paired_delta_detects_a_clear_gain_and_respects_p_floor():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 600)
    good = y.copy()
    bad = rng.integers(0, 2, 600)
    g = np.repeat(np.arange(60), 10)
    d = M.paired_f1_delta(y, good, bad, g, 2000, 13)
    assert d["delta"] > 0.3 and d["ci"][0] > 0
    assert d["p_boot"] >= 1 / (d["n_boot"] + 1) - 1e-12 and d["p_boot"] < 0.01


def test_paired_delta_refuses_different_units_and_handles_one_group():
    with pytest.raises(ValueError):
        M.paired_f1_delta([1, 0], [1, 0], [1], ["a", "b"])
    d = M.paired_f1_delta([1, 0, 1], [1, 0, 1], [0, 0, 1], ["a", "a", "a"])
    assert np.isnan(d["p_boot"]) and np.isnan(d["ci"][0])


def test_holm_known_values_and_nan_handling():
    adj = M.holm([0.01, 0.04, 0.03])
    assert adj == pytest.approx(
        [0.03, 0.06, 0.06]
    )  # (3*.01, max(2*.03, .03), max(1*.04, .06))
    assert M.holm([0.5]) == [0.5]
    adj = M.holm([0.01, float("nan"), 0.02])  # NaN is not counted in m
    assert (
        adj[0] == pytest.approx(0.02)
        and np.isnan(adj[1])
        and adj[2] == pytest.approx(0.02)
    )
    assert all(
        a >= p for a, p in zip(M.holm([0.2, 0.001, 0.04, 0.5]), [0.2, 0.001, 0.04, 0.5])
    )
    assert max(M.holm([0.9, 0.8, 0.7])) <= 1.0


def test_brier_and_ece_known_values():
    assert M.brier([1, 0], [1.0, 0.0]) == 0
    assert M.brier([1, 0], [0.5, 0.5]) == pytest.approx(0.25)
    assert M.ece([0] * 10, [0.9] * 10) == pytest.approx(0.9)
    y = [1] + [0] * 9
    assert M.ece(y, [0.1] * 10) == pytest.approx(0.0)
    assert np.isnan(M.ece([], []))
    rel = M.reliability([1, 0, 1, 0], [0.05, 0.05, 0.95, 0.95], 10)
    assert rel[0][2] == 2 and rel[9][2] == 2 and rel[9][4] == 0.5
    assert M.reliability([1], [1.0], 10)[9][2] == 1  # score 1.0 falls in the last bin


def test_boot_diff_ci_contains_the_point_estimate_for_a_real_difference():
    rng = np.random.default_rng(3)
    y = rng.integers(0, 2, 400)
    a = np.clip(y * 0.8 + rng.normal(0, 0.1, 400), 0, 1)
    b = np.full(400, 0.5)
    g = np.repeat(np.arange(40), 10)
    d = M.boot_diff(M.brier, y, a, b, g, 300, 13)
    assert d["delta"] < 0 and d["ci"][1] < 0
