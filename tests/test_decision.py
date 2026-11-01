import numpy as np
import pytest

from mrcd import decision as D

LABELS = ["fluent", "filler_word", "prolongation"]


def test_v0_rule_and_scores():
    p = np.array([[0.6, 0.3, 0.1], [0.2, 0.5, 0.3]])
    assert D.any_window_argmax(p, LABELS, "filler_word") == 1
    assert D.any_window_argmax(p, LABELS, "prolongation") == 0
    assert D.unit_score(p, LABELS, "filler_word", "max") == 0.5
    assert D.unit_score(p, LABELS, "filler_word", "mean") == pytest.approx(0.4)
    assert D.unit_score(p, LABELS, "filler_word", "top2") == pytest.approx(0.4)
    assert np.isnan(D.unit_score(np.zeros((0, 3)), LABELS, "filler_word"))
    with pytest.raises(ValueError):
        D.unit_score(p, LABELS, "filler_word", "median")


def test_threshold_separates_perfect_scores():
    s = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    y = np.array([0, 0, 0, 1, 1, 1])
    t, f = D.best_threshold(s, y)
    assert f == 1.0 and 0.3 < t <= 0.7


def test_group_folds_never_split_a_speaker():
    groups = np.array([f"s{i % 17}" for i in range(300)])
    folds = D.group_folds(groups, 5)
    for g in set(groups.tolist()):
        assert len(set(folds[groups == g])) == 1
    assert len(set(folds.tolist())) == 5
    assert np.array_equal(folds, D.group_folds(groups, 5))


def test_cv_threshold_oof_is_honest_on_noise():
    rng = np.random.default_rng(0)
    n = 600
    groups = np.array([f"s{i % 30}" for i in range(n)])
    y = rng.integers(0, 2, n)
    s = rng.random(n)  # no signal
    r = D.cv_threshold(s, y, groups)
    assert (
        r["dev_f1_in_sample"] >= r["oof_f1"] - 0.05
    )  # in-sample is not worse by construction
    assert len(r["oof_pred"]) == n and len(r["fold_thresholds"]) == 5


def test_three_state_coherence():
    assert D.three_state(0.9, 0.5) == D.PRESENT
    assert D.three_state(0.1, 0.5) == D.ABSENT
    assert D.three_state(0.5, 0.5, margin=0.1) == D.UNCERTAIN
    assert D.three_state(float("nan"), 0.5) == D.UNCERTAIN
    # monotone: raising the score never moves present -> absent
    order = {D.ABSENT: 0, D.UNCERTAIN: 1, D.PRESENT: 2}
    st = [order[D.three_state(x, 0.5, 0.1)] for x in np.linspace(0, 1, 50)]
    assert st == sorted(st)


def test_rule_json_roundtrip():
    r = D.ThresholdRule({"filler_word": "max"}, {"filler_word": 0.42})
    r2 = D.ThresholdRule.from_json(r.to_json())
    assert r2.agg == r.agg and r2.threshold == r.threshold
    p = np.array([[0.1, 0.5, 0.4]])
    assert r2.predict(p, LABELS, "filler_word") == D.PRESENT
