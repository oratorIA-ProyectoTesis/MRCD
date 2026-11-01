"""Nested speaker-grouped model: nothing learned from the test fold; calibration stays inside the train."""

import numpy as np
import pytest

from mrcd.context_eval.model import fit_lr, nested_oof, platt, select_fit
from mrcd.context_eval.partition import GroupCrossingError, group_folds


def _data(seed=0, n_groups=30, per=20, d=4):
    rng = np.random.default_rng(seed)
    g = np.repeat([f"S{i:02d}" for i in range(n_groups)], per)
    X = rng.normal(size=(len(g), d))
    y = (X[:, 0] + 0.5 * rng.normal(size=len(g)) > 0.7).astype(int)
    return X, y, g


def test_scores_of_a_fold_do_not_depend_on_the_labels_of_that_fold():
    X, y, g = _data()
    fold = group_folds(g, 5, 13)
    base = nested_oof(X, y, g, fold)
    y2 = y.copy()
    te = fold == 2
    y2[te] = 1 - y2[te]  # corrupt ONLY the held-out labels of fold 2
    alt = nested_oof(X, y2, g, fold)
    assert np.allclose(base["score"][te], alt["score"][te]), (
        "test-fold scores changed when test labels changed"
    )
    assert np.array_equal(base["decision"][te], alt["decision"][te])
    assert not np.allclose(
        base["score"][~te], alt["score"][~te]
    )  # other folds do train on those labels


def test_normalisation_is_fitted_on_train_rows_only():
    X, y, g = _data()
    m = fit_lr(X[:200], y[:200], 0.3)
    assert np.allclose(m.mu, X[:200].mean(0)) and not np.allclose(m.mu, X.mean(0))
    shifted = X.copy()
    shifted[200:] += 100.0
    assert np.allclose(
        fit_lr(shifted[:200], y[:200], 0.3).mu, m.mu
    )  # rows outside the fit are irrelevant


def test_every_row_is_predicted_exactly_once_and_threshold_comes_from_train():
    X, y, g = _data()
    fold = group_folds(g, 5, 13)
    out = nested_oof(X, y, g, fold, calibrate=True)
    assert np.isfinite(out["score"]).all() and np.isfinite(out["calibrated"]).all()
    assert ((out["calibrated"] >= 0) & (out["calibrated"] <= 1)).all()
    assert len(out["per_fold"]) == 5 and sum(
        p["n_test"] for p in out["per_fold"]
    ) == len(y)
    for p in out["per_fold"]:
        assert 0 < p["threshold"] < 1 and p["C"] in (0.03, 0.3, 3.0)
    # the learned signal transfers: out-of-fold AUC-like check through the decision rule
    assert out["decision"][y == 1].mean() > out["decision"][y == 0].mean() + 0.3


def test_a_fold_assignment_that_splits_a_speaker_is_rejected():
    X, y, g = _data(n_groups=10, per=10)
    bad = group_folds(g, 5, 13).copy()
    bad[0] = (bad[0] + 1) % 5  # one unit of speaker S00 moves to another fold
    with pytest.raises(GroupCrossingError):
        nested_oof(X, y, g, bad)


def test_single_class_training_returns_a_constant_without_crashing():
    X, y, g = _data(n_groups=12, per=5)
    sel = select_fit(X, np.zeros(len(y), int), g)
    assert np.allclose(sel["model"].score(X), 0.0)
    out = nested_oof(X, np.zeros(len(y), int), g, group_folds(g, 3, 13))
    assert out["decision"].sum() == 0


def test_platt_is_monotone_and_handles_one_class():
    s = np.linspace(0.01, 0.99, 200)
    y = (s > 0.6).astype(int)
    f = platt(s, y)
    out = f(np.array([0.1, 0.5, 0.9]))
    assert out[0] < out[1] < out[2]
    assert np.allclose(platt(s, np.zeros(200, int))(np.array([0.3, 0.7])), 0.0)
