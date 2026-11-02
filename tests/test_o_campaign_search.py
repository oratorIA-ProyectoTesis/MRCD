import copy

import numpy as np
import pytest

from mrcd.context_eval.partition import GroupCrossingError
from mrcd.optimize import search as SE
from mrcd.optimize import space as SP
from tests.o_campaign_synth import make_data


def test_outer_and_inner_folds_never_share_speakers():
    d = make_data()
    for f in sorted(set(d.outer.tolist())):
        te, tr = np.flatnonzero(d.outer == f), np.flatnonzero(d.outer != f)
        assert not set(d.g[te]) & set(d.g[tr])
        s = SE.inner_oof(SP.sample_order("filler_word", 1, 1)[0], d, tr)
        assert np.isfinite(s).all() and len(s) == len(tr)


def test_disjoint_guard_raises():
    g = np.array(["a", "a", "b"])
    with pytest.raises(GroupCrossingError):
        SE._assert_disjoint(g, np.array([0]), np.array([1]), "x")


def test_selection_uses_only_outer_train_labels():
    """Flip every label of the outer-test rows: search trials, chosen config, thresholds and calibration must be identical."""
    d = make_data(seed=3)
    f = 2
    a = SE.outer_fold(d, f, budget=5, patience=5)
    d2 = copy.copy(d)
    te = np.flatnonzero(d.outer == f)
    y2 = d.y.copy()
    y2[te] = 1 - y2[te]
    d2.y = y2
    d2.rule = {}
    b = SE.outer_fold(d2, f, budget=5, patience=5)
    assert a["config"] == b["config"] and a["thr"] == b["thr"] and a["t_or"] == b["t_or"]
    assert [t["inner_f1"] for t in a["trials"]] == [t["inner_f1"] for t in b["trials"]]
    assert a["calibration"] == b["calibration"] and a["band"] == b["band"]
    assert a["n_test"] == len(te)


def test_selection_independent_of_outer_test_features():
    d = make_data(seed=4)
    f = 1
    a = SE.outer_fold(d, f, budget=4, patience=4)
    te = np.flatnonzero(d.outer == f)
    recs = d.store.recs
    for i in te:  # destroy test-row inputs (including words that feed the duration normaliser)
        recs[i] = {**recs[i], "probs": recs[i]["probs"] * 0 + 0.125, "words": {"p0": [dict(w, end=w["end"] + 5) for w in recs[i]["words"]["p0"]]}}
    d.store._cache.clear()
    d.store._st.clear()
    b = SE.outer_fold(d, f, budget=4, patience=4)
    assert a["config"] == b["config"] and a["thr"] == b["thr"]
    assert [t["inner_f1"] for t in a["trials"]] == [t["inner_f1"] for t in b["trials"]]


def _fake_eval(values):
    it = iter(values)
    return lambda cfg: (np.zeros(3), {"f1": next(it), "thr": 0.5, "t_or": None})


def test_stop_rule_budget():
    d = make_data()
    tr = np.arange(len(d.y))
    r = SE.search(d, tr, 1, budget=7, patience=100, evaluate=_fake_eval([0.1 * i for i in range(7)]))
    assert r["stop"] == "budget" and r["n_evaluated"] == 7


def test_stop_rule_patience_and_best_kept():
    d = make_data()
    tr = np.arange(len(d.y))
    vals = [0.5, 0.6, 0.6, 0.6005, 0.55, 0.59, 0.1, 0.1]
    r = SE.search(d, tr, 1, budget=8, patience=3, evaluate=_fake_eval(vals))
    assert r["stop"] == "patience" and r["n_evaluated"] == 5  # gains below eps do not reset the counter
    assert r["best"]["inner_f1"] == pytest.approx(0.6005)


def test_nested_pools_every_row_once_and_learning_curve_shapes():
    d = make_data(seed=5)
    out = SE.nested(d, budget=3, patience=3)
    assert (out["decision"] >= 0).all() and np.isfinite(out["score"]).all()
    assert len(out["per_fold"]) == 5
    cfg = SP.Config.from_dict(out["per_fold"][0]["config"])
    lc = SE.learning_curve(d, 0, cfg, fracs=(0.5,))
    assert lc["0.5"]["n_train"] < out["per_fold"][0]["n_train"]
    assert set(lc["0.5"]["te"]) == set(np.flatnonzero(d.outer == 0))


def test_prolongation_and_repetition_run_on_synthetic():
    for cls in ("prolongation", "repetition"):
        d = make_data(cls, seed=6)
        r = SE.outer_fold(d, 0, budget=3, patience=3)
        assert r["config"]["cls"] == cls and r["n_evaluated"] <= 3
