import numpy as np
import pytest

from mrcd.optimize import evaluate as EV


def _toy(n_spk=60, per=8, seed=0):
    rng = np.random.default_rng(seed)
    g = np.repeat([f"S{i}" for i in range(n_spk)], per)
    y = (rng.random(len(g)) < 0.3).astype(int)
    return y, g, rng


def test_holm_and_family_verdicts():
    y, g, rng = _toy()
    good = np.where(rng.random(len(y)) < 0.9, y, 1 - y)
    bad = np.where(rng.random(len(y)) < 0.6, y, 1 - y)
    rows = EV.compare_family(y, good, {"bad": bad, "same": good}, g, n_boot=500)
    by = {r["comparator"]: r for r in rows}
    assert by["same"]["delta"] == 0 and by["same"]["verdict"] == "not demonstrated"
    assert by["bad"]["delta"] > 0 and by["bad"]["ci"][0] > 0
    assert by["bad"]["p_holm"] >= by["bad"]["p_boot"]


def test_bootstrap_is_resampled_by_speaker_and_reproducible():
    y, g, rng = _toy()
    p = np.where(rng.random(len(y)) < 0.8, y, 1 - y)
    a = EV.describe(y, p, g, n_boot=300)
    b = EV.describe(y, p, g, n_boot=300)
    assert a["f1_ci95"] == b["f1_ci95"] and a["f1_ci95"][0] <= a["w_f1"] <= a["f1_ci95"][1]
    assert a["n_groups"] == 60


def test_weighted_metrics_use_weights():
    y = np.array([1, 1, 0, 0])
    p = np.array([1, 0, 1, 0])
    g = np.array(list("abcd"))
    plain = EV.describe(y, p, g, n_boot=50)["w_f1"]
    heavy = EV.describe(y, p, g, w=np.array([1, 1, 10, 1.0]), n_boot=50)["w_f1"]
    assert heavy < plain


def _r(d, lo, hi, p):
    return {"delta": d, "ci": [lo, hi], "p_boot": p}


def test_promotion_requires_superiority_and_non_inferiority():
    ok = EV.promotion({"filler_word": _r(0.05, 0.02, 0.08, 0.001), "prolongation": _r(0.0, -0.02, 0.02, 0.9),
                       "repetition": _r(0.01, -0.01, 0.03, 0.4)})
    assert ok["promote"] and ok["superior_in"] == ["filler_word"]
    worse = EV.promotion({"filler_word": _r(0.05, 0.02, 0.08, 0.001), "prolongation": _r(-0.05, -0.09, -0.01, 0.01),
                          "repetition": _r(0.0, -0.01, 0.01, 0.9)})
    assert not worse["promote"] and worse["worse_or_unevaluable_in"] == ["prolongation"]
    none = EV.promotion({"filler_word": _r(0.02, 0.001, 0.04, 0.04), "prolongation": _r(0, -0.01, 0.01, 1.0),
                         "repetition": _r(0, -0.01, 0.01, 1.0)})
    assert not none["promote"]  # p=0.04 x3 classes -> Holm 0.12
    nan = float("nan")
    unev = EV.promotion({"filler_word": _r(0.05, 0.02, 0.08, 0.001), "prolongation": _r(nan, nan, nan, nan),
                         "repetition": _r(0.0, -0.01, 0.01, 0.9)})
    assert not unev["promote"] and "prolongation" in unev["worse_or_unevaluable_in"]


def test_coverage_risk_counts_abstention_as_not_accepted():
    y = np.array([1, 0, 1, 0, 1, 0])
    st = np.array(["present", "absent", "uncertain", "uncertain", "absent", "present"])
    g = np.array(list("aabbcc"))
    r = EV.coverage_risk(y, st, g, n_boot=100)
    assert r["coverage"] == pytest.approx(4 / 6)
    assert r["error_accepted"] == pytest.approx(2 / 4)
    assert r["coverage_ci95"][0] <= r["coverage"] <= r["coverage_ci95"][1]
