"""S3 modules: calibration, quantile ECE, three states, weighted LR, OR fusion, route round-trip."""

import numpy as np
import pytest

from mrcd import calibration as C
from mrcd import fusion_text as F


def _data(n=600, seed=0):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.3).astype(int)
    s = np.clip(0.5 * y + rng.normal(0.25, 0.15, n), 0.01, 0.99)
    g = np.array([f"s{i % 30}" for i in range(n)])
    return s, y, g


@pytest.mark.parametrize("method", ["platt", "isotonic"])
def test_calibrator_roundtrip_and_range(method):
    s, y, _ = _data()
    c = C.fit_calibrator(s, y, method)
    p = c.apply(s)
    assert ((p >= 0) & (p <= 1)).all()
    p2 = C.Calibrator.from_dict(c.to_dict()).apply(s)
    assert np.allclose(p, p2)


def test_crossfit_never_uses_own_group():
    s, y, g = _data()
    out = C.crossfit(s, y, g, "isotonic")
    assert np.isfinite(out).all()


def test_ece_quantile_equal_mass_and_perfect_score():
    y = np.array([0, 1] * 50)
    rel = C.reliability_quantile(y, np.linspace(0, 1, 100), 10)
    assert {r[0] for r in rel} == {10}
    assert C.ece_quantile(y, y.astype(float)) == 0.0
    assert C.brier(y, y.astype(float)) == 0.0


def test_three_state_and_nan_is_uncertain():
    assert C.three_state(0.9, 0.3, 0.7) == C.PRESENT
    assert C.three_state(0.1, 0.3, 0.7) == C.ABSENT
    assert C.three_state(0.5, 0.3, 0.7) == C.UNCERTAIN
    assert C.three_state(float("nan"), 0.3, 0.7) == C.UNCERTAIN


def test_selective_report_counts_and_band():
    s, y, _ = _data()
    p = C.fit_calibrator(s, y, "platt").apply(s)
    b = C.choose_band(p, y, 0.4)
    r = C.selective_report(y, C.states(p, b["lo"], b["hi"]))
    assert r["n_present"] + r["n_absent"] + r["n_uncertain"] == len(y)
    if b["feasible"]:
        assert r["error_accepted"] <= 0.75 * b["baseline_error"] + 1e-9
    else:
        assert b["lo"] == b["hi"] == 0.4


def test_weighted_lr_route_roundtrip(tmp_path):
    texts = [
        "quiero ir a a Madrid",
        "quiero ir a Madrid",
        "el el tren sale",
        "el tren sale hoy",
    ] * 30
    y = np.array([1, 0, 1, 0] * 30)
    g = [f"s{i % 12}" for i in range(len(y))]
    from mrcd.context_eval.textfeat import featurize

    X = featurize(texts, "rep_literal")
    sel = F.select_fit_w(X, y, g, np.ones(len(y)))
    r = F.route_from_fit(sel, "rep_literal", "p1_disfluent_prompt", F.NEVER, "test")
    r2 = F.TextRoute.from_json(r.to_json())
    assert np.allclose(r.score_texts(texts), r2.score_texts(texts))
    assert np.allclose(r.score_texts(texts), sel["model"].score(X))
    assert '"experimental": true' in r.to_json()


def test_or_fusion_never_is_rule_alone_and_weights_matter():
    rule = np.array([1, 0, 0, 1, 0, 0])
    score = np.array([0.1, 0.9, 0.2, 0.3, 0.1, 0.1])
    y = np.array([1, 1, 0, 1, 0, 0])
    t = F.fit_or(rule, score, y)
    assert F.wf1(y, F.apply_or(rule, score, t), np.ones(6)) >= F.wf1(
        y, rule, np.ones(6)
    )
    assert (F.apply_or(rule, score, F.NEVER) == rule).all()
