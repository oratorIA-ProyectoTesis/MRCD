"""Bootstrap, Holm, LOSO, kappa and verdict rules of protocol G on synthetic counts."""

from __future__ import annotations

import json

import numpy as np
import pytest

from scripts.protocol_g import common as C
from scripts.protocol_g import stats_g as S

H = C.H


def counts(seed=0, g=19):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 6, (g, 2, 3)).astype(float)


def test_seed_and_resamples_are_those_of_protocol_f():
    assert (C.SEED, C.N_BOOT_CI, C.N_BOOT_TEST) == (13, 2000, 5000)


def test_bootstrap_is_deterministic():
    c = counts()
    a = S.test_stat(c, S.f1_diff, null=0.0, predicted="gt", two_sided=True)
    b = S.test_stat(c, S.f1_diff, null=0.0, predicted="gt", two_sided=True)
    assert a == b


def test_matches_protocol_f_paired_delta_when_nothing_is_undefined():
    c = counts(3)
    f = H.paired_delta(c[:, 0, :], c[:, 1, :], H.f1_of)
    mine = S.test_stat(c, S.f1_diff, null=0.0, predicted="gt", two_sided=True)
    assert mine["n_undefined"] == 0
    assert mine["estimate"] == pytest.approx(f["delta"])
    assert mine["ci95"] == pytest.approx(f["ci"])
    assert mine["p"] == pytest.approx(f["p_two"])


def test_boot_ci_parity_with_protocol_f():
    c = counts(5)[:, 0, :]
    assert S.ci_only(c, H.f1_of)["ci95"] == pytest.approx(H.boot_ci(c, H.f1_of))


def test_undefined_resamples_count_against_the_hypothesis():
    # one recording carries all the events: resamples that miss it are undefined for survival rate
    c = np.zeros((10, 2))
    c[0] = [1, 10]
    t = S.test_stat(c, S.rate_of, null=0.8, predicted="lt", two_sided=False)
    assert t["n_undefined"] > 0
    # H1: survival < 0.8 and the estimate is 0.1; every undefined resample is added to the p numerator
    base = (1 + 0) / (t["n_boot"] + 1)
    assert t["p"] >= base + t["n_undefined"] / (t["n_boot"] + 1) - 1e-12


def test_one_sided_p_opposite_counts_undefined_resamples_against_the_opposite_claim():
    c = np.zeros((10, 2))
    c[0] = [9, 10]
    t = S.test_stat(c, S.rate_of, null=0.8, predicted="lt", two_sided=False)
    assert t["n_undefined"] > 0 and t["p_opposite"] is not None
    fin = t["n_boot"] - t["n_undefined"]
    assert t["p_opposite"] >= (t["n_undefined"] + 1) / (t["n_boot"] + 1)  # undefined added to the opposite tail too
    assert fin > 0
    assert S.test_stat(c, S.rate_of, null=0.8, predicted="gt", two_sided=True)["p_opposite"] is None


def test_two_sided_undefined_resamples_are_added_to_both_tails():
    c = np.zeros((10, 2))
    c[0] = [5, 10]
    t = S.test_stat(c, S.rate_of, null=0.5, predicted="gt", two_sided=True)
    assert t["n_undefined"] > 0
    assert t["p"] >= min(1.0, 2 * (t["n_undefined"] + 1) / (t["n_boot"] + 1))


def test_ci_only_lower_bound_decision_puts_undefined_resamples_below_the_floor():
    c = np.zeros((10, 2))
    c[0] = [10, 10]  # survival 1.0 where defined; most resamples miss recording 0 and are undefined
    r = S.ci_only(c, S.rate_of)
    assert r["ci95"][0] == pytest.approx(1.0)  # protocol F interval: defined resamples only
    lb = r["lower_bound_decision"]  # the criterion counts them against; the bound itself is never printed as a number
    assert lb["status"] == "not_available" and "tail" in lb["reason"]
    c2 = counts(5)[:, 0, :]
    assert S.ci_only(c2, H.f1_of)["lower_bound_decision"] == pytest.approx(H.boot_ci(c2, H.f1_of)[0])


def test_undefined_point_estimate_is_not_available_not_zero():
    t = S.test_stat(
        np.zeros((5, 2)), S.rate_of, null=0.8, predicted="lt", two_sided=False
    )
    assert C.is_na(t) and "estimate" in t and t["estimate"] is None
    assert C.clean(float("nan")) is None  # serialised as null, never 0


def test_holm_family_size_is_fixed_and_missing_members_still_count():
    p = [0.01, float("nan"), 0.02]
    assert S.holm_fixed(p)[0] == pytest.approx(
        min(1, 3 * 0.01)
    )  # factor 3 even though one member is not computable
    assert np.isnan(S.holm_fixed(p)[1])
    assert S.holm_fixed([0.01, 0.02, 0.03, 0.5, 0.6, 0.7])[0] == pytest.approx(0.06)


def test_kappa_of_matches_the_kit_cohen_kappa_and_flags_constant_margins():
    from scripts.annotation.agreement import cohen_kappa

    rng = np.random.default_rng(1)
    h, s = rng.integers(0, 2, 60), rng.integers(0, 2, 60)
    tab = np.array(
        [[np.sum((h == a) & (s == b)) for a in (0, 1) for b in (0, 1)]], float
    )
    assert S.kappa_of(tab)[0] == pytest.approx(cohen_kappa(h.tolist(), s.tolist()))
    assert np.isnan(S.kappa_of(np.array([[5.0, 0, 0, 0]]))[0])  # both constant and equal: pe = 1, undefined as in the kit
    assert S.kappa_of(np.array([[5.0, 0, 3, 0]]))[0] == pytest.approx(0.0)  # constant system only: defined, zero

    assert np.isnan(S.kappa_of(np.zeros(4))) and S.pabak_of(
        np.array([2.0, 1, 1, 6])
    ) == pytest.approx(2 * 0.8 - 1)


def test_loso_requires_n_minus_one_speakers():
    # 7 speakers, each one recording; delta is positive for every speaker
    c = np.zeros((7, 2, 3))
    c[:, 0, :] = [4, 1, 1]
    c[:, 1, :] = [1, 3, 3]
    spk = [f"s{i}" for i in range(7)]
    lo = S.loso(c, S.f1_diff, spk, null=0.0, predicted="gt")
    assert lo["n_kept"] == 7 and lo["required"] == 6 and lo["passes"]
    # one recording flips the sign so that dropping any of the OTHER speakers flips the pooled direction
    c2 = c.copy()
    c2[0, 0, :], c2[0, 1, :] = [0, 9, 9], [9, 0, 0]
    c2[1:, 0, :], c2[1:, 1, :] = [2, 2, 2], [2, 2, 2]  # nearly null elsewhere
    lo2 = S.loso(c2, S.f1_diff, spk, null=0.0, predicted="gt")
    assert lo2["n_kept"] < lo2["required"] and not lo2["passes"]


def test_loso_undefined_estimate_does_not_keep_the_direction():
    c = np.zeros((2, 2))
    c[0] = [1, 2]  # dropping speaker 'a' leaves no events -> undefined
    lo = S.loso(c, S.rate_of, ["a", "b"], null=0.8, predicted="lt")
    assert (
        lo["per_speaker"]["a"]["kept"] is False
        and lo["per_speaker"]["a"]["estimate"] != lo["per_speaker"]["a"]["estimate"]
    )


def _t(est, lo, hi, null=0.0, pred="gt", kind="two-sided", p_opp=None):
    return {
        "status": "ok",
        "estimate": est,
        "ci95": [lo, hi],
        "null": null,
        "predicted": pred,
        "p_kind": kind,
        "p_opposite": p_opp,
    }


LO_OK = {"passes": True}
LO_BAD = {"passes": False}


def test_verdict_supported_requires_all_four_criteria():
    assert (
        S.verdict(_t(0.2, 0.1, 0.3), LO_OK, 0.01)["verdict"]
        == "supported_on_these_recordings"
    )
    assert (
        S.verdict(_t(0.2, 0.1, 0.3), LO_OK, 0.06)["verdict"] == "not_demonstrated"
    )  # Holm
    assert (
        S.verdict(_t(0.2, -0.1, 0.3), LO_OK, 0.01)["verdict"] == "not_demonstrated"
    )  # interval
    assert (
        S.verdict(_t(0.2, 0.1, 0.3), LO_BAD, 0.01)["verdict"] == "not_demonstrated"
    )  # LOSO
    assert S.verdict(_t(-0.2, -0.3, -0.1), LO_OK, 0.01)["verdict"] == "contradicted"


def test_verdict_one_sided_threshold_test_and_conditional_annotation():
    v = S.verdict(
        _t(0.5, 0.3, 0.7, null=0.8, pred="lt", kind="one-sided"),
        LO_OK,
        0.01,
        conditional_note="inter-rater gate not assessable: x",
    )
    assert (
        v["verdict"] == "supported_on_these_recordings"
        and v["conditional"]
        and "not assessable" in v["annotation"]
    )
    # one-sided: contradicted needs Bonferroni p_opposite * family size < 0.05 AND the interval wholly on the other side
    hi = dict(null=0.8, pred="lt", kind="one-sided")
    assert (
        S.verdict(_t(0.95, 0.9, 0.99, **hi, p_opp=0.001), LO_OK, 0.9, family_size=3)["verdict"]
        == "contradicted"
    )
    assert (
        S.verdict(_t(0.95, 0.9, 0.99, **hi, p_opp=0.02), LO_OK, 0.9, family_size=3)["verdict"]
        == "not_demonstrated"
    )  # 3 x 0.02 = 0.06
    assert (
        S.verdict(_t(0.95, 0.78, 0.99, **hi, p_opp=0.001), LO_OK, 0.9, family_size=3)["verdict"]
        == "not_demonstrated"
    )  # interval reaches the null


def test_verdict_not_evaluable_passes_the_reason_through():
    v = S.verdict(C.not_available("floor"), LO_OK, float("nan"))
    assert v == {"verdict": "not_evaluable", "reason": "floor"}


def test_not_available_is_never_zero_in_serialised_output():
    na = C.not_available("decode failed")
    out = json.loads(C.dumps({"x": na, "y": float("nan"), "z": np.float64("inf")}))
    assert out["x"]["status"] == "not_available" and "value" not in out["x"]
    assert out["y"] is None and out["z"] is None


def test_kendall_tau_b():
    assert S.kendall_tau_b([1, 2, 3, 4], [1, 2, 3, 4]) == pytest.approx(1.0)
    assert S.kendall_tau_b([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert np.isnan(S.kendall_tau_b([1, 1, 1], [1, 2, 3]))
