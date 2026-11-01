import numpy as np

from mrcd.av.ablation import (
    FUSION,
    VARIANTS,
    boot_p,
    grouped_folds,
    holm,
    identity_probe,
    oof_predictions,
    subsample_negatives,
    summarize,
    verdict,
)
from mrcd.av.config import ANALYSIS
from mrcd.av.features import V_NAMES, speaker_baseline, visual_features


def _data(n_sp=30, per=14, v_signal=0.0, seed=0):
    rng = np.random.default_rng(seed)
    g, y = [], []
    for s in range(n_sp):
        g += [f"s{s}"] * per
        y += list(rng.random(per) < 0.3)
    g, y = np.array(g), np.array(y).astype(int)
    A = rng.normal(size=(len(y), 4)) + 0.5 * y[:, None]
    T = rng.normal(size=(len(y), 3))
    V = rng.normal(size=(len(y), 3))
    V[:, 0] += v_signal * y
    conf = np.ones(len(y))
    return A, T, V, conf, y, g


def test_folds_are_speaker_disjoint_and_cover_all():
    _, _, _, _, y, g = _data()
    f = grouped_folds(g, y, 5, 1)
    for s in np.unique(g):
        assert len(set(f[g == s])) == 1
    assert set(f) == set(range(5))


def test_negative_subsample_is_deterministic_and_keeps_positives():
    ids = [f"u{i}" for i in range(500)]
    y = np.array([1 if i % 10 == 0 else 0 for i in range(500)])
    k1, k2 = subsample_negatives(ids, y, 3.0), subsample_negatives(ids, y, 3.0)
    assert (
        (k1 == k2).all() and k1[y == 1].all() and (k1 & (y == 0)).sum() == 3 * y.sum()
    )


def test_holm_monotone_and_bounded():
    h = holm({"a": 0.01, "b": 0.04, "c": 0.5})
    assert h["a"] == 0.03 and h["b"] == 0.08 and h["c"] == 0.5
    assert boot_p(np.ones(100)) < 0.05


def test_informative_visual_signal_is_detected_and_noise_is_not():
    cfg = dict(ANALYSIS)
    res = {}
    for name, sig in (("signal", 2.5), ("noise", 0.0)):
        A, T, V, c, y, g = _data(v_signal=sig, seed=3)
        P = oof_predictions(A, T, V, c, y, g, 5, 2, 7)
        res[name] = summarize(P, y, g, 300, 5, cfg["equivalence_margin_f1"])
    assert res["signal"]["paired"]["F1_concat"]["delta_f1"] > 0.03
    assert res["signal"]["paired"]["F1_concat"]["ci95"][0] > 0
    assert res["noise"]["paired"]["F1_concat"]["ci95"][0] < 0.02
    assert set(res["signal"]["paired"]) == set(FUSION)
    for r in res["signal"]["variants"].values():
        assert r["tp"] + r["fn"] > 0
    v = verdict(
        res["noise"],
        res["noise"],
        {**cfg, "min_admitted_positives": 1, "min_positive_speakers": 1},
    )
    assert v["overall"] in ("sin aporte", "no concluyente")


def test_variants_share_the_same_units():
    A, T, V, c, y, g = _data(seed=4)
    P = oof_predictions(A, T, V, c, y, g, 5, 1, 1)
    assert P.shape == (1, len(y), len(VARIANTS)) and np.isfinite(P).all()


def test_speaker_normalisation_reduces_identity_leakage():
    rng = np.random.default_rng(0)
    fps = 25.0
    raw, nrm, grp = [], [], []
    for s in range(8):
        off, sc = rng.normal(0, 3), 1 + rng.random()
        frames = {
            k: sc * rng.normal(size=(2000,)) + off
            for k in ("ap", "jaw", "pucker", "pitch", "yaw", "roll")
        }
        frames["speed"] = np.abs(rng.normal(size=2000)) * sc
        base = speaker_baseline(frames, 250)
        for _ in range(20):
            i = rng.integers(0, 1900)
            sig = {k: v[i : i + 150] for k, v in frames.items() if k != "speed"}
            raw.append(visual_features(sig, fps, None))
            nrm.append(visual_features(sig, fps, base))
            grp.append(f"s{s}")
    ar, ch, k, _ = identity_probe(np.array(raw), grp)
    an, _, _, _ = identity_probe(np.array(nrm), grp)
    assert ar > 0.9 and an < ar - 0.3 and ch == 1 / k


def test_visual_features_unavailable_is_none_not_zero():
    sig = {
        k: np.full(50, np.nan) for k in ("ap", "jaw", "pucker", "pitch", "yaw", "roll")
    }
    assert visual_features(sig, 25.0, None) is None
    sig["ap"][:5] = 0.1
    assert visual_features(sig, 25.0, None) is None
    assert (
        speaker_baseline(
            {k: np.zeros(10) for k in ("ap", "jaw", "pucker", "pitch", "yaw", "roll")},
            250,
        )
        is None
    )
    assert len(V_NAMES) == 12


def test_inadequate_sample_is_never_a_verdict():
    A, T, V, c, y, g = _data(n_sp=12, per=6, v_signal=0.0, seed=5)
    P = oof_predictions(A, T, V, c, y, g, 3, 1, 3)
    r = summarize(P, y, g, 200, 1, 0.5)  # huge margin would otherwise read as "sin aporte"
    v = verdict(r, r, dict(ANALYSIS, min_admitted_positives=10**6))
    assert set(v["per_variant"].values()) == {"no concluyente"} and v["overall"] == "no concluyente"
