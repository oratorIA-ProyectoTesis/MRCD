"""Core guarantees of the ASR-recovery experiment: partition, subset/weights, metrics edge cases, no test-row influence, fold 5, decode variants."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from mrcd.asr_recovery import stats as S
from mrcd.asr_recovery.common import check_pool_counts, outer_folds
from mrcd.asr_recovery.cvfold import NanFill, nested_oof_fold_aware
from mrcd.asr_recovery.decode import (
    C1_PROMPT,
    COMMON,
    P1_PROMPT,
    VARIANTS,
    decode_kwargs,
    transcribe,
)
from mrcd.asr_recovery.signals import NAMES, SignalStats, signal_features
from mrcd.asr_recovery.subset import build_subset, choose_ratio
from mrcd.constants import WHISPER_INITIAL_PROMPT
from mrcd.context_eval.partition import find_crossings
from mrcd.context_eval.pool import HistoricalFoldError, assert_no_historical

REPO = Path(__file__).resolve().parents[1]


def fake_pool(n_spk=40, per=12, seed=0):
    rng = np.random.default_rng(seed)
    pool = []
    for s in range(n_spk):
        for t in range(per):
            rl = int(rng.random() < 0.12)
            rv = int(rng.random() < 0.10)
            if rng.random() < 0.05:
                rl = None
            pool.append(
                {
                    "unit_id": f"S{s:03d}_D{t // 3}_U{t:02d}",
                    "speaker": f"S{s:03d}",
                    "dialogue": f"S{s:03d}_D{t // 3}",
                    "fold": 1 + s % 4,
                    "text_A": "a b",
                    "text_B": "a b",
                    "ref": {"rep_literal": rl, "revision": rv},
                }
            )
    return pool


# ---------------------------------------------------------------- partition
def test_outer_partition_has_no_speaker_or_dialogue_crossing():
    pool = fake_pool()
    f = outer_folds(pool)
    assert find_crossings([u["speaker"] for u in pool], f) == []
    assert find_crossings([u["dialogue"] for u in pool], f) == []
    assert set(f.tolist()) == {0, 1, 2, 3, 4}


def test_outer_partition_is_the_parent_one():
    from mrcd.context_eval.partition import group_folds

    pool = fake_pool()
    assert (outer_folds(pool) == group_folds([u["speaker"] for u in pool], 5, 13)).all()


# ---------------------------------------------------------------- subset
def test_subset_contains_all_positives_and_no_ambiguous_unit_as_negative():
    pool = fake_pool()
    sub = build_subset(pool, 2)
    ids = {s["unit_id"] for s in sub}
    pos = {
        u["unit_id"]
        for u in pool
        if u["ref"]["rep_literal"] == 1 or u["ref"]["revision"] == 1
    }
    assert pos <= ids
    amb = {
        u["unit_id"]
        for u in pool
        if u["ref"]["rep_literal"] is None and u["ref"]["revision"] == 0
    }
    assert not (amb & ids)
    pos_spk = {u["speaker"] for u in pool if u["unit_id"] in pos}
    assert {s["speaker"] for s in sub} <= pos_spk
    for s in sub:
        if s["kind"] == "neg":
            assert s["rep_literal"] == 0 and s["revision"] == 0


def test_subset_is_deterministic_and_weights_restore_the_negative_counts():
    pool = fake_pool()
    a, b = build_subset(pool, 2), build_subset(pool, 2)
    assert a == b
    for spk in {s["speaker"] for s in a}:
        m = sum(
            1
            for u in pool
            if u["speaker"] == spk
            and u["ref"]["rep_literal"] == 0
            and u["ref"]["revision"] == 0
        )
        assert sum(
            s["w"] for s in a if s["speaker"] == spk and s["kind"] == "neg"
        ) == pytest.approx(m)
    assert all(s["w"] == 1.0 for s in a if s["kind"] == "pos")


def test_subset_ratio_caps_at_available_negatives_and_grows_with_ratio():
    pool = fake_pool()
    n1 = sum(s["kind"] == "neg" for s in build_subset(pool, 1))
    n3 = sum(s["kind"] == "neg" for s in build_subset(pool, 3))
    assert n3 >= n1


def test_choose_ratio_respects_budget():
    secs = {1: 3000.0, 2: 4500.0, 3: 6000.0}
    assert choose_ratio(secs, rtf=0.5) == 3
    assert choose_ratio(secs, rtf=0.77) == 2
    assert choose_ratio(secs, rtf=5.0) == 1  # nothing fits: smallest, never truncated


# ---------------------------------------------------------------- metrics
def test_weighted_confusion_with_unit_weights_equals_plain_counts():
    y = np.array([1, 1, 0, 0, 1, 0])
    p = np.array([1, 0, 1, 0, 1, 0])
    m = S.wconfusion(y, p)
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (2, 1, 1, 2)
    assert m["w_f1"] == pytest.approx(m["f1"])


def test_weighted_confusion_inverse_sampling_changes_precision_not_recall():
    y = np.array([1, 1, 0, 0])
    p = np.array([1, 1, 1, 0])
    m = S.wconfusion(y, p, np.array([1, 1, 10, 10]))
    assert m["w_recall"] == 1.0
    assert m["w_precision"] == pytest.approx(2 / 12)


def test_metrics_degenerate_cases_are_nan_not_zero():
    m = S.wconfusion(np.zeros(5, int), np.zeros(5, int))
    assert (
        np.isnan(m["w_f1"]) and np.isnan(m["w_precision"]) and np.isnan(m["w_recall"])
    )
    d = S.paired_wf1_delta(
        np.array([1, 0]), np.array([1, 0]), np.array([1, 0]), np.array(["a", "a"])
    )
    assert np.isnan(d["p_boot"]) and d["n_groups"] == 1  # one speaker: no bootstrap


def test_paired_delta_identical_predictions_is_zero_and_detects_a_real_gain():
    rng = np.random.default_rng(0)
    g = np.repeat([f"s{i}" for i in range(30)], 10)
    y = (rng.random(300) < 0.3).astype(int)
    good = y.copy()
    bad = np.zeros_like(y)
    assert S.paired_wf1_delta(y, good, good, g, None, 500)["delta"] == 0.0
    d = S.paired_wf1_delta(y, good, bad, g, None, 500)
    assert d["delta"] > 0.9 and d["ci"][0] > 0 and d["p_boot"] < 0.05


def test_paired_delta_requires_identical_units():
    with pytest.raises(ValueError):
        S.paired_wf1_delta(
            np.array([1, 0]), np.array([1]), np.array([1, 0]), np.array(["a", "b"])
        )


def test_rate_ci_and_paired_rate_delta():
    g = np.repeat([f"s{i}" for i in range(20)], 5)
    s1 = np.ones(100)
    s0 = np.zeros(100)
    r = S.rate_ci(s1, g, 200)
    assert r["rate"] == 1.0 and r["ci"] == [1.0, 1.0]
    d = S.paired_rate_delta(s1, s0, g, 200)
    assert d["gained"] == 100 and d["lost"] == 0 and d["delta"] == 1.0


def test_wer_edge_cases():
    assert S.wer("", "") == (0, 0)
    assert S.wer("", "a b") == (2, 0)
    assert S.wer("hola, mi nombre", "Hola mi nombre") == (
        0,
        3,
    )  # case / punctuation ignored
    assert S.wer("la casa", "la la casa") == (1, 2)
    assert np.isnan(S.corpus_wer([("", "x")]))


def test_prompt_echo_detects_four_token_overlap_only():
    assert S.prompt_echo("yo quería, quería preguntar por el", P1_PROMPT)
    assert not S.prompt_echo("yo quería ir", P1_PROMPT)


# ---------------------------------------------------------------- no test-row influence
def _toy(n_spk=30, per=8, seed=1):
    rng = np.random.default_rng(seed)
    g = np.repeat([f"s{i:02d}" for i in range(n_spk)], per)
    X = rng.normal(size=(len(g), 4))
    y = ((X[:, 0] + 0.5 * rng.normal(size=len(g))) > 0.7).astype(int)
    from mrcd.context_eval.partition import group_folds

    return X, y, g, group_folds(g, 5, 13)


def test_test_labels_do_not_change_any_test_score():
    X, y, g, fo = _toy()
    a = nested_oof_fold_aware(lambda tr: X, y, g, fo)
    y2 = y.copy()
    te0 = np.flatnonzero(fo == 0)
    y2[te0] = 1 - y2[te0]  # flip every label of outer fold 0
    b = nested_oof_fold_aware(lambda tr: X, y2, g, fo)
    assert np.allclose(a["score"][te0], b["score"][te0])
    assert a["per_fold"][0]["threshold"] == b["per_fold"][0]["threshold"]


def test_make_X_only_sees_train_indices_and_test_rows_do_not_move_train_statistics():
    X, y, g, fo = _toy()
    seen = []

    def mk(tr):
        seen.append(set(tr.tolist()))
        fill = NanFill().fit(X[tr])
        return fill.transform(X)

    a = nested_oof_fold_aware(mk, y, g, fo)
    for f, tr in zip(sorted(set(fo.tolist())), seen):
        assert tr == set(np.flatnonzero(fo != f).tolist())
    # fold-2 test scores come from a model that never saw fold-2 labels or features
    te2 = np.flatnonzero(fo == 2)
    y3 = y.copy()
    y3[te2] = 1 - y3[te2]
    c = nested_oof_fold_aware(lambda tr: NanFill().fit(X[tr]).transform(X), y3, g, fo)
    assert np.allclose(a["score"][te2], c["score"][te2])


def test_signal_stats_use_only_the_given_train_records():
    mk = lambda d: {
        "words": [{"text": "x", "norm": "xx", "start": 0.0, "end": d, "prob": 0.9}],
        "n_samples": 16000,
    }
    train = [mk(0.2), mk(0.3), mk(0.25)]
    st1 = SignalStats.fit(train)
    st2 = SignalStats.fit(train)
    assert (st1.med, st1.scale) == (st2.med, st2.scale)
    st3 = SignalStats.fit(train + [mk(5.0)])
    assert st3.med != st1.med or st3.scale != st1.scale


def test_signal_features_shape_empty_turn_and_repeat_overlap():
    st = SignalStats(-2.0, 0.5)
    empty = signal_features({"words": [], "n_samples": 16000}, st)
    assert empty.shape == (len(NAMES),) and empty[-1] == 1.0
    w = [
        {"text": "no", "norm": "no", "start": 0.0, "end": 0.9, "prob": 0.9},
        {"text": "no", "norm": "no", "start": 0.85, "end": 1.1, "prob": 0.4},
    ]
    f = dict(zip(NAMES, signal_features({"words": w, "n_samples": 24000}, st)))
    assert (
        f["n_overlap_pairs"] == 1
        and f["n_same_pairs"] == 1
        and f["n_same_close_or_overlap"] == 1
    )
    assert f["min_prob_same_pair"] == pytest.approx(0.4)


# ---------------------------------------------------------------- fold 5
def _with_fold5():
    pool = fake_pool()
    pool[0]["fold"] = 5
    return pool


def test_fold_5_is_refused_by_the_pool_checks():
    with pytest.raises(HistoricalFoldError):
        assert_no_historical(u["fold"] for u in _with_fold5())
    with pytest.raises(HistoricalFoldError):
        check_pool_counts(_with_fold5())


def test_no_module_or_script_of_this_experiment_reads_fold5_artifacts():
    banned = ("predictions_dihana_fold5", "historical_fold5", "fold == 5", "fold==5")
    files = list((REPO / "mrcd" / "asr_recovery").glob("*.py")) + list(
        (REPO / "scripts" / "asr_recovery").glob("*.py")
    )
    assert files
    for p in files:
        text = p.read_text("utf8")
        for b in banned:
            assert b not in text, f"{p.name} mentions {b}"


# ---------------------------------------------------------------- decode variants
class _FakeModel:
    def __init__(self):
        self.kw = None

    def transcribe(self, audio, **kw):
        self.kw = kw
        return iter(()), None


class _FakeASR:
    language = "es"

    def __init__(self):
        self.model = _FakeModel()


def test_p0_uses_exactly_the_current_production_settings():
    from mrcd.extractors.linguistic import VerbatimASR

    prod = _FakeASR()
    VerbatimASR.transcribe(prod, np.zeros(16000, np.float32))
    mine = _FakeASR()
    transcribe(mine, np.zeros(16000, np.float32), "P0")
    assert {k: v for k, v in prod.model.kw.items()} == {
        k: v for k, v in mine.model.kw.items()
    }
    assert VARIANTS["P0"]["initial_prompt"] == WHISPER_INITIAL_PROMPT


def test_p1_changes_only_the_prompt_and_p2_adds_only_the_fallback():
    p0, p1, p2 = (decode_kwargs(v) for v in ("P0", "P1", "P2"))
    diff01 = {k for k in p0 if p0[k] != p1.get(k)}
    assert diff01 == {"initial_prompt"}
    assert p2["initial_prompt"] == p1["initial_prompt"]
    assert p2["temperature"] == (0.0, 0.2, 0.4)
    assert (
        p2["compression_ratio_threshold"],
        p2["log_prob_threshold"],
        p2["no_speech_threshold"],
    ) == (2.4, -1.0, 0.6)
    assert p2["condition_on_previous_text"] is False
    with pytest.raises(KeyError):
        decode_kwargs("P3")  # no open-ended search: only the fixed variants exist
    assert COMMON["beam_size"] == 5 and COMMON["vad_filter"] is False


def test_p1_prompt_is_not_a_dihana_turn_and_is_written_once():
    assert P1_PROMPT.startswith("Eh... yo quería")
    assert "tren" not in P1_PROMPT.lower() and "billete" not in P1_PROMPT.lower()


def test_c1_control_differs_from_p1_only_in_the_prompt_and_has_no_disfluent_style():
    """PROTOCOL_ANNEX_20261031_3: C1 is fluent, comparable in length, same decoding parameters as P1."""
    import hashlib
    import re

    c1, p1 = decode_kwargs("C1"), decode_kwargs("P1")
    assert {k for k in c1 if c1[k] != p1.get(k)} == {"initial_prompt"}
    assert c1["initial_prompt"] == C1_PROMPT != P1_PROMPT
    assert hashlib.sha256(C1_PROMPT.encode("utf8")).hexdigest() == (
        "6714aaf4e61a9e1853cf93d7f0e6364d2e571be4a8a04597b6b6a4e133668dd7"
    )  # the text hashed in the c1_prompt_check.json written before any C1 decode
    words = re.findall(r"\w+", C1_PROMPT.lower())
    assert not any(a == b for a, b in zip(words, words[1:])), "adjacent repeated words"
    assert "..." not in C1_PROMPT
    assert not ({"eh", "este", "mmm", "digo", "perdón", "perdone", "bueno", "sea"} & set(words))
    assert {"horario", "preguntar"} <= set(words)
    assert 0.8 <= len(words) / len(re.findall(r"\w+", P1_PROMPT.lower())) <= 1.25


# ---------------------------------------------------------------- OMP pin of the entry points
def test_pin_refuses_to_run_after_numpy_and_sets_omp_before():
    sd = REPO / "scripts" / "asr_recovery"
    bad = subprocess.run(
        [
            sys.executable,
            "-c",
            f"import numpy, sys; sys.path.insert(0, r'{sd}'); import _pin",
        ],
        capture_output=True,
        text=True,
        cwd=REPO,
    )
    assert bad.returncode != 0 and "numpy was imported before" in (
        bad.stderr + bad.stdout
    )
    good = subprocess.run(
        [
            sys.executable,
            "-c",
            f"import sys, os; sys.path.insert(0, r'{sd}'); import _pin; import numpy; print(os.environ['OMP_NUM_THREADS'])",
        ],
        capture_output=True,
        text=True,
        cwd=REPO,
        env={
            k: v for k, v in __import__("os").environ.items() if k != "OMP_NUM_THREADS"
        },
    )
    assert good.returncode == 0 and good.stdout.strip() == "4"


def test_bootstrap_p_is_smoothed_and_never_zero():
    """REVISION6 mutant N4: p = 2 * min((#(d<=0)+1)/(B+1), (#(d>=0)+1)/(B+1)); a perfect gain gives 2/(B+1)."""
    rng = np.random.default_rng(0)
    g = np.repeat([f"s{i}" for i in range(30)], 10)
    y = (rng.random(300) < 0.3).astype(int)
    d = S.paired_wf1_delta(y, y.copy(), np.zeros_like(y), g, None, 500)
    assert d["n_boot"] == 500 and d["p_boot"] == pytest.approx(2 / 501)
    same = S.paired_wf1_delta(y, y, y, g, None, 500)
    assert same["p_boot"] == 1.0
