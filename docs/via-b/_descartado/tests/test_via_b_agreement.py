import numpy as np
import pytest

from mrcd.annotation_eval import metrics as M
from mrcd.annotation_eval.agreement import (
    RecPair,
    _alpha_binary,
    agreement_report,
    segment_positive,
    segments,
)
from mrcd.annotation_eval.eaf import Ev


def ev(tier, value, s, e, ann="A1"):
    return Ev(tier, ann, value, int(round(s * 1000)), int(round(e * 1000)))


def pair(a, b, dur=60.0, speaker="VBT_S01", classes=("filler_word",)):
    return RecPair("VBT_S01_PRES1", speaker, dur, a, b, classes)


def test_cohen_kappa_known_values():
    r = M.cohen_kappa(20, 5, 10, 15)
    assert r["kappa"] == pytest.approx(0.4)
    assert M.cohen_kappa(45, 3, 2, 0)["kappa"] == pytest.approx(
        -0.0504, abs=1e-3
    )  # high-prevalence paradox
    assert (
        M.cohen_kappa(10, 0, 0, 0)["kappa"] is None
    )  # no variation: undefined, not 1 or 0
    assert M.cohen_kappa(0, 0, 0, 0)["reason"] == "no_units"


def test_krippendorff_alpha_known_values_and_closed_form():
    def units(n11, n10, n01, n00):
        return [[1, 1]] * n11 + [[1, 0]] * n10 + [[0, 1]] * n01 + [[0, 0]] * n00

    assert M.krippendorff_alpha_nominal(units(20, 5, 10, 15))["alpha"] == pytest.approx(
        0.4
    )
    assert M.krippendorff_alpha_nominal(units(45, 3, 2, 0))["alpha"] == pytest.approx(
        -0.0421, abs=1e-3
    )
    for c in [(20, 5, 10, 15), (45, 3, 2, 0), (7, 9, 1, 30)]:
        assert _alpha_binary(np.array(c, float)) == pytest.approx(
            M.krippendorff_alpha_nominal(units(*c))["alpha"]
        )
    assert M.krippendorff_alpha_nominal([[1, 1], [1, 1]])["alpha"] is None
    # a missing rating does not create a pairable unit
    assert (
        M.krippendorff_alpha_nominal([[1, None], [0, 0], [1, 1], [0, 1]])["alpha"]
        is not None
    )


def test_iou_and_one_to_one_matching():
    assert M.iou((0, 1), (0.5, 1.5)) == pytest.approx(1 / 3)
    assert M.iou((0, 1), (2, 3)) == 0.0
    # two hypotheses overlap the same reference: only one may match
    m = M.match_events([(0, 1)], [(0, 1), (0.1, 1.0)], 0.5)
    assert len(m) == 1 and m[0][1] == 0
    assert M.match_events([(0, 1)], [(0.6, 1.6)], 0.5) == []
    assert len(M.match_events([(0, 1)], [(0.6, 1.6)], 0.5, any_overlap=True)) == 1


def test_holm_known_and_none_preserved():
    assert M.holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    out = M.holm([0.5, None, 0.01])
    assert (
        out[1] is None
        and out[2] == pytest.approx(0.02)
        and out[0] == pytest.approx(0.5)
    )


def test_segment_grid_and_presence_rule():
    grid = segments(10.0)
    assert grid == [
        (0.0, 3.0),
        (3.0, 6.0),
        (6.0, 9.0),
    ]  # 1 s tail is shorter than 1.5 s: dropped
    assert segments(10.6)[-1] == (9.0, 10.6)
    assert (
        segment_positive((0.0, 3.0), [(2.9, 3.4)]) is False
    )  # 0.1 s of a 0.5 s event: below min(0.30, 0.25)
    assert segment_positive((0.0, 3.0), [(2.7, 3.4)]) is True


def test_perfect_agreement_gives_one():
    evs = [ev("filler_word", "filler_word", 1 + 5 * i, 1.3 + 5 * i) for i in range(10)]
    b = [Ev(e.tier, "A2", e.value, e.start_ms, e.end_ms) for e in evs]
    rep = agreement_report([pair(evs, b)], ["filler_word"])
    r = rep["classes"]["filler_word"]
    assert r["segment"]["cohen_kappa"]["point"] == pytest.approx(1.0)
    assert r["event"]["f1_between_annotators"]["point"] == pytest.approx(1.0)
    assert r["event"]["mean_iou_matched"]["point"] == pytest.approx(1.0)
    assert r["passes_gate"] is True and r["gate"]["reached_reference_0_70"] is True


def test_boundary_shift_breaks_event_match_but_not_segment_presence():
    a = [ev("filler_word", "filler_word", 1 + 5 * i, 1.3 + 5 * i) for i in range(10)]
    b = [
        ev("filler_word", "filler_word", 1.25 + 5 * i, 1.55 + 5 * i, "A2")
        for i in range(10)
    ]  # IoU 0.09
    r = agreement_report([pair(a, b)], ["filler_word"])["classes"]["filler_word"]
    assert r["event"]["matched"] == 0
    assert r["event"]["f1_between_annotators"]["point"] == 0.0
    assert r["segment"]["cohen_kappa"]["point"] == pytest.approx(1.0)
    assert r["passes_gate"] is False and "event_f1" in r["gate"]["failed"]


def test_uncertain_and_excluded_regions_are_ignored_not_disagreements():
    a = [
        ev("filler_word", "filler_word", 10, 10.3),
        ev("filler_word", "filler_word", 20, 20.3),
    ]
    b = [
        ev("filler_word", "filler_word_incierto", 10, 10.3, "A2"),
        ev("filler_word", "filler_word", 20, 20.3, "A2"),
    ]
    r = agreement_report([pair(a, b)], ["filler_word"])["classes"]["filler_word"]
    assert (
        r["event"]["matched"] == 1
        and r["event"]["n_annotator_a"] == 1
        and r["event"]["n_annotator_b"] == 1
    )
    assert r["uncertain"] == {"a": 0, "b": 1}
    # an event inside a region excluded by the other annotator disappears from both the counts and the frames
    a2 = a + [ev("filler_word", "filler_word", 40, 40.3)]
    b2 = b + [ev("excluir", "claqueta", 39, 41, "A2")]
    r2 = agreement_report([pair(a2, b2)], ["filler_word"])["classes"]["filler_word"]
    assert r2["event"]["n_annotator_a"] == 1


def test_gate_not_passed_for_random_annotators_and_insufficient_data_is_none():
    rng = np.random.default_rng(0)
    times = np.sort(rng.uniform(0, 3000, 400))
    a = [ev("filler_word", "filler_word", t, t + 0.3) for t in times[:200]]
    b = [ev("filler_word", "filler_word", t, t + 0.3, "A2") for t in times[200:]]
    r = agreement_report([pair(a, b, dur=3000.0)], ["filler_word"])["classes"][
        "filler_word"
    ]
    assert r["passes_gate"] is False
    few = agreement_report([pair(a[:3], b[:3], dur=3000.0)], ["filler_word"])[
        "classes"
    ]["filler_word"]
    assert few["passes_gate"] is None  # fewer than 10 events: no verdict, never a pass


def test_class_not_declared_complete_by_both_is_not_scored():
    rep = agreement_report([pair([], [], classes=())], ["filler_word"])
    assert rep["classes"]["filler_word"]["status"] == "not_annotated_by_both"


def test_speaker_bootstrap_brackets_point_and_is_deterministic():
    rng = np.random.default_rng(1)
    counts = {
        f"s{i}": np.array(
            [rng.integers(5, 15), rng.integers(0, 4), rng.integers(0, 4)], float
        )
        for i in range(12)
    }
    f1 = lambda v: 2 * v[0] / (2 * v[0] + v[1] + v[2])  # noqa: E731
    r1 = M.speaker_bootstrap(counts, f1, B=500, seed=3)
    r2 = M.speaker_bootstrap(counts, f1, B=500, seed=3)
    assert r1 == r2 and r1["lo"] <= r1["point"] <= r1["hi"] and r1["n_speakers"] == 12


def test_paired_bootstrap_detects_clear_difference_and_requires_same_speakers():
    a = {f"s{i}": np.array([9.0, 1, 1]) for i in range(10)}
    b = {f"s{i}": np.array([5.0, 3, 5]) for i in range(10)}
    f1 = lambda v: 2 * v[0] / (2 * v[0] + v[1] + v[2])  # noqa: E731
    r = M.paired_speaker_bootstrap(a, b, f1, B=500, seed=1)
    assert r["diff"] > 0.2 and r["lo"] > 0 and r["p"] < 0.01
    with pytest.raises(ValueError):
        M.paired_speaker_bootstrap(a, {"x": np.array([1.0, 1, 1])}, f1)
