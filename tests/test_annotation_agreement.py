import csv
import math

from scripts.annotation.agreement import (
    cohen_kappa,
    compute,
    iou,
    krippendorff_alpha_nominal,
    match_events,
)

HEAD = ["item_id", "annotator", "class", "state", "start_s", "end_s", "note"]


def write(path, who, rows):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(HEAD)
        for r in rows:
            w.writerow([r[0], who, *r[1:], ""])
    return path


def test_kappa_known_value():
    assert cohen_kappa(list("aabb"), list("aabb")) == 1.0
    # 10 units: po = 0.7; marginals y=7/10 vs 6/10 -> pe = 0.7*0.6 + 0.3*0.4 = 0.54
    x = ["y"] * 5 + ["y"] * 2 + ["n"] * 1 + ["n"] * 2
    y = ["y"] * 5 + ["n"] * 2 + ["y"] * 1 + ["n"] * 2
    assert math.isclose(cohen_kappa(x, y), (0.7 - 0.54) / (1 - 0.54), abs_tol=1e-9)


def test_kappa_degenerate_is_none_not_zero():
    assert cohen_kappa(["a", "a"], ["a", "a"]) is None
    assert cohen_kappa([], []) is None


def test_alpha_perfect_and_disagreement():
    assert krippendorff_alpha_nominal(list("abab"), list("abab")) == 1.0
    assert krippendorff_alpha_nominal(list("ab"), list("ba")) < 0.0


def test_iou_and_matching_one_to_one():
    a = {"start": 0.0, "end": 1.0}
    b = {"start": 0.5, "end": 1.5}
    assert math.isclose(iou(a, b), 1 / 3)
    pairs, ua, ub = match_events([a], [b, {"start": 0.0, "end": 0.9}], 0.1)
    assert len(pairs) == 1 and pairs[0][1]["end"] == 0.9 and len(ub) == 1 and not ua


def test_full_agreement(tmp_path):
    rows = [
        ("i1", "filled_pause", "disfluent", "1.0", "1.5"),
        ("i2", "repetition", "functional", "2.0", "2.8"),
        ("i3", "none", "none", "", ""),
    ]
    r = compute(
        write(tmp_path / "a.csv", "A", rows), write(tmp_path / "b.csv", "B", rows)
    )
    assert r["kappa_label"] == 1.0 and r["alpha_label"] == 1.0
    assert r["mean_iou_matched"] == 1.0 and r["boundary_agreement"] == 1.0
    assert r["per_class"]["repetition"]["dice"] == 1.0


def test_state_disagreement_and_boundaries(tmp_path):
    a = [
        ("i1", "repetition", "disfluent", "1.0", "2.0"),
        ("i2", "repetition", "disfluent", "1.0", "2.0"),
        ("i3", "filled_pause", "disfluent", "0.0", "0.5"),
        ("i4", "none", "none", "", ""),
    ]
    b = [
        (
            "i1",
            "repetition",
            "functional",
            "1.1",
            "2.1",
        ),  # state differs, boundaries within tol
        (
            "i2",
            "repetition",
            "disfluent",
            "1.6",
            "2.6",
        ),  # same label, boundaries off by 0.6 s
        ("i3", "filled_pause", "disfluent", "0.0", "0.5"),
        ("i4", "filled_pause", "disfluent", "3.0", "3.4"),  # B-only event
    ]
    r = compute(write(tmp_path / "a.csv", "A", a), write(tmp_path / "b.csv", "B", b))
    assert (
        r["n_matched_pairs"] == 3
        and r["n_unmatched_b"] == 1
        and r["n_unmatched_a"] == 0
    )
    assert r["kappa_class"] < 1.0
    assert r["kappa_label"] < r["kappa_class"] + 1e-9
    assert math.isclose(r["boundary_agreement"], 2 / 3)
    assert r["kappa_item_has_event"] is not None
    assert (
        r["per_class"]["filled_pause"]["n_a"] == 1
        and r["per_class"]["filled_pause"]["n_b"] == 2
    )


def test_missing_items_reported(tmp_path):
    a = [("i1", "none", "none", "", ""), ("i2", "none", "none", "", "")]
    b = [("i1", "none", "none", "", "")]
    r = compute(write(tmp_path / "a.csv", "A", a), write(tmp_path / "b.csv", "B", b))
    assert r["items_only_in_a"] == ["i2"] and r["n_items_common"] == 1
    assert r["kappa_label"] is None  # no events: undefined, not 0


def test_intra_mode_requires_same_annotator(tmp_path):
    rows = [
        ("i1", "filled_pause", "disfluent", "1.0", "1.5"),
        ("i2", "repetition", "functional", "1.0", "1.5"),
    ]
    p1 = write(tmp_path / "p1.csv", "author", rows)
    p2 = write(tmp_path / "p2.csv", "author", rows)
    r = compute(p1, p2, mode="intra")
    assert r["mode"] == "intra" and r["kappa_label"] == 1.0
    other = write(tmp_path / "o.csv", "other", rows)
    import pytest

    with pytest.raises(ValueError):
        compute(p1, other, mode="intra")
    with pytest.raises(ValueError):
        compute(p1, p2, mode="inter")  # same rater cannot be inter-rater


def test_subset_mode_uses_intersection_and_flags_small_n(tmp_path):
    full = [(f"i{k}", "repetition", "disfluent", "1.0", "2.0") for k in range(60)]
    sub = full[:10]
    r = compute(write(tmp_path / "a.csv", "A", full), write(tmp_path / "b.csv", "B", sub), mode="subset")
    assert r["n_items_a"] == 60 and r["n_items_b"] == 10 and r["n_items_common"] == 10
    assert r["subset_below_minimum"] is True
    sub40 = full[:40]
    r = compute(write(tmp_path / "a.csv", "A", full), write(tmp_path / "c.csv", "C", sub40), mode="subset")
    assert r["subset_below_minimum"] is False


def test_boundary_tolerance_is_millisecond_rounded(tmp_path):
    # 0.1 vs 0.35 differ by 0.25 up to float noise: must count as within tolerance
    a = [("i1", "repetition", "disfluent", "0.1", "1.1"), ("i2", "filled_pause", "disfluent", "0", "1")]
    b = [("i1", "repetition", "disfluent", "0.35", "1.35"), ("i2", "filled_pause", "disfluent", "0", "1")]
    r = compute(write(tmp_path / "a.csv", "A", a), write(tmp_path / "b.csv", "B", b), tol=0.25)
    assert r["boundary_agreement"] == 1.0
