"""Discourse-marker function layer: parsing, closed list, agreement, compatibility.
Synthetic data only."""

import csv
import math

import pytest

from scripts.annotation import unblind as ub
from scripts.annotation.agreement import compute, compute_markers
from scripts.annotation.labels_to_csv import convert
from scripts.annotation.markers import (
    ADJACENCY_CUES,
    DECISION_ORDER,
    MARKER_FUNCTIONS,
    MARKER_TOKENS,
    check_marker,
)

OLD_HEAD = ["item_id", "annotator", "class", "state", "start_s", "end_s", "note"]
NEW_HEAD = OLD_HEAD + ["marker_token", "marker_function"]


def write_m(path, who, rows):
    """rows: (item, token, function, start, end); token 'none' = item without events."""
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(NEW_HEAD)
        for r in rows:
            if r[1] == "none":
                w.writerow([r[0], who, "none", "none", "", "", "", "", ""])
            else:
                w.writerow(
                    [r[0], who, "discourse_marker", "none", r[3], r[4], "", r[1], r[2]]
                )
    return path


def spans(n, fn_a, fn_b):
    a = [(f"i{k}", "este", fn_a(k), "1.0", "1.4") for k in range(n)]
    b = [(f"i{k}", "este", fn_b(k), "1.05", "1.45") for k in range(n)]
    return a, b


def setup_labels(tmp_path, labels):
    clips, lab = tmp_path / "clips", tmp_path / "labels"
    clips.mkdir(parents=True)
    lab.mkdir(parents=True)
    for cid, text in labels.items():
        (clips / f"{cid}.wav").write_bytes(b"")
        (lab / f"{cid}.txt").write_bytes(text.encode("utf-8"))
    return clips, lab


def test_closed_list_is_the_declared_one():
    assert MARKER_TOKENS == (
        "este",
        "o_sea",
        "bueno",
        "pues",
        "a_ver",
        "tipo",
    )
    assert MARKER_FUNCTIONS == ("functional", "disfluent", "uncertain")
    assert check_marker(" Este ", "FUNCTIONAL") == ("este", "functional")


def test_label_parsing_marker_and_event_rows(tmp_path):
    txt = "1.0\t1.4\tMK este d\n2.0\t2.3\tMK o_sea f dudoso\n3.0\t3.5\tFP d\n"
    clips, lab = setup_labels(tmp_path, {"c1": txt})
    rows, _ = convert(clips, lab, "me")
    assert rows[0][2:] == [
        "discourse_marker",
        "none",
        "1.000",
        "1.400",
        "",
        "este",
        "disfluent",
    ]
    assert rows[1][6:] == ["dudoso", "o_sea", "functional"]
    assert rows[2][2:4] == ["filled_pause", "disfluent"] and rows[2][7:] == ["", ""]


def test_label_parsing_validates_token_and_function(tmp_path):
    for i, bad in enumerate(("MK vamos d", "MK este x", "MK este", "MK d")):
        clips, lab = setup_labels(tmp_path / str(i), {"c1": f"1.0\t2.0\t{bad}\n"})
        with pytest.raises(ValueError, match="c1"):
            convert(clips, lab, "me")


def test_every_closed_list_token_is_accepted(tmp_path):
    txt = "".join(f"{i}.0\t{i}.5\tMK {t} i\n" for i, t in enumerate(MARKER_TOKENS))
    clips, lab = setup_labels(tmp_path, {"c1": txt})
    rows, _ = convert(clips, lab, "me")
    assert [r[7] for r in rows] == list(MARKER_TOKENS)


def test_agreement_perfect_and_partial(tmp_path):
    f = lambda k: MARKER_FUNCTIONS[k % 3]
    a, b = spans(30, f, f)
    r = compute_markers(
        write_m(tmp_path / "a.csv", "A", a), write_m(tmp_path / "b.csv", "B", b)
    )
    assert r["n_matched_spans"] == 30 and r["kappa"] == 1.0 and r["alpha"] == 1.0
    assert r["not_available_reason"] is None
    assert math.isclose(r["proportion_uncertain"], 1 / 3)
    g = lambda k: "functional" if k % 2 else "disfluent"
    h = lambda k: "functional" if k % 4 else "disfluent"
    a, b = spans(40, g, h)
    r = compute_markers(
        write_m(tmp_path / "a2.csv", "A", a), write_m(tmp_path / "b2.csv", "B", b)
    )
    assert 0 < r["kappa"] < 1


def test_below_minimum_is_not_available_not_zero(tmp_path):
    f = lambda k: "functional" if k % 2 else "disfluent"
    a, b = spans(10, f, f)
    r = compute_markers(
        write_m(tmp_path / "a.csv", "A", a), write_m(tmp_path / "b.csv", "B", b)
    )
    assert r["n_matched_spans"] == 10 and r["kappa"] is None and r["alpha"] is None
    assert "fewer than 30" in r["not_available_reason"]
    assert r["proportion_uncertain"] == 0.0  # a real zero: spans exist, none uncertain


def test_matching_needs_same_token_and_overlap(tmp_path):
    a = [
        ("i1", "este", "functional", "1.0", "1.4"),
        ("i1", "pues", "functional", "3.0", "3.3"),
    ]
    b = [
        ("i1", "bueno", "functional", "1.0", "1.4"),
        ("i1", "pues", "disfluent", "5.0", "5.3"),
    ]
    r = compute_markers(
        write_m(tmp_path / "a.csv", "A", a),
        write_m(tmp_path / "b.csv", "B", b),
        min_spans=1,
    )
    assert (
        r["n_matched_spans"] == 0
        and r["n_unmatched_a"] == 2
        and r["n_unmatched_b"] == 2
    )


def test_modes_and_value_validation(tmp_path):
    a = [("i1", "este", "functional", "1.0", "1.4")]
    p1, p2 = (
        write_m(tmp_path / "p1.csv", "me", a),
        write_m(tmp_path / "p2.csv", "me", a),
    )
    assert (
        compute_markers(p1, p2, mode="intra", min_spans=1)["kappa"] is None
    )  # zero variance
    with pytest.raises(ValueError):
        compute_markers(p1, p2, mode="inter")
    bad = write_m(
        tmp_path / "bad.csv", "x", [("i1", "vamos", "functional", "1.0", "1.4")]
    )
    with pytest.raises(ValueError, match="closed list"):
        compute_markers(bad, p1)
    bad = write_m(tmp_path / "bad2.csv", "x", [("i1", "este", "maybe", "1.0", "1.4")])
    with pytest.raises(ValueError, match="marker_function"):
        compute_markers(bad, p1)


def test_marker_rows_do_not_change_event_statistics(tmp_path):
    ev = ["i1", "filled_pause", "disfluent", "1.0", "1.5"]
    plain_a, plain_b = tmp_path / "pa.csv", tmp_path / "pb.csv"
    for p, who in ((plain_a, "A"), (plain_b, "B")):
        with open(p, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(OLD_HEAD)
            w.writerow([ev[0], who, *ev[1:], ""])
    base = compute(plain_a, plain_b)
    with open(tmp_path / "m.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(NEW_HEAD)
        w.writerow(["i1", "A", "filled_pause", "disfluent", "1.0", "1.5", "", "", ""])
        w.writerow(
            [
                "i1",
                "A",
                "discourse_marker",
                "none",
                "2.0",
                "2.3",
                "",
                "este",
                "uncertain",
            ]
        )
    r = compute(tmp_path / "m.csv", plain_b)
    assert r["n_units"] == base["n_units"] and r["kappa_label"] == base["kappa_label"]


def test_old_sheets_without_marker_columns_still_work(tmp_path):
    for who in ("A", "B"):
        with open(tmp_path / f"{who}.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(OLD_HEAD)
            w.writerow(["i1", who, "filled_pause", "disfluent", "1.0", "1.5", ""])
            w.writerow(["i2", who, "none", "none", "", "", ""])
    assert (
        compute(tmp_path / "A.csv", tmp_path / "B.csv")["kappa_label"] is None
    )  # single label
    r = compute_markers(tmp_path / "A.csv", tmp_path / "B.csv")
    assert r["n_matched_spans"] == 0 and r["kappa"] is None
    assert r["proportion_uncertain"] is None and r["n_items_common"] == 2


def test_unblind_preserves_marker_columns():
    key = [{"item_id": "b1", "orig_item_id": "n001"}]
    row = {
        "item_id": "b1",
        "annotator": "me",
        "class": "discourse_marker",
        "state": "none",
        "start_s": "1",
        "end_s": "2",
        "note": "",
        "marker_token": "este",
        "marker_function": "uncertain",
    }
    out = ub.unblind([row], key)
    assert out[0]["item_id"] == "n001" and out[0]["marker_function"] == "uncertain"


def test_removed_tokens_are_rejected(tmp_path):
    for tok in ("eh", "em_mm", "em", "mm"):
        with pytest.raises(ValueError):
            check_marker(tok, "uncertain")
    clips, lab = setup_labels(tmp_path, {"c1": "1.0\t1.4\tMK eh d\n"})
    with pytest.raises(ValueError, match="c1"):
        convert(clips, lab, "me")


def test_decision_order_constants():
    assert [k for k, _ in DECISION_ORDER] == ["functional", "disfluent", "uncertain"]
    assert set(k for k, _ in DECISION_ORDER) == set(MARKER_FUNCTIONS)
    assert len(ADJACENCY_CUES) == 3


def test_o_sea_and_a_ver_hint(tmp_path):
    for i, (bad, good) in enumerate((("MK o sea d", "o_sea"), ("MK a ver f", "a_ver"))):
        clips, lab = setup_labels(tmp_path / str(i), {"c1": f"1.0\t2.0\t{bad}\n"})
        with pytest.raises(ValueError, match=good) as ei:
            convert(clips, lab, "me")
        assert "underscore" in str(ei.value)
    clips, lab = setup_labels(tmp_path / "ok", {"c1": "1.0\t2.0\tMK o_sea d\n"})
    assert convert(clips, lab, "me")[0][0][7] == "o_sea"


def test_match_rate_and_detection_f1(tmp_path):
    f = lambda k: "functional"
    a, b = spans(10, f, f)
    b = b[:8] + [(f"i{k}", "none", "", "", "") for k in (8, 9)]  # B detects 8 of 10
    r = compute_markers(
        write_m(tmp_path / "a.csv", "A", a), write_m(tmp_path / "b.csv", "B", b)
    )
    assert r["n_matched_spans"] == 8
    assert math.isclose(r["match_rate"], 8 / 10)
    assert math.isclose(r["detection_f1"], 16 / 18)
    assert r["n_token_mismatches"] == 0
    assert r["function_kappa_descriptive_only"] is False


def test_token_mismatch_counted_and_low_f1_flags_descriptive(tmp_path):
    a = [(f"i{k}", "este", "functional", "1.0", "1.4") for k in range(10)]
    b = [(f"i{k}", "pues", "functional", "1.0", "1.4") for k in range(10)]
    r = compute_markers(
        write_m(tmp_path / "a.csv", "A", a), write_m(tmp_path / "b.csv", "B", b)
    )
    assert r["n_matched_spans"] == 0 and r["match_rate"] == 0.0
    assert r["n_matched_any_token"] == 10 and r["n_token_mismatches"] == 10
    assert r["match_rate_any_token"] == 1.0 and r["detection_f1_any_token"] == 1.0
    assert r["function_kappa_descriptive_only"] is True


def test_detection_none_when_no_spans(tmp_path):
    a = [("i0", "none", "", "", "")]
    r = compute_markers(
        write_m(tmp_path / "a.csv", "A", a), write_m(tmp_path / "b.csv", "B", a)
    )
    assert r["match_rate"] is None and r["detection_f1"] is None
    assert r["function_kappa_descriptive_only"] is True


def test_subset_flag_in_markers(tmp_path):
    f = lambda k: "functional"
    a, b = spans(10, f, f)
    r = compute_markers(
        write_m(tmp_path / "a.csv", "A", a),
        write_m(tmp_path / "b.csv", "B", b),
        mode="subset",
    )
    assert r["subset_below_minimum"] is True
    a, b = spans(40, f, f)
    r = compute_markers(
        write_m(tmp_path / "a2.csv", "A", a),
        write_m(tmp_path / "b2.csv", "B", b),
        mode="subset",
    )
    assert r["subset_below_minimum"] is False
    r = compute_markers(
        write_m(tmp_path / "a3.csv", "A", a), write_m(tmp_path / "b3.csv", "B", b)
    )
    assert r["subset_below_minimum"] is False
