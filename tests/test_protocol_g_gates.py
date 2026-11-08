"""Reference-gate generator (QG-1..QG-5): synthetic annotation sheets only."""

from __future__ import annotations

import copy
import json

import pytest

from scripts.annotation import agreement as AG
from scripts.protocol_g import build_reference_gates as G
from scripts.protocol_g import common as C
from tests.protocol_g_synth import base_events, sidecar, write_sheet

ITEMS = [f"it{i:03d}" for i in range(60)]


def sheets(tmp_path, p1=None, p2=None, r2=None, r2_items=ITEMS[:45], r2_who="rater2"):
    base = base_events(ITEMS)
    a = write_sheet(tmp_path / "p1.csv", "author", ITEMS, p1 or base)
    b = write_sheet(tmp_path / "p2.csv", "author", ITEMS, p2 or base)
    c = (
        write_sheet(
            tmp_path / "r2.csv",
            r2_who,
            r2_items,
            r2 or {k: v for k, v in base.items() if k in r2_items},
        )
        if r2_who
        else None
    )
    return a, b, c


def test_all_gates_pass_on_identical_sheets(tmp_path):
    a, b, c = sheets(tmp_path)
    g = G.build_gates(a, b, c, require_hashed=False)
    assert g["classes"]["repetition"]["qg1"]["status"] == "pass"
    assert g["classes"]["repetition"]["reference_status"] == "ok"
    assert g["qg2_repetition_state"]["status"] == "pass" and g["rep_state_gate_ok"]
    assert (
        g["markers"]["qg5"]["status"] == "pass"
        and g["markers"]["qg2"]["status"] == "pass"
    )
    assert (
        g["markers"]["qg3"]["status"] == "pass"
        and g["markers"]["pg3_reference_status"] == "ok"
    )
    assert g["classes"]["prolongation"]["reference_status"] == "ok"


def test_qg1_not_assessable_below_five_events_makes_the_class_descriptive(tmp_path):
    a, b, c = sheets(tmp_path)
    g = G.build_gates(a, b, c, require_hashed=False)
    fp = g["classes"]["filled_pause"]
    assert (
        fp["qg1"]["status"] == "not_assessable"
        and fp["reference_status"] == "descriptive"
    )
    assert fp["qg1"]["n_events_pass1"] == 3


def test_qg1_fails_when_kappa_or_dice_is_below_threshold(tmp_path):
    base = base_events(ITEMS)
    p2 = copy.deepcopy(base)
    for i, it in enumerate(ITEMS):  # lexical_filler: disjoint items in the two passes
        base[it] = [e for e in base[it]] + (
            [("lexical_filler", "disfluent", 6.0, 6.4)] if i % 4 == 0 else []
        )
        p2[it] = [e for e in p2[it]] + (
            [("lexical_filler", "disfluent", 6.0, 6.4)] if i % 4 == 1 else []
        )
    a, b, c = sheets(tmp_path, p1=base, p2=p2, r2=None)
    g = G.build_gates(a, b, c, require_hashed=False)
    lf = g["classes"]["lexical_filler"]
    assert lf["qg1"]["status"] == "fail" and lf["reference_status"] == "descriptive"
    assert lf["qg1"]["dice"] == 0.0  # verbatim per_class dice, zero (a measured value)


def test_undefined_intra_kappa_counts_as_a_failed_gate_not_zero(tmp_path):
    base = base_events(ITEMS)
    for (
        it
    ) in ITEMS:  # REP in every item of both passes: constant margin, kappa undefined
        base[it] = [e for e in base[it] if e[0] != "repetition"] + [
            ("repetition", "disfluent", 1.0, 2.0)
        ]
    a, b, c = sheets(tmp_path, p1=base, p2=copy.deepcopy(base), r2=None, r2_who=None)
    g = G.build_gates(a, b, c, require_hashed=False)
    q = g["classes"]["repetition"]["qg1"]
    assert (
        q["status"] == "fail" and q["kappa"] is None and "not_available" in q["reason"]
    )


def test_subset_below_minimum_is_read_verbatim_and_gives_a_conditional_status(tmp_path):
    a, b, c = sheets(tmp_path, r2_items=ITEMS[:20])
    g = G.build_gates(a, b, c, require_hashed=False)
    sub = AG.compute(a, c, mode="subset")
    assert sub["subset_below_minimum"] is True
    assert (
        g["agreement"]["subset"]["pass1"]["subset_below_minimum"]
        is sub["subset_below_minimum"]
    )
    rep = g["classes"]["repetition"]
    assert rep["qg3"]["status"] == "not_assessable"
    assert rep["reference_status"] == "conditional"
    assert any("inter-rater gate not assessable" in n for n in rep["notes"])
    assert (
        g["markers"]["qg3"]["status"] == "not_assessable"
        and g["markers"]["pg3_reference_status"] == "conditional"
    )


def test_no_rater2_file_is_not_assessable_not_failed(tmp_path):
    a, b, _ = sheets(tmp_path, r2_who=None)
    g = G.build_gates(a, b, None, require_hashed=False)
    assert g["rater2_available"] is False
    assert g["classes"]["repetition"]["qg3"]["status"] == "not_assessable"
    assert g["classes"]["repetition"]["reference_status"] == "conditional"


def test_failed_rater2_gate_makes_the_class_descriptive(tmp_path):
    a, b, c = sheets(tmp_path)
    r2 = {
        k: [e for e in v if e[0] != "repetition"]
        for k, v in base_events(ITEMS[:45]).items()
    }
    for i, it in enumerate(ITEMS[:45]):  # rater 2 marks REP on the complementary items
        if i % 3 == 1:
            r2[it].append(("repetition", "disfluent", 1.0, 2.0))
    c = write_sheet(tmp_path / "r2.csv", "rater2", ITEMS[:45], r2)
    g = G.build_gates(a, b, c, require_hashed=False)
    rep = g["classes"]["repetition"]
    assert rep["qg3"]["status"] == "fail" and rep["reference_status"] == "descriptive"


def test_qg5_function_kappa_descriptive_only_is_read_verbatim_and_blocks_pg3(tmp_path):
    base = base_events(ITEMS)
    p2 = copy.deepcopy(base)
    for i, it in enumerate(
        ITEMS
    ):  # pass 2 puts the markers on other items: detection F1 below 0.5
        p2[it] = [e for e in p2[it] if e[0] != "discourse_marker"] + (
            [("discourse_marker", "none", 5.0, 5.3, "este", "disfluent")]
            if 40 <= i < 60
            else []
        )
    a, b, c = sheets(tmp_path, p2=p2)
    g = G.build_gates(a, b, c, require_hashed=False)
    m = AG.compute_markers(a, b, mode="intra")
    assert m["function_kappa_descriptive_only"] is True
    assert (
        g["markers"]["qg5"]["function_kappa_descriptive_only"]
        is m["function_kappa_descriptive_only"]
    )
    assert g["markers"]["qg5"]["status"] == "fail"
    assert g["markers"]["qg2"]["status"] == "not_read"
    assert g["markers"]["pg3_reference_status"] == "not_evaluable"


def test_marker_kappa_below_threshold_fails_qg2(tmp_path):
    base = base_events(ITEMS)
    p2 = copy.deepcopy(base)
    for it in ITEMS[:40]:  # all functions flipped: detection is perfect, kappa = -1
        p2[it] = [
            (
                e[0],
                e[1],
                e[2],
                e[3],
                e[4],
                "functional" if e[5] == "disfluent" else "disfluent",
            )
            if e[0] == "discourse_marker"
            else e
            for e in p2[it]
        ]
    a, b, c = sheets(tmp_path, p2=p2)
    g = G.build_gates(a, b, c, require_hashed=False)
    assert g["markers"]["qg5"]["status"] == "pass"
    assert g["markers"]["qg2"]["status"] == "fail"
    assert g["markers"]["pg3_reference_status"] == "not_evaluable"


def test_fewer_than_30_matched_marker_spans_is_a_failed_qg2(tmp_path):
    base = {
        k: [e for e in v if e[0] != "discourse_marker"]
        for k, v in base_events(ITEMS).items()
    }
    for it in ITEMS[:20]:
        base[it].append(("discourse_marker", "none", 5.0, 5.3, "este", "disfluent"))
    a, b, c = sheets(tmp_path, p1=base, p2=copy.deepcopy(base), r2=None)
    g = G.build_gates(a, b, c, require_hashed=False)
    assert (
        g["markers"]["qg2"]["status"] == "fail" and g["markers"]["qg2"]["kappa"] is None
    )
    assert "fewer than" in g["markers"]["qg2"]["reason"]


def test_unhashed_or_tampered_inputs_are_refused(tmp_path):
    a, b, c = sheets(tmp_path)
    with pytest.raises(C.Refusal) as e:
        G.build_gates(a, b, c, require_hashed=True)
    assert e.value.code == "annotation_unhashed"
    for p in (a, b, c):
        sidecar(p)
    assert G.build_gates(a, b, c)["classes"]["repetition"]["qg1"]["status"] == "pass"
    a.write_text(a.read_text("utf-8") + "\n", "utf-8")
    with pytest.raises(C.Refusal) as e:
        G.build_gates(a, b, c)
    assert e.value.code == "annotation_hash_mismatch"


def test_output_is_deterministic_and_has_no_timestamp(tmp_path):
    a, b, c = sheets(tmp_path)
    g1, g2 = (
        C.dumps(G.build_gates(a, b, c, require_hashed=False)),
        C.dumps(G.build_gates(a, b, c, require_hashed=False)),
    )
    assert g1 == g2
    json.loads(g1)  # valid JSON: no NaN literal
    assert "frozen_utc" not in g1 and "utc" not in json.loads(g1).get("inputs", {})


def test_qg3_rater2_who_hears_few_events_fails_the_gate_instead_of_making_it_not_assessable(tmp_path):
    r2 = {
        k: [e for e in v if e[0] != "repetition"]
        for k, v in base_events(ITEMS[:45]).items()
    }
    for it in ITEMS[:3]:  # rater 2: 3 REP events where each pass has 15 on the subset items
        r2[it].append(("repetition", "disfluent", 1.0, 2.0))
    a, b, _ = sheets(tmp_path, r2_who=None)
    c = write_sheet(tmp_path / "r2.csv", "rater2", ITEMS[:45], r2)
    g = G.build_gates(a, b, c, require_hashed=False)
    q = g["classes"]["repetition"]["qg3"]
    assert q["vs"]["pass1"]["n_events_pass"] == 15 and q["vs"]["pass1"]["n_events_rater2"] == 3
    assert q["status"] == "fail" and g["classes"]["repetition"]["reference_status"] == "descriptive"


def test_qg3_fewer_than_ten_pass_events_on_the_subset_is_not_assessable(tmp_path):
    a, b, c = sheets(tmp_path)
    g = G.build_gates(a, b, c, require_hashed=False)
    pro = g["classes"]["prolongation"]["qg3"]  # 10 PRO events in the pass on the subset items: assessable
    assert pro["status"] == "pass"
    lf = g["classes"]["lexical_filler"]["qg3"]  # none in the pass: not assessable (the class is descriptive via QG-1)
    assert lf["status"] == "not_assessable" and any("in the pass" in r for r in lf["reasons"])


def test_qg3_undefined_subset_kappa_when_assessable_counts_as_failed(tmp_path):
    base = base_events(ITEMS)
    for it in ITEMS:  # REP in every item of every file: both presence vectors constant -> kappa None
        base[it] = [e for e in base[it] if e[0] != "repetition"] + [("repetition", "disfluent", 1.0, 2.0)]
    a, b, c = sheets(tmp_path, p1=base, p2=copy.deepcopy(base), r2={k: base[k] for k in ITEMS[:45]})
    g = G.build_gates(a, b, c, require_hashed=False)
    q = g["classes"]["repetition"]["qg3"]
    assert q["vs"]["pass1"]["kappa"] is None and q["status"] == "fail"


def test_qg3_markers_constant_functions_with_enough_spans_fail_and_few_spans_are_not_assessable(tmp_path):
    base = base_events(ITEMS)
    for it in ITEMS[:40]:  # every marker span disfluent in every file: >= 30 matched spans, kappa None (pe = 1)
        base[it] = [
            (e[0], e[1], e[2], e[3], e[4], "disfluent") if e[0] == "discourse_marker" else e
            for e in base[it]
        ]
    a, b, c = sheets(tmp_path, p1=base, p2=copy.deepcopy(base), r2={k: base[k] for k in ITEMS[:45]})
    m = G.qg3_markers(
        {n: AG.compute_markers(f, c, mode="subset") for n, f in (("pass1", a), ("pass2", b))}
    )
    assert m["vs"]["pass1"]["n_matched_spans"] >= 30 and m["vs"]["pass1"]["kappa"] is None
    assert m["status"] == "fail"
    few = {k: [e for e in v if e[0] != "discourse_marker"] for k, v in base.items()}
    for it in ITEMS[:10]:
        few[it].append(("discourse_marker", "none", 5.0, 5.3, "este", "disfluent"))
    a2, b2, c2 = sheets(tmp_path, p1=few, p2=copy.deepcopy(few), r2={k: few[k] for k in ITEMS[:45]})
    m2 = G.qg3_markers(
        {n: AG.compute_markers(f, c2, mode="subset") for n, f in (("pass1", a2), ("pass2", b2))}
    )
    assert m2["status"] == "not_assessable"
