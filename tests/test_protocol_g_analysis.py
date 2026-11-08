"""Hypotheses, item sets, floors, gates, not_available paths and arm independence on SYNTHETIC inputs."""

from __future__ import annotations

import copy
import json

import pytest

from scripts.protocol_g import analysis as AN
from scripts.protocol_g import common as C
from scripts.protocol_g import refs as R
from scripts.protocol_g import stats_g as S
from tests.protocol_g_synth import fast_cfg, ok_gates, synth_inputs


@pytest.fixture(scope="module")
def base():
    inp = synth_inputs()
    return inp, AN.evaluate(inp, fast_cfg(), descriptive=False)


def member(res, fam, pid):
    return res[fam]["members"][pid]


def test_family_sizes_are_fixed_and_all_members_are_listed(base):
    _inp, res = base
    assert res["primary"]["family_size"] == 3 and set(res["primary"]["members"]) == {
        "PG1",
        "PG2",
        "PG3",
    }
    assert (
        res["secondary"]["family_size"] == 6 and len(res["secondary"]["members"]) == 6
    )
    assert res["cw2_arm"]["family_size"] == 3 and res["ft_arm"]["family_size"] == 3


def test_survival_point_estimate_pg2(base):
    _inp, res = base
    pg2 = member(res, "primary", "PG2")
    assert pg2["status"] == "ok" and pg2["n_events"] == 36
    assert pg2["test"]["estimate"] == pytest.approx(
        24 / 36
    )  # j % 8 == 4 events lose their second copy
    assert pg2["test"]["p_kind"] == "one-sided" and pg2["test"]["null"] == C.SURVIVAL_T
    assert (
        pg2["verdict"] == "supported_on_these_recordings"
        and pg2["loso"]["required"] == 3
    )


def test_primary_result_is_deterministic(base):
    inp, res = base
    again = AN.evaluate(copy.deepcopy(inp), fast_cfg(), descriptive=False)
    assert C.dumps(again) == C.dumps(res)


def test_cw2_arm_does_not_change_the_primary_or_secondary_family(base):
    inp, res = base
    without = AN.evaluate(synth_inputs(with_cw2=False), fast_cfg(), descriptive=False)
    assert C.dumps(without["primary"]) == C.dumps(res["primary"])
    assert C.dumps(without["secondary"]) == C.dumps(res["secondary"])
    assert without["cw2_arm"]["status"] == "not_available"
    off = AN.evaluate(inp, fast_cfg(include_cw2=False), descriptive=False)
    assert C.dumps(off["primary"]) == C.dumps(res["primary"])


def test_ft_arm_removed_unless_od4_met(base):
    _inp, res = base
    assert (
        res["ft_arm"]["status"] == "not_available" and "OD-4" in res["ft_arm"]["reason"]
    )
    inp = synth_inputs(with_ft=True)
    on = AN.evaluate(inp, fast_cfg(include_ft=True), descriptive=False)
    assert (
        on["ft_arm"]["members"]["PGF3"]["status"] == "ok"
        and on["ft_arm"]["members"]["PGF1"]["status"] == "ok"
    )
    off = AN.evaluate(inp, fast_cfg(include_ft=False), descriptive=False)
    assert off["ft_arm"]["status"] == "not_available"
    assert C.dumps(on["primary"]) == C.dumps(
        res["primary"]
    )  # the candidate arm never touches the primary result


def test_pgf3_and_pga_do_not_depend_on_r_t_interpretability():
    def mut(inp):
        for r in list(inp.rt)[:2]:  # two aligner failures: R_T not interpretable
            inp.rt[r] = R.RTRec(False, "synthetic aligner failure")

    inp = synth_inputs(with_ft=True, mutate=mut)
    res = AN.evaluate(inp, fast_cfg(include_ft=True), descriptive=False)
    assert res["reference_R_T"]["interpretable"] is False
    assert member(res, "primary", "PG1")["status"] == "not_available"
    assert res["ft_arm"]["members"]["PGF3"]["status"] == "ok"  # H_disf contrast, no R_T
    assert res["absolute"]["PGA1"]["status"] == "ok"


def test_pg3_constant_system_gives_a_defined_kappa_of_zero_not_supported():
    recs = [f"sp{k}_es_{m}" for k in range(4) for m in (1, 2, 3)]
    spec = [
        (r, "disfluent" if i % 2 else "functional", False) for r in recs for i in range(3)
    ]  # the system never says disfluent: kappa = 0 (defined), not "undefined"
    res = AN.evaluate(_marker_world(spec), fast_cfg(), descriptive=False)
    pg3 = member(res, "primary", "PG3")
    assert pg3["status"] == "ok" and pg3["test"]["estimate"] == pytest.approx(0.0)
    assert pg3["verdict"] == "not_demonstrated"


def test_descriptives_below_twenty_reference_units_are_counts_only():
    def mut(inp):  # keep the filled pauses of two recordings only: fewer than 20 events and items
        keep = {"sp0_es_1", "sp0_es_2"}
        for k, evs in inp.human.events.items():
            inp.human.events[k] = [
                e for e in evs if e["class"] != "filled_pause" or e["recording"] in keep
            ]

    d = AN.evaluate(synth_inputs(mutate=mut), fast_cfg())["descriptive"]
    row = d["system_reference_metrics"]["default|filled_pause|H_disf"]
    assert row["n_reference_positive"] < 20 and "f1" not in row
    assert row["rates"]["status"] == "not_available" and "counts only" in row["rates"]["reason"]
    sv = d["survival"]["p0|filled_pause"]
    assert sv["n_events"] < 20 and "ci95" not in sv and sv["rate"]["status"] == "not_available"
    assert "f1" in d["system_reference_metrics"]["default|repetition|H_disf"]


def test_failed_asr_decode_makes_the_arm_not_available_never_zero():
    def mut(inp):
        inp.failed_asr["p0"] = {"sp0_es_1"}

    res = AN.evaluate(synth_inputs(mutate=mut), fast_cfg(), descriptive=False)
    for fam, pid in (
        ("primary", "PG2"),
        ("secondary", "PGS1"),
        ("secondary", "PGS2"),
        ("cw2_arm", "PGC2"),
    ):
        m = member(res, fam, pid)
        assert (
            m["status"] == "not_available"
            and "p0" in m["reason"]
            and "test" not in m
            and m["verdict"] == "not_evaluable"
        )
    assert "estimate" not in member(res, "primary", "PG2")
    # the Holm family keeps its size
    assert res["primary"]["family_size"] == 3


def test_system_with_more_than_one_failed_recording_is_not_interpretable():
    def mut(inp):
        inp.failed_sys["default"] = {"sp0_es_1", "sp0_es_2"}

    res = AN.evaluate(synth_inputs(mutate=mut), fast_cfg(), descriptive=False)
    assert res["systems"]["default"]["interpretable"] is False
    assert (
        member(res, "secondary", "PGS6")["status"] == "not_available"
        and "not interpretable" in member(res, "secondary", "PGS6")["reason"]
    )
    assert res["absolute"]["PGA1"]["status"] == "not_available"
    assert member(res, "primary", "PG1")["status"] == "ok"  # PG1 does not use `default`


def test_one_failed_recording_is_excluded_for_every_member_of_the_contrast():
    def mut(inp):
        inp.failed_sys["default"] = {"sp0_es_1"}

    inp = synth_inputs(mutate=mut)
    res = AN.evaluate(inp, fast_cfg(), descriptive=False)
    full = AN.evaluate(synth_inputs(), fast_cfg(), descriptive=False)
    n_full = member(full, "secondary", "PGS6")["n_items"]
    assert (
        member(res, "secondary", "PGS6")["n_items"] == n_full - 10
    )  # one recording of 10 items, never scored as all-zero


def test_aligner_gate_excludes_recordings_for_every_r_t_contrast_and_s_nogate_restores_them():
    def mut(inp):
        r = inp.rt["sp1_es_1"]
        r.gate_passed, r.gate = False, {"passed": False, "share": 0.4}

    inp = synth_inputs(mutate=mut)
    res = AN.evaluate(inp, fast_cfg(), descriptive=False)
    assert res["reference_R_T"]["recordings_gate_excluded"] == ["sp1_es_1"]
    full = AN.evaluate(synth_inputs(), fast_cfg(), descriptive=False)
    assert (
        member(res, "primary", "PG1")["n_items"]
        == member(full, "primary", "PG1")["n_items"] - 10
    )
    nog = AN.evaluate(inp, fast_cfg(gate_on=False), descriptive=False)
    assert (
        member(nog, "primary", "PG1")["n_items"]
        == member(full, "primary", "PG1")["n_items"]
    )
    assert (
        member(res, "primary", "PG2")["n_events"]
        == member(full, "primary", "PG2")["n_events"]
    )  # C2 does not use R_T


def test_more_than_one_aligner_failure_makes_every_r_t_contrast_not_interpretable():
    def mut(inp):
        for r in ("sp0_es_1", "sp0_es_2"):
            inp.rt[r] = R.RTRec(False, "AlignmentError: test")

    res = AN.evaluate(synth_inputs(mutate=mut), fast_cfg(), descriptive=False)
    assert res["reference_R_T"]["interpretable"] is False
    for pid in ("PG1",):
        assert "R_T not interpretable" in member(res, "primary", pid)["reason"]
    assert "R_T not interpretable" in member(res, "secondary", "PGS6")["reason"]
    assert member(res, "primary", "PG2")["status"] == "ok"


def test_evaluability_floor_gives_not_evaluable_not_a_number():
    def mut(inp):
        keep = 0
        for evs in inp.human.events.values():
            for e in list(evs):
                if e["class"] == "repetition" and e["state"] == "disfluent":
                    keep += 1
                    if keep > 10:
                        evs.remove(e)

    res = AN.evaluate(synth_inputs(mutate=mut), fast_cfg(), descriptive=False)
    pg2 = member(res, "primary", "PG2")
    assert (
        pg2["status"] == "not_available"
        and "floor" in pg2["reason"]
        and pg2["n_events"] == 10
    )
    assert "test" not in pg2


def test_descriptive_gate_makes_the_class_not_evaluable_and_conditional_is_annotated():
    inp = synth_inputs()
    inp.gates = ok_gates()
    inp.gates["classes"]["repetition"] = {
        "reference_status": "descriptive",
        "notes": ["QG-1 fail"],
    }
    res = AN.evaluate(inp, fast_cfg(), descriptive=False)
    for pid in ("PG1", "PG2"):
        assert "quality gate" in member(res, "primary", pid)["reason"]
    assert member(res, "secondary", "PGS1")["status"] == "ok"  # filled pause unaffected
    inp.gates["classes"]["repetition"] = {
        "reference_status": "conditional",
        "notes": ["inter-rater gate not assessable: no rater 2"],
    }
    res = AN.evaluate(inp, fast_cfg(), descriptive=False)
    pg1 = member(res, "primary", "PG1")
    assert (
        pg1["status"] == "ok"
        and pg1["conditional"] is True
        and "not assessable" in pg1["annotation"]
    )


def test_failed_rep_state_gate_removes_only_the_convention_contrasts():
    inp = synth_inputs()
    inp.gates = ok_gates(rep_state_gate_ok=False)
    res = AN.evaluate(inp, fast_cfg(), descriptive=False)
    assert (
        member(res, "secondary", "PGS4")["status"] == "not_available"
        and member(res, "secondary", "PGS5")["status"] == "not_available"
    )
    assert (
        member(res, "secondary", "PGS6")["status"] == "ok"
        and member(res, "primary", "PG1")["status"] == "ok"
    )


def test_decomposition_is_exact_on_s_dec(base):
    inp, res = base
    ctx = AN.Ctx(inp, fast_cfg())
    sel = ctx.sel(
        "R_T:repetition", "H_lit:repetition", "p0_rule:repetition", "H_disf:repetition"
    )
    c, _g, _s = ctx.counts(
        [
            ("R_T:repetition", "p0_rule:repetition"),
            ("H_lit:repetition", "p0_rule:repetition"),
            ("H_disf:repetition", "p0_rule:repetition"),
        ],
        sel,
    )
    tot = c.sum(0)
    f = [float(S.f1_of(tot[i])) for i in range(3)]
    origin, convention = f[0] - f[1], f[1] - f[2]
    pgs4, pgs5 = (
        member(res, "secondary", "PGS4")["test"]["estimate"],
        member(res, "secondary", "PGS5")["test"]["estimate"],
    )
    assert pgs4 == pytest.approx(convention)
    assert pgs5 == pytest.approx(origin + convention)


def test_item_sets_uncertain_only_items_leave_s_dec_but_not_s_lit():
    def mut(inp):
        # a functional-state-free uncertain-only repetition on two extra items of one recording
        for j in (1, 3):
            iid = f"sp2_es_1-{j}"
            inp.human.events[iid].append(
                dict(
                    **{"class": "repetition"},
                    state="uncertain",
                    start=20.0 * j + 2.0,
                    end=20.0 * j + 3.0,
                    item_id=iid,
                    recording="sp2_es_1",
                    speaker="sp2",
                )
            )

    inp = synth_inputs(mutate=mut)
    lit = R.human_labels(inp.human, inp.items, "repetition", "lit")
    dis = R.human_labels(inp.human, inp.items, "repetition", "disf")
    assert lit["sp2_es_1-1"] == 1 and dis["sp2_es_1-1"] is None
    res = AN.evaluate(inp, fast_cfg(), descriptive=False)
    base = AN.evaluate(synth_inputs(), fast_cfg(), descriptive=False)
    assert (
        member(res, "primary", "PG1")["n_items"]
        == member(base, "primary", "PG1")["n_items"]
    )
    assert (
        member(res, "secondary", "PGS4")["n_items"]
        == member(base, "secondary", "PGS4")["n_items"] - 2
    )
    assert (
        R.human_labels(inp.human, inp.items, "repetition", "unc_pos")["sp2_es_1-1"] == 1
    )  # S-UNC-POS
    assert (
        R.human_labels(inp.human, inp.items, "repetition", "unc_neg")["sp2_es_1-1"] == 0
    )  # S-UNC-NEG


def test_item_with_uncertain_and_functional_events_only_is_not_scorable_under_h_disf():
    items = [R.Item(k, "r", "r.mp3", "0" * 64, 20.0 * i, 20.0 * i + 8, "s") for i, k in enumerate("abcd")]
    ev = lambda k, st: {"class": "repetition", "state": st, "start": 1.0, "end": 2.0, "item_id": k}  # noqa: E731
    h = R.HumanRef(
        {"a": [ev("a", "functional"), ev("a", "uncertain")], "b": [ev("b", "functional")],
         "c": [ev("c", "uncertain"), ev("c", "disfluent")], "d": []},
        [],
        set(),
    )
    lab = R.human_labels(h, items, "repetition", "disf")
    # the uncertain event may be disfluent: the H_disf label of 'a' is undetermined, so 'a' leaves S_dec
    assert lab == {"a": None, "b": 0, "c": 1, "d": 0}
    assert R.human_labels(h, items, "repetition", "lit") == {"a": 1, "b": 1, "c": 1, "d": 0}
    assert R.human_labels(h, items, "repetition", "unc_pos")["a"] == 1
    assert R.human_labels(h, items, "repetition", "unc_neg")["a"] == 0


def test_inaudible_items_are_excluded_from_every_class(tmp_path):
    from tests.protocol_g_synth import write_sheet

    items = [
        R.Item("a", "r", "r.mp3", "0" * 64, 10.0, 18.0, "s"),
        R.Item("b", "r", "r.mp3", "0" * 64, 30.0, 38.0, "s"),
    ]
    write_sheet(
        tmp_path / "h.csv",
        "author",
        ["a", "b"],
        {
            "a": [
                ("inaudible", "none", 0.0, 4.0),
                ("repetition", "disfluent", 1.0, 2.0),
            ],
            "b": [("inaudible", "none", 0.0, 3.9)],
        },
    )
    h = R.load_human(tmp_path / "h.csv", items)
    assert h.inaudible_excluded == {
        "a"
    }  # 4.0 / 8.0 = exactly 50 % -> excluded; 3.9 / 8 is not
    assert (
        R.human_labels(h, items, "repetition", "lit")["a"] is None
        and R.human_labels(h, items, "filled_pause", "disf")["a"] is None
    )
    assert R.human_event_list(h, "repetition", ("disfluent",)) == []
    assert h.events["a"][0]["start"] == pytest.approx(
        11.0
    )  # clip-relative -> recording time


def test_pg3_point_estimate_and_companions(base):
    _inp, res = base
    pg3 = member(res, "primary", "PG3")
    c3 = pg3["c3"]
    assert (
        pg3["status"] == "ok"
        and pg3["test"]["predicted"] == "gt"
        and pg3["test"]["p_kind"] == "one-sided"
    )
    t = c3["companions"]["table"]
    assert sum(t.values()) == c3["n_non_uncertain"] - c3["n_not_transcribed"] == 60
    assert 0.0 < pg3["test"]["estimate"] < 1.0 and "pabak" in c3["companions"]


def _marker_world(spec):
    """spec: list of (recording, human function, system disfluent?) -> inputs whose p1 words/markers follow it."""
    inp = synth_inputs()
    inp.human.markers = []
    for rec in inp.words["p1"]:
        inp.words["p1"][rec] = []
    n = {}
    for rec, fn, sysd in spec:
        k = n[rec] = n.get(rec, 0) + 1
        t0 = 20.0 * (k - 1)
        speaker = rec.split("_")[0]
        inp.human.markers.append(
            dict(
                item_id=f"{rec}-{k - 1}",
                recording=rec,
                speaker=speaker,
                token="este",
                function=fn,
                start=t0 + 5.0,
                end=t0 + 5.3,
            )
        )
        gap, dur = (0.5, 0.3) if sysd else (0.05, 0.2)
        ws = inp.words["p1"][rec]
        ws += [
            dict(
                text="a",
                norm="a",
                start=t0 + 5.0 - gap - 0.1,
                end=t0 + 5.0 - gap,
                punct_after="",
            ),
            dict(
                text="este",
                norm="este",
                start=t0 + 5.0,
                end=t0 + 5.0 + dur,
                punct_after="",
            ),
            dict(
                text="b",
                norm="b",
                start=t0 + 5.0 + dur + gap,
                end=t0 + 5.0 + dur + gap + 0.1,
                punct_after="",
            ),
        ]
    return inp


def test_pg3_undefined_kappa_is_not_evaluable_not_zero():
    recs = [f"sp{k}_es_{m}" for k in range(4) for m in (1, 2, 3)]
    spec = [
        (r, "disfluent", True) for r in recs for _ in range(3)
    ]  # constant margins: every span disfluent on both sides
    res = AN.evaluate(_marker_world(spec), fast_cfg(), descriptive=False)
    pg3 = member(res, "primary", "PG3")
    assert (
        pg3["status"] == "not_available"
        and "undefined" in pg3["reason"]
        and "test" not in pg3
    )
    assert pg3["verdict"] == "not_evaluable"


def test_pg3_more_than_ten_percent_degenerate_resamples_is_not_evaluable():
    recs = [f"sp{k}_es_{m}" for k in range(4) for m in (1, 2, 3)]
    spec = [(recs[0], "disfluent", True)] * 15 + [
        (recs[0], "functional", False)
    ] * 15  # all the information in one recording
    spec += [(r, "disfluent", True) for r in recs[1:]]
    res = AN.evaluate(_marker_world(spec), fast_cfg(n_boot_test=500), descriptive=False)
    pg3 = member(res, "primary", "PG3")
    assert pg3["status"] == "not_available" and "degenerate" in pg3["reason"]


def test_pg3_erasure_floor_reports_the_not_transcribed_count():
    recs = [f"sp{k}_es_{m}" for k in range(4) for m in (1, 2, 3)]
    spec = [
        (r, "disfluent" if i % 2 else "functional", bool(i % 2))
        for r in recs
        for i in range(3)
    ]
    inp = _marker_world(spec)
    for rec in recs[:9]:  # the ASR never wrote 'este' there
        inp.words["p1"][rec] = []
    res = AN.evaluate(inp, fast_cfg(), descriptive=False)
    pg3 = member(res, "primary", "PG3")
    assert pg3["status"] == "not_available" and "transcribed" in pg3["reason"]
    assert pg3["c3"]["n_not_transcribed"] == 27 and pg3["c3"][
        "erasure_share"
    ] == pytest.approx(27 / 36)


def test_pg3_descriptive_when_quality_gates_say_so():
    inp = synth_inputs()
    inp.gates = ok_gates()
    inp.gates["markers"] = {
        "pg3_reference_status": "not_evaluable",
        "notes": ["QG-5 failed"],
        "qg5": {"status": "fail", "function_kappa_descriptive_only": True},
    }
    pg3 = member(AN.evaluate(inp, fast_cfg(), descriptive=False), "primary", "PG3")
    assert pg3["status"] == "not_available" and "QG-5" in pg3["reason"]


def test_absolute_criteria_use_the_lower_bound_and_always_present_floor(base):
    _inp, res = base
    a = res["absolute"]
    assert (
        a["PGA1"]["criterion"].startswith("lower 95% bound >=")
        and a["PGA1"]["floor_value"] == 0.30
    )
    assert a["PGA2"]["floor_value"] == 0.40
    assert (
        a["PGA3"]["criterion"].startswith("lower 95% bound >")
        and a["PGA3"]["floor_value"] == 0.0
    )
    assert all(a[k]["status"] == "ok" for k in a)


def test_prolongation_text_survival_is_not_available_by_construction():
    inp = synth_inputs()
    for iid in list(inp.human.events)[:30]:
        t0 = float(iid.rsplit("-", 1)[1]) * 20.0
        inp.human.events[iid].append(
            dict(
                **{"class": "prolongation"},
                state="disfluent",
                start=t0 + 6.0,
                end=t0 + 6.6,
                item_id=iid,
                recording=iid.rsplit("-", 1)[0],
                speaker="sp0",
            )
        )
    d = AN.evaluate(inp, fast_cfg())["descriptive"]["survival"]["p0|prolongation"]
    assert (
        d["text_survival"]["status"] == "not_available"
        and d["n_events"] == 30
        and "share_with_asr_word_in_window" in d
    )


def test_descriptive_section_reports_ranking_prevalence_and_events_without_item():
    d = AN.evaluate(synth_inputs(), fast_cfg())["descriptive"]
    row = d["system_reference_metrics"]["default|repetition|R_T"]
    assert {
        "tp",
        "fp",
        "fn",
        "precision",
        "recall",
        "f1",
        "prevalence",
        "f1_ci95",
    } <= set(row)
    assert (
        "tau_b" in d["ranking_change"]
        and "R_T_vs_H_lit" in d["ranking_change"]["tau_b"]
    )
    assert set(d["events_without_item"]) >= {"default:repetition", "p0_rule:repetition"}
    assert d["event_level_detection"]["default|repetition"]["tp"] > 0
    # p0 transcribes no marker: below the descriptive floor the 2x2 table is reported, rates are counts-only
    o = d["c3_other_asr"]["p0"]
    assert o["coverage"] == 0.0 and o["n_transcribed"] == 0 and "table" in o
    assert "pabak" not in o and o["rates"]["status"] == "not_available"


def test_sensitivities_are_declared_and_leave_the_verdicts_alone():
    inp = synth_inputs()
    inp.rt_anchor = copy.deepcopy(inp.rt)
    res = AN.evaluate(inp, fast_cfg(), descriptive=False)
    sens = AN.sensitivities(inp, fast_cfg())
    assert {
        "S-MARIA",
        "S-MARIA-ENRIQUE",
        "S-UNC-POS",
        "S-UNC-NEG",
        "S-TOL-0.10",
        "S-TOL-0.50",
        "S-OVERLAP",
        "S-NOGATE",
        "S-ALLSTATES",
        "S-P0ANCHOR",
    } <= set(sens)
    assert "verdict" not in json.dumps(sens)
    assert (
        member(AN.evaluate(inp, fast_cfg(), descriptive=False), "primary", "PG2")[
            "verdict"
        ]
        == member(res, "primary", "PG2")["verdict"]
    )


def test_speaker_exclusion_changes_the_loso_requirement():
    inp = synth_inputs()
    res = AN.evaluate(inp, fast_cfg(exclude_speakers=("sp3",)), descriptive=False)
    pg2 = member(res, "primary", "PG2")
    assert pg2["loso"]["n_speakers"] == 3 and pg2["loso"]["required"] == 2


def test_event_widening_and_midpoint_mapping():
    items = [R.Item("i", "r", "r.mp3", "0" * 64, 10.0, 18.0, "s")]
    ev = {"r": {"repetition": [(17.97, 17.97), (18.5, 18.6), (9.0, 9.5)]}}
    lab, dropped = R.system_item_preds(ev, items, "repetition", set())
    assert (
        lab == {"i": 1} and dropped == 2
    )  # zero-length event widened to 20 ms keeps its midpoint inside; 2 events in no item
    lab, _ = R.system_item_preds(ev, items, "repetition", {"r"})
    assert lab == {"i": None}  # a failed recording is excluded, not all-zero
    lab, _ = R.system_item_preds(
        {"r": {"repetition": [(9.5, 10.5)]}}, items, "repetition", set(), overlap=True
    )
    assert lab == {"i": 1}


def test_fp_tag_map_parity_with_the_repository_module():
    from mrcd.verbatim import FP_TAG_NORM

    assert R.FP_TAG_NORM == FP_TAG_NORM
    words = [
        dict(text="[UH]", start=0, end=1),
        dict(text="[LAUGH]", start=1, end=2),
        dict(text="casa", start=2, end=3),
    ]
    assert [t for t, _, _ in R.asr_tokens(words)] == ["eh", "casa"]


def test_reference_parity_with_the_repository_transcript_reference():
    from mrcd.context_eval import transcript_reference as TR

    assert R.H.NONLEXICAL_FILLER_RE.pattern == TR.NONLEXICAL_FILLER_RE.pattern
    for text in (
        "la casa la casa es grande",
        "eh y luego fuimos",
        "e indigenas y hijos",
        "no no quiero ehm no se",
    ):
        toks, fl = R.transcript_tokens(text)
        keep = [t for t, f in zip(toks, fl) if not f]
        ref = R.H.reference_of(text)
        assert ref["rep_literal"] == int(bool(R.H.repeat_positions(keep))) and ref[
            "n_fillers"
        ] == sum(fl)


def test_rt_from_alignment_events_and_gate():
    from scripts.protocol_g.ctc_align import AlignOutput, TokenTimes

    segs = ["eh la casa la casa es grande"]
    toks, _fl, _seg = R.segment_tokens(segs)
    out = AlignOutput(
        [TokenTimes(t, 0.1 * i, 0.1 * i + 0.08, 0.9, True) for i, t in enumerate(toks)]
    )
    rt = R.rt_from_alignment(segs, out)
    assert rt.available and rt.gate_passed and rt.n_lexical == 6
    assert len(rt.rep_events) == 1 and len(rt.fill_events) == 1
    s, e = rt.rep_events[0]  # the second copy "la casa" = tokens 3 and 4
    assert s == pytest.approx(0.3) and e == pytest.approx(0.48)
    out.tokens[3].conf = out.tokens[4].conf = out.tokens[5].conf = 0.1
    assert not R.rt_from_alignment(segs, out).gate_passed


def test_rt_repetitions_and_filler_context_stay_within_a_segment():
    from scripts.protocol_g.ctc_align import AlignOutput, TokenTimes

    segs = ["vimos la casa", "casa grande", "y luego e", "indigenas"]
    toks, fl, seg = R.segment_tokens(segs)
    assert seg == [0, 0, 0, 1, 1, 2, 2, 2, 3]
    # "casa | casa" crosses a segment boundary: no repetition (protocol F scored one transcript unit at a time)
    out = AlignOutput([TokenTimes(t, i, i + 0.5, 0.9, True) for i, t in enumerate(toks)])
    rt = R.rt_from_alignment(segs, out)
    assert rt.rep_events == []
    # the bare "e" closes its segment: no next token in the segment -> filled pause, even if the next segment says "i..."
    assert fl[7] is True and len(rt.fill_events) == 1
    # the same tokens in one segment: repetition found, "e" before "indigenas" is the conjunction
    one = [" ".join(segs)]
    t1, f1, _ = R.segment_tokens(one)
    rt1 = R.rt_from_alignment(one, AlignOutput([TokenTimes(t, i, i + 0.5, 0.9, True) for i, t in enumerate(t1)]))
    assert len(rt1.rep_events) == 1 and not any(f1)


SYNTH_TS = """<recording id="X_ES_9" story="a story" speaker="X" date="1 jan. 2000" language="ES" transcript="T" segments="" annotation="T" annotation_date="" words="">
<transcript>

<segment id="ES_09_1_0001">
			la casa, # la_ casa.  
</segment>
<segment id="ES_09_1_0002">
			[eh] y &amp; luego  
</segment>

</transcript>
</recording>
"""


def test_transcript_parse_rule_reads_segment_content_only():
    segs = R.transcript_segments(R.read_transcript(SYNTH_TS.encode("utf-8")))
    assert len(segs) == 2
    toks, fl, seg = R.segment_tokens(segs)
    # attribute values (ids, story, speaker, dates) and tag names never become tokens
    assert toks == ["la", "casa", "la", "casa", "eh", "y", "luego"] and seg == [0, 0, 0, 0, 1, 1, 1]
    assert fl == [False, False, False, False, True, False, False]
    # same decoding as protocol F: a UTF-8 BOM is accepted, Latin-1 is the fallback
    assert R.transcript_segments(R.read_transcript(b"\xef\xbb\xbf" + SYNTH_TS.encode())) == segs
    assert R.transcript_segments(R.read_transcript(SYNTH_TS.replace("luego", "lu\u00e9go").encode("latin-1")))


@pytest.mark.parametrize(
    "bad",
    [
        "plain text without markup",
        "<recording><transcript>loose text<segment id='1'>a b</segment></transcript></recording>",
        "<recording><transcript><segment id='1'>a <b>b</b></segment></transcript></recording>",
    ],
)
def test_transcript_with_another_layout_is_a_format_error_not_a_guess(bad):
    with pytest.raises(R.TranscriptFormatError):
        R.transcript_segments(bad)
