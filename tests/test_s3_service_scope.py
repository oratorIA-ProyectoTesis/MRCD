"""S3 contract revision 1.2: output scope, three-state decision, experimental text route."""

import copy

import pytest

from mrcd.engine import Event
from mrcd.service.analyzer import AnalysisOptions, MrcdAnalyzer
from mrcd.service.contract import validate_document
from tests.test_service_contract import _doc, _result


def _events():
    return [
        Event("filler_word", 1.0, 2.0, 0.60, 2, modalities=("acoustic",)),
        Event("block", 3.0, 3.5, 0.70, 1, modalities=("acoustic",)),
        Event("rhetorical_pause", 4.0, 4.5, 0.55, 1, modalities=("acoustic",)),
        Event("prolongation", 6.0, 7.0, 0.65, 2, modalities=("acoustic",)),
        Event("neutral_pause", 8.0, 8.5, 0.50, 1, modalities=("acoustic",)),
        Event("revision", 9.0, 9.5, 0.52, 1, modalities=("acoustic",)),
    ]


def test_default_scope_hides_out_of_scope_and_does_not_count_them():
    doc = _doc(_result(events=_events(), duration=60.0))
    assert validate_document(doc) == []
    assert {e["class"] for e in doc["events"]} == {"filler_word", "prolongation"}
    assert all(e["in_scope"] for e in doc["events"])
    s = doc["summary"]
    assert s["n_events"] == 2 and s["counts_by_class"] == {
        "filler_word": 1,
        "prolongation": 1,
    }
    assert s["events_per_minute"]["value"] == 2.0
    assert s["out_of_scope"] == {
        "included": False,
        "n_events": 4,
        "counts_by_class": {
            "block": 1,
            "rhetorical_pause": 1,
            "neutral_pause": 1,
            "revision": 1,
        },
    }
    assert "out_of_scope_hidden" in {w["code"] for w in doc["warnings"]}
    assert doc["request"]["output_scope"] == "evaluated"


def test_explicit_scope_includes_marked_events_but_totals_stay_in_scope():
    doc = _doc(_result(events=_events(), duration=60.0), output_scope="all")
    assert validate_document(doc) == []
    assert len(doc["events"]) == 6
    oos = [e for e in doc["events"] if not e["in_scope"]]
    assert len(oos) == 4 and all(
        e["validation"] == "out_of_scope_unvalidated" and e["experimental"] for e in oos
    )
    assert doc["summary"]["n_events"] == 2
    assert doc["summary"]["events_per_minute"]["value"] == 2.0
    assert doc["summary"]["out_of_scope"]["included"] is True
    assert "class_block_not_measurable" in {w["code"] for w in doc["warnings"]}


def test_only_out_of_scope_events_means_no_events_warning():
    doc = _doc(
        _result(events=[Event("block", 1.0, 2.0, 0.7, 1, modalities=("acoustic",))])
    )
    assert doc["events"] == [] and doc["summary"]["n_events"] == 0
    assert "no_events" in {w["code"] for w in doc["warnings"]}


def test_prolongation_is_described_as_observable_lengthening():
    doc = _doc(_result(events=_events()))
    p = next(e for e in doc["events"] if e["class"] == "prolongation")
    assert (
        "observable lengthening" in p["validation"]
        and "not demonstrated" in p["validation"]
    )


def test_score_warning_says_softmax_not_probability():
    doc = _doc()
    msg = next(
        w["message"] for w in doc["warnings"] if w["code"] == "scores_uncalibrated"
    )
    assert "softmax" in msg and "class probability" not in msg


def test_revision_1_1_style_document_still_validates():
    doc = _doc()
    old = copy.deepcopy(doc)
    for e in old["events"]:
        e.pop("in_scope", None)
    old["summary"].pop("out_of_scope", None)
    for k in ("output_scope", "decision", "text_route"):
        old["request"].pop(k, None)
    assert validate_document(old) == []


def test_decision_is_optional_three_state_without_calibration_claim():
    doc = _doc(_result(events=_events()), decision=True)
    assert validate_document(doc) == []
    for e in doc["events"]:
        d = e["decision"]
        assert (
            d["state"] in ("present", "absent", "uncertain")
            and d["calibration_claim"] == "none"
        )
        assert 0 <= d["estimate"] <= 1 and d["lo"] <= d["hi"]
        assert e["score"]["calibrated"] is False
    assert "decision:three_state" in doc["experimental"]["components"]
    assert all("decision" not in e for e in _doc(_result(events=_events()))["events"])


def test_unknown_scope_and_route_are_rejected():
    from mrcd.service.audio_io import ServiceError

    for bad in (
        AnalysisOptions(output_scope="everything"),
        AnalysisOptions(text_route="nope"),
    ):
        with pytest.raises(ServiceError) as ei:
            MrcdAnalyzer.validate_options(bad)
        assert ei.value.code == "invalid_request"


def test_text_route_events_come_from_words_and_replace_v0_repetition():
    words = [
        {
            "text": t,
            "norm": t,
            "start": 0.2 * i,
            "end": 0.2 * i + 0.15,
            "prob": 0.9,
            "punct_after": "",
        }
        for i, t in enumerate("yo quiero quiero ir a madrid".split())
    ]
    r = _result(
        events=[Event("repetition", 0.0, 1.0, 0.8, 2, modalities=("acoustic",))]
    )
    r.words = words
    doc = _doc(r, text_route="rep_ctx_p1", asr_profile="p1_disfluent_prompt")
    assert validate_document(doc) == []
    reps = [e for e in doc["events"] if e["class"] == "repetition"]
    assert all(
        e["score"]["kind"] == "text_route_logistic_score" and e["experimental"]
        for e in reps
    )
    assert all(e["modalities_used"] == ["linguistic"] for e in reps)
    assert doc["provenance"]["text_route"]["name"] == "rep_ctx_p1"
    assert "text_route:rep_ctx_p1" in doc["experimental"]["components"]
    assert "text_route_asr_mismatch" not in {w["code"] for w in doc["warnings"]}


def test_text_route_without_p1_warns():
    r = _result(events=[])
    r.words = []
    doc = _doc(r, text_route="rep_ctx_p1")
    assert "text_route_asr_mismatch" in {w["code"] for w in doc["warnings"]}
    assert validate_document(doc) == []
