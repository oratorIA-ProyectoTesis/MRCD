"""Contract v1 of the MRCD service: schema file, validator, no-zero-for-absence, experimental marking."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from mrcd.engine import AnalysisResult, Event
from mrcd.service import (
    SCHEMA,
    AnalysisOptions,
    build_analysis,
    error_document,
    validate_document,
)
from mrcd.service.validate import validate

REPO = Path(__file__).resolve().parents[1]
SCHEMA_FILE = REPO / "mrcd" / "service" / "schemas" / "mrcd.analysis.v1.schema.json"
FIXTURES = REPO / "results" / "cierre_oe1_oe3_20261031" / "integracion" / "fixtures"


def _result(
    events=None,
    duration=30.0,
    warning="checkpoint notice",
    asr_experimental=False,
    head=None,
):
    events = (
        [
            Event(
                "filler_word",
                1.0,
                2.5,
                0.91,
                3,
                text="eh",
                modalities=("acoustic", "linguistic"),
            )
        ]
        if events is None
        else events
    )
    return AnalysisResult(
        events=events,
        transcript="eh bueno",
        duration=duration,
        mode="audio_text",
        warning=warning,
        metadata={
            "window_plan": {
                "window_s": 3.0,
                "stride_s": 0.5,
                "terminal_window": False,
                "span": None,
                "asr_chunk_s": 8.0,
            },
            "asr_profile": {
                "name": "p1_disfluent_prompt" if asr_experimental else "p0",
                "experimental": asr_experimental,
                "validation": None,
                "model": {
                    "model_size": "small",
                    "model_dir": r"C:\Users\someone\.cache",
                    "model_bin_sha256": "a" * 64,
                },
                "anomalies": [],
            },
            "environment": {"omp_num_threads": "4"},
        },
        head=head,
    )


def _doc(result=None, **opts):
    return build_analysis(
        _result() if result is None else result,
        options=AnalysisOptions(**opts),
        audio=None,
        engine_info={
            "checkpoint": {
                "file": "c.pt",
                "sha256": "b" * 64,
                "mode": "audio_text",
                "labels": ["fluent"],
            }
        },
    )


def test_schema_file_is_generated_from_the_python_source():
    assert json.loads(SCHEMA_FILE.read_text(encoding="utf-8")) == SCHEMA


def test_built_document_is_valid_and_has_the_required_fields():
    doc = _doc()
    assert validate_document(doc) == []
    ev = doc["events"][0]
    assert ev["class"] == "filler_word" and (ev["start_s"], ev["end_s"]) == (1.0, 2.5)
    assert ev["score"] == {
        "value": 0.91,
        "kind": "max_window_softmax_score",
        "calibrated": False,
        "label": "uncalibrated",
    }
    assert ev["interval_semantics"] == "decision_stride_not_human_boundary"
    assert doc["provenance"]["checkpoint"]["sha256"] == "b" * 64
    assert doc["provenance"]["asr"]["profile"] == "p0"
    codes = {w["code"] for w in doc["warnings"]}
    assert {"scores_uncalibrated", "no_human_validation", "engine_provenance"} <= codes


def test_no_absolute_local_paths_or_user_names_leak_into_the_document():
    text = json.dumps(_doc())
    assert (
        "someone" not in text and "C:\\\\Users" not in text and "model_dir" not in text
    )


def test_absence_is_never_encoded_as_zero():
    doc = _doc(_result(events=[], duration=0.5))
    assert validate_document(doc) == []
    epm = doc["summary"]["events_per_minute"]
    assert epm == {
        "available": False,
        "value": None,
        "unavailable_reason": epm["unavailable_reason"],
    }
    assert epm["unavailable_reason"]
    video = doc["modalities"]["kinesic_video"]
    assert video["available"] is False and video["reason"]
    assert "no_events" in {
        w["code"] for w in doc["warnings"]
    }  # zero events is not "fluent"


def test_zero_value_with_available_false_is_rejected_by_the_schema():
    doc = _doc()
    doc["summary"]["events_per_minute"] = {
        "available": False,
        "value": 0,
        "unavailable_reason": "x",
    }
    assert validate_document(doc)
    doc["summary"]["events_per_minute"] = {
        "available": True,
        "value": None,
        "unavailable_reason": None,
    }
    assert validate_document(doc)
    doc["summary"]["events_per_minute"] = {
        "available": False,
        "value": None,
        "unavailable_reason": "",
    }
    assert validate_document(doc)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: d["events"][0]["score"].update(calibrated=True),
        lambda d: d["events"][0]["score"].update(label="calibrated"),
        lambda d: d["events"][0].update(score=0.9),
        lambda d: d["events"][0].update(confidence_percent=91),  # unknown field
        lambda d: d["events"][0]["score"].update(value=float("nan")),
        lambda d: d["events"][0]["score"].update(value=1.5),
        lambda d: d.update(schema_version="mrcd.analysis/v2"),
        lambda d: d.pop("warnings"),
        lambda d: d["modalities"].update(
            kinesic_video={"available": True, "reason": None, "emotion": "anxious"}
        ),
    ],
)
def test_validator_rejects_contract_violations(mutate):
    doc = copy.deepcopy(_doc())
    mutate(doc)
    assert validate_document(doc)


def test_contract_has_no_uncertain_state_and_no_psychological_fields():
    text = SCHEMA_FILE.read_text(encoding="utf-8").lower()
    # rev 1.2: "uncertain" exists ONLY as a value of the optional per-event decision (S3, opt-in)
    assert text.count("uncertain") == 1
    for forbidden in (
        "incierto",
        "anxiety",
        "ansiedad",
        "emotion",
        "confidence_level",
    ):
        assert forbidden not in text
    # the engine emits no abstention: every event class is one of the engine labels
    assert all(e["class"] == "filler_word" for e in _doc()["events"])


def test_experimental_components_are_marked_everywhere():
    plain = _doc()
    assert plain["experimental"] == {"active": False, "components": []}
    assert not any(e["experimental"] for e in plain["events"])

    p1 = _doc(_result(asr_experimental=True), asr_profile="p1_disfluent_prompt")
    assert p1["experimental"] == {
        "active": True,
        "components": ["asr_profile:p1_disfluent_prompt"],
    }
    assert "experimental_components" in {w["code"] for w in p1["warnings"]}

    head = {
        "experimental": True,
        "provenance": {"head": "vc", "sha256": "c" * 64},
        "scores_calibrated": False,
    }
    vc = _doc(_result(head=head), head="vc")
    assert vc["experimental"]["components"] == ["head:vc"]
    assert vc["events"][0]["experimental"] is True
    assert vc["head_decision"]["scores_calibrated"] is False
    assert vc["provenance"]["head"]["sha256"] == "c" * 64
    assert validate_document(p1) == [] and validate_document(vc) == []


def test_text_is_excluded_unless_requested():
    off, on = _doc(), _doc(include_text=True)
    assert off["transcript"] is None and off["events"][0]["text"] is None
    assert on["transcript"] == "eh bueno" and on["events"][0]["text"] == "eh"


def test_block_class_carries_the_not_measurable_warning():
    doc = _doc(
        _result(events=[Event("block", 1.0, 2.0, 0.7, 2, modalities=("acoustic",))]),
        output_scope="all",
    )
    assert "class_block_not_measurable" in {w["code"] for w in doc["warnings"]}


def test_error_document_is_valid_for_every_code():
    from mrcd.service.contract import ERROR_CODES

    for code in ERROR_CODES:
        doc = error_document(code, "message")
        assert validate_document(doc) == [], code
        assert (
            doc["status"] == "failed" and doc["events"] == [] and doc["summary"] is None
        )
        assert doc["request"]["audio_duration_s"]["value"] is None  # unknown, not 0


def test_validator_refuses_unsupported_keywords():
    with pytest.raises(ValueError):
        validate({}, {"type": "object", "patternProperties": {}})


@pytest.mark.skipif(
    not FIXTURES.exists(), reason="results/ fixtures are local (git-ignored)"
)
def test_every_generated_fixture_validates_against_the_contract():
    files = sorted(FIXTURES.glob("*.json"))
    assert files, "fixtures directory is empty"
    for f in files:
        assert validate_document(json.loads(f.read_text(encoding="utf-8"))) == [], (
            f.name
        )


def test_score_kind_is_not_called_probability_and_legacy_alias_still_validates():
    from mrcd.service.contract import (
        CONTRACT_REVISION,
        SCORE_KIND,
        SCORE_KIND_DEPRECATED,
    )

    assert SCORE_KIND == "max_window_softmax_score"
    assert SCORE_KIND_DEPRECATED == "max_window_class_probability"
    assert CONTRACT_REVISION == "1.4"
    doc = _doc()
    assert doc["provenance"]["service"]["contract_revision"] == "1.4"
    assert doc["provenance"]["service"]["contract"] == "mrcd.analysis/v1"
    # old consumers' documents (revision 1.0) keep validating through the deprecated alias
    doc["events"][0]["score"]["kind"] = SCORE_KIND_DEPRECATED
    assert validate_document(doc) == []
    doc["events"][0]["score"]["kind"] = "calibrated_probability"
    assert validate_document(doc) != []
