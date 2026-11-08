"""Decision profile o2026_12 (plan A, opt-in): provenance, artifact integrity, single P1 pass, event mapping."""

import json

import numpy as np
import pytest

from mrcd.service import contextual as CX
from mrcd.service import decision_profile as DP
from mrcd.service.analyzer import (
    AnalysisOptions,
    MrcdAnalyzer,
    ServiceError,
    build_analysis,
)
from mrcd.service.contract import validate_document
from tests.test_decision_profile import StubEngine, _result_with, _v0_events, _words


# ------------------------------------------------------------------------------------------- artifacts / provenance
def test_pins_match_the_files_and_the_source_record():
    src = json.loads((DP.PROFILE_12_DIR / "SOURCE.json").read_text("utf8"))
    assert src["profile"] == "o2026_12" and src["calibration_claim"] == "none"
    assert src["asr_profile_trained_on"] == DP.P1_ASR_PROFILE
    assert len(src["protocol_sha256"]) == 64
    for name, pin in DP.PINNED_SHA256_12.items():
        assert (
            DP.sha256_hex((DP.PROFILE_12_DIR / name).read_bytes())
            == pin
            == src["files"][name]["sha256"]
        )


def test_o12_pipelines_were_fitted_on_p1_with_the_o2026_11_blocks_and_lr():
    p11, p12 = DP.o_pipelines(), DP.o12_pipelines()
    for cls in DP.O_CLASSES:
        a, b = p11[cls].cfg, p12[cls].cfg
        assert b.profile == "p1" and b.model == "lr"
        assert b.blocks == a.blocks and b.fusion == a.fusion
        assert dict(b.hp)["C"] in (0.03, 0.3, 3.0)


def test_profile_meta_declares_single_pass_rules_and_counts():
    meta = DP.profile_meta(DP.PROFILE_12, stats={"repetition": {"n_candidates": 3}})
    assert meta["name"] == "o2026_12" and meta["status"] == "exploratory"
    assert meta["calibration_claim"] == "none"
    rep = meta["classes"]["repetition"]
    assert set(rep["demotion_rules"]) == set(CX.REPETITION_RULES)
    assert rep["emitted_states"] == ["disfluent"] and rep["candidates"] == {
        "n_candidates": 3
    }
    for cls in DP.O_CLASSES:
        c = meta["classes"][cls]
        assert c["asr_profile_used"] == c["asr_profile_trained_on"] == DP.P1_ASR_PROFILE
        assert c["artifact_sha256"] == DP.PINNED_SHA256_12[f"{cls}.json"]
    assert set(meta["classes"]["filler_word"]["subclasses"]) == set(
        CX.FILLER_SUBCLASSES
    )
    assert DP.replaced_classes(DP.PROFILE_12) == (
        "filler_word",
        "prolongation",
        "repetition",
    )


def test_o2026_12_owns_its_asr_profile_and_the_default_is_unchanged():
    assert (
        AnalysisOptions().decision_profile == "o2026_11"
        and AnalysisOptions().asr_profile == "p0"
    )
    o = AnalysisOptions(decision_profile="o2026_12")
    assert o.asr_profile == DP.P1_ASR_PROFILE
    MrcdAnalyzer.validate_options(o)
    with pytest.raises(ServiceError):
        MrcdAnalyzer.validate_options(
            AnalysisOptions(decision_profile="o2026_12", asr_profile="cw2")
        )
    # other profiles keep the requested ASR profile untouched
    assert AnalysisOptions(decision_profile="o2026_11b").asr_profile == "p0"


# ------------------------------------------------------------------------------------------- events
def _audio_and_words():
    sr = 16000
    audio = np.zeros(sr * 20, np.float32)
    words = _words(
        [
            ("quería", 1.0, 1.3),
            ("el", 1.35, 1.5),
            ("el", 1.55, 1.7),
            ("horario", 1.75, 2.2),
            ("si", 5.0, 5.2),
            ("si", 5.25, 5.45),
            ("pues", 9.0, 9.2),
            ("eh", 9.5, 9.8),
            ("mañana", 10.0, 10.4),
        ]
    )
    return audio, sr, words


def test_apply_profile_12_emits_only_disfluent_candidates_and_counts_the_rest():
    audio, sr, words = _audio_and_words()
    events, stats = DP.apply_profile_12(StubEngine(), audio, sr, words)
    rep = [e for e in events if e["class"] == "repetition"]
    assert [(e["start_s"], e["end_s"]) for e in rep] == [
        (1.55, 1.7)
    ]  # "el el"; "si si" is functional
    assert stats["repetition"]["by_state"] == {
        "disfluent": 1,
        "functional": 1,
        "uncertain": 0,
    }
    lex = [
        e
        for e in events
        if (e.get("context") or {}).get("subclass") == CX.LEXICAL_FILLER
    ]
    assert [e["text"] for e in lex] == [
        "pues"
    ]  # 0.3 s gap after "pues" is hesitation evidence
    assert all(
        e["context"]["state"] == CX.DISFLUENT for e in events if e.get("context")
    )
    filled = [
        e for e in events if (e.get("context") or {}).get("subclass") == CX.FILLED_PAUSE
    ]
    assert all(e["class"] == "filler_word" for e in filled)


def test_document_validates_and_drops_the_v0_events_of_replaced_classes():
    audio, sr, words = _audio_and_words()
    events, stats = DP.apply_profile_12(StubEngine(), audio, sr, words)
    opts = AnalysisOptions(decision_profile="o2026_12")
    doc = build_analysis(
        _result_with(_v0_events(), words=words),
        options=opts,
        audio=None,
        engine_info={},
        profile_events=events,
        profile_stats=stats,
    )
    assert validate_document(doc) == []
    prov = doc["provenance"]["decision_profile"]
    assert (
        prov["name"] == "o2026_12"
        and prov["classes"]["repetition"]["candidates"]["n_candidates"] == 2
    )
    assert doc["request"]["asr_profile"] == DP.P1_ASR_PROFILE
    codes = {w["code"] for w in doc["warnings"]}
    assert (
        "decision_profile_exploratory" in codes
        and "decision_profile_asr_mismatch" not in codes
    )
    # no V0 (window softmax) event of a replaced class survives; repetition and filler events carry a context
    for ev in doc["events"]:
        if ev["class"] in DP.replaced_classes("o2026_12"):
            assert ev["score"]["kind"] in (
                "unit_pipeline_model_score",
                "lexical_rule_indicator",
            )
            if ev["class"] in ("repetition", "filler_word"):
                assert ev["context"]["state"] == "disfluent"
            else:
                assert "context" not in ev
    assert any(ev["class"] == "repetition" for ev in doc["events"])
