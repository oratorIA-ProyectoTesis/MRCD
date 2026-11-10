"""Decision profile o2026_13 (plan F, opt-in): ft_small_v1 single pass, primary repetition rule, ASR filler tokens."""

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
from mrcd.service.contract import DECISION_PROFILE_NAMES, validate_document
from tests.test_decision_profile import StubEngine, _result_with, _v0_events, _words


# ------------------------------------------------------------------------------------------- opt-in / default
def test_o2026_13_is_registered_but_the_default_is_unchanged():
    assert DP.PROFILE_13 == "o2026_13" and DP.PROFILE_13 in DP.DECISION_PROFILES
    assert "o2026_13" in DECISION_PROFILE_NAMES
    assert DP.DEFAULT_DECISION_PROFILE == "o2026_11"
    o = AnalysisOptions()
    assert o.decision_profile == "o2026_11" and o.asr_profile == "p0"
    assert (
        DP.required_asr_profile("o2026_11") is None
        and DP.required_asr_profile("o2026_12") == DP.P1_ASR_PROFILE
    )


def test_o2026_13_owns_ft_small_v1_and_rejects_other_asr_profiles():
    o = AnalysisOptions(decision_profile="o2026_13")
    assert o.asr_profile == DP.FT_ASR_PROFILE == "ft_small_v1"
    MrcdAnalyzer.validate_options(o)
    with pytest.raises(ServiceError):
        MrcdAnalyzer.validate_options(
            AnalysisOptions(
                decision_profile="o2026_13", asr_profile="p1_disfluent_prompt"
            )
        )
    assert AnalysisOptions(decision_profile="o2026_12").asr_profile == DP.P1_ASR_PROFILE


def test_o2026_13_replaces_all_three_classes_and_reuses_the_o2026_11_prolongation_pipeline():
    assert DP.replaced_classes("o2026_13") == (
        "filler_word",
        "prolongation",
        "repetition",
    )
    assert DP.pipelines_for("o2026_13") is DP.pipelines_for(
        "o2026_11"
    )  # the same verified, unmodified artifacts
    assert (
        DP.trained_asr_profile("o2026_13") == "p0"
    )  # the prolongation head was fitted on P0 text


def test_contract_ts_client_lists_the_profile():
    from pathlib import Path

    ts = (
        Path(__file__).resolve().parents[1] / "clients" / "typescript" / "mrcd.ts"
    ).read_text("utf8")
    assert '"o2026_12" | "o2026_13"' in ts


# ------------------------------------------------------------------------------------------- pure rule wiring
def _w(text, start, end):
    return _words([(text, start, end)])[0]


def test_primary_rule_has_no_demotion_layer():
    # "si si" (polarity), a copy pair 2 s apart (gap) and a loop would all be demoted by o2026_12; here all are events.
    words = _words(
        [
            ("si", 1.0, 1.2),
            ("si", 1.25, 1.45),
            ("el", 5.0, 5.2),
            ("el", 7.5, 7.7),
            ("na", 9.0, 9.1),
            ("na", 9.15, 9.25),
            ("na", 9.3, 9.4),
            ("na", 9.45, 9.55),
            ("na", 9.6, 9.7),
        ]
    )
    cands = CX.primary_repetition_candidates(words)
    assert cands and all(
        c["state"] == CX.DISFLUENT and c["rule"] is None for c in cands
    )
    demoted = CX.classify_repetitions(words, 20.0)
    assert any(
        c["state"] != CX.DISFLUENT for c in demoted
    )  # the o2026_12 layer would have demoted some
    # identical to the p0_rule / p1_rule generator (rules off)
    assert cands == CX.classify_repetitions(words, 20.0, rules=())


def test_ft_filler_words_are_the_filled_pause_tokens_and_not_lexical_items():
    words = _words(
        [
            ("Eh", 0.0, 0.2),
            ("quería", 0.3, 0.6),
            ("mm,", 0.7, 0.9),
            ("ah", 1.0, 1.2),
            ("ehh", 1.3, 1.5),
            ("este", 1.6, 1.8),  # lexical item: not a filled pause
            ("pues", 1.9, 2.1),
            ("e", 2.2, 2.3),  # conjunction before "i..."
            ("indio", 2.4, 2.7),
            ("hola", 3.0, 3.3),
        ]
    )
    got = CX.ft_filler_words(words)
    assert [g["token"] for g in got] == ["eh", "mm", "ah", "ehh"]
    assert [g["word_index"] for g in got] == [0, 2, 3, 4]
    assert got[0]["start_s"] == 0.0 and got[0]["end_s"] == 0.2


def test_ft_filler_tokens_follow_the_plan_b_convention():
    from mrcd import ft_whisper

    tokens = [t for t in set(ft_whisper.FT_FILLER_TOKENS.values())]
    words = _words([(t, i * 0.3, i * 0.3 + 0.2) for i, t in enumerate(sorted(tokens))])
    assert sorted(g["token"] for g in CX.ft_filler_words(words)) == sorted(tokens)


def test_ft_filler_predicate_equals_the_plan_b_scoring_predicate():
    from mrcd.context_eval.transcript_reference import is_nonlexical_filler

    toks = [
        "eh",
        "e",
        "ehm",
        "em",
        "mm",
        "mmm",
        "hm",
        "ah",
        "aah",
        "ahh",
        "este",
        "pues",
        "he",
        "a",
        "i",
        "o",
    ]
    for nxt in (None, "indio", "hola"):
        for t in toks:
            assert CX.is_filler(t, nxt) == is_nonlexical_filler(t, nxt), (t, nxt)


# ------------------------------------------------------------------------------------------- events / document
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


def test_apply_profile_13_events_use_the_primary_rule_and_filler_tokens():
    audio, sr, words = _audio_and_words()
    events, stats = DP.apply_profile_13(StubEngine(), audio, sr, words)
    rep = [e for e in events if e["class"] == "repetition"]
    assert [(e["start_s"], e["end_s"]) for e in rep] == [
        (1.55, 1.7),
        (5.25, 5.45),
    ]  # "si si" is NOT demoted
    assert all(
        e["context"]
        == {"state": "disfluent", "subclass": None, "basis": "rule_primary_lexical"}
        for e in rep
    )
    fil = [e for e in events if e["class"] == "filler_word"]
    assert [e["text"] for e in fil] == ["eh"] and fil[0]["context"][
        "subclass"
    ] == CX.FILLED_PAUSE
    assert (fil[0]["start_s"], fil[0]["end_s"]) == (9.5, 9.8)
    assert stats["repetition"]["n_candidates"] == 2 and stats["filler_word"] == {
        "n_candidates": 1
    }
    assert events == sorted(events, key=lambda e: (e["start_s"], e["end_s"]))
    # prolongation, if any, comes from the unit pipeline (no context field), never from the rule/token routes
    assert all("context" not in e for e in events if e["class"] == "prolongation")


def test_profile_meta_and_warning_are_truthful():
    meta = DP.profile_meta(
        DP.PROFILE_13,
        stats={"repetition": {"n_candidates": 2}, "filler_word": {"n_candidates": 1}},
    )
    assert (
        meta["name"] == "o2026_13"
        and meta["status"] == "exploratory"
        and meta["calibration_claim"] == "none"
    )
    cl = meta["classes"]
    assert (
        cl["repetition"]["route"] == "primary_lexical_rule"
        and "NO demotion" in cl["repetition"]["rule"]
    )
    assert cl["repetition"]["candidates"] == {"n_candidates": 2}
    assert "eh" in cl["filler_word"]["tokens"] and "mm" in cl["filler_word"]["tokens"]
    pr = cl["prolongation"]
    assert (
        pr["artifact_sha256"] == DP.PINNED_SHA256["prolongation.json"]
        and pr["asr_profile_trained_on"] == "p0"
    )
    assert all(c["asr_profile_used"] == "ft_small_v1" for c in cl.values())
    w = DP.warning_for(DP.PROFILE_13)
    assert (
        w["code"] == "decision_profile_exploratory"
        and "36 held-out DIHANA speakers" in w["message"]
    )
    assert "not refitted" in w["message"] or "not refitted" in meta["note"]


def test_document_validates_and_drops_the_v0_events_of_replaced_classes():
    audio, sr, words = _audio_and_words()
    events, stats = DP.apply_profile_13(StubEngine(), audio, sr, words)
    opts = AnalysisOptions(decision_profile="o2026_13")
    doc = build_analysis(
        _result_with(_v0_events(), words=words),
        options=opts,
        audio=None,
        engine_info={},
        profile_events=events,
        profile_stats=stats,
    )
    assert validate_document(doc) == []
    assert (
        doc["request"]["asr_profile"] == "ft_small_v1"
        and doc["request"]["decision_profile"] == "o2026_13"
    )
    assert doc["provenance"]["decision_profile"]["name"] == "o2026_13"
    msgs = {w["code"]: w["message"] for w in doc["warnings"]}
    assert (
        "decision_profile_exploratory" in msgs
        and "prolongation pipelines were trained on P0"
        in msgs["decision_profile_asr_mismatch"]
    )
    for ev in doc["events"]:
        if ev["class"] in ("repetition", "filler_word"):
            assert (
                ev["score"]["kind"] == "lexical_rule_indicator"
                and ev["context"]["state"] == "disfluent"
            )
    assert any(ev["class"] == "repetition" for ev in doc["events"])
