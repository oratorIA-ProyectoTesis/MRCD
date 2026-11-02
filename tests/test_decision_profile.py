"""Decision profiles of the service (contract rev 1.4): o2026_11 (default) and v0 (historical)."""

import copy
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from mrcd.optimize import product_pipeline as PP
from mrcd.optimize import safe_pipeline as SAFE
from mrcd.service import decision_profile as DP
from mrcd.service.analyzer import AnalysisOptions, build_analysis
from mrcd.service.contract import (
    DECISION_PROFILE_NAMES,
    SCORE_KIND_UNIT_PIPELINE,
    validate_document,
)

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models" / "o_campaign_20261101"
GOLDEN = ROOT / "tests" / "data" / "o2026_11_regression.json"


# ------------------------------------------------------------------------------------------ artifacts
def test_pinned_hashes_match_the_files_and_the_export_record():
    src = json.loads((DP.PROFILE_DIR / "SOURCE.json").read_text("utf8"))
    for name, pin in DP.PINNED_SHA256.items():
        assert (
            DP.sha256_hex((DP.PROFILE_DIR / name).read_bytes())
            == pin
            == src["files"][name]["sha256"]
        )
    for c in DP.O_CLASSES:
        assert (
            src["files"][f"{c}.json"]["source_pickle_sha256"]
            == DP.SOURCE_PICKLE_SHA256[c]
        )


@pytest.mark.skipif(not MODELS.is_dir(), reason="campaign artifacts not present")
def test_source_pickle_hashes_match_the_campaign_manifest():
    man = json.loads((MODELS / "MANIFEST.json").read_text("utf8"))
    for c in DP.O_CLASSES:
        assert (
            man["files"][f"o_pipeline_{c}.pkl"]["sha256"] == DP.SOURCE_PICKLE_SHA256[c]
        )


def test_altered_artifact_is_refused(tmp_path):
    shutil.copytree(DP.PROFILE_DIR, tmp_path / "p")
    f = tmp_path / "p" / "filler_word.json"
    doc = json.loads(f.read_text("utf8"))
    doc["thr"] = 0.0  # a plausible tamper: valid JSON, different decision
    f.write_bytes(SAFE.dumps(doc))
    with pytest.raises(DP.ProfileArtifactError, match="sha256 mismatch"):
        DP.load_o_pipelines(tmp_path / "p")
    f.unlink()
    with pytest.raises(DP.ProfileArtifactError, match="missing"):
        DP.load_o_pipelines(tmp_path / "p")


def test_altered_artifact_gives_a_failed_document_not_a_silent_fallback(
    monkeypatch, tmp_path
):
    from tests.test_service_analyzer import analyzer, wav

    shutil.copytree(DP.PROFILE_DIR, tmp_path / "p")
    (tmp_path / "p" / "prolongation.json").write_bytes(b"{}")
    monkeypatch.setattr(DP, "PROFILE_DIR", tmp_path / "p")
    DP.o_pipelines.cache_clear()
    try:
        doc = analyzer().analyze_bytes(wav())
        assert doc["status"] == "failed" and validate_document(doc) == []
        assert doc["error"]["code"] == "analysis_failed"
        assert "decision_profile=v0" in doc["error"]["message"]
        ok = analyzer().analyze_bytes(wav(), AnalysisOptions(decision_profile="v0"))
        assert ok["status"] == "completed"
    finally:
        DP.o_pipelines.cache_clear()


def test_the_product_never_unpickles():
    for mod in (DP, SAFE):
        src = Path(mod.__file__).read_text("utf8")
        assert "import pickle" not in src and "pickle.load" not in src and "joblib" not in src


def test_profile_names_are_the_same_in_contract_and_service():
    assert tuple(DECISION_PROFILE_NAMES) == DP.DECISION_PROFILES
    assert (
        DP.DEFAULT_DECISION_PROFILE == "o2026_11"
        and AnalysisOptions().decision_profile == "o2026_11"
    )
    with pytest.raises(Exception):
        from mrcd.service.analyzer import MrcdAnalyzer

        MrcdAnalyzer.validate_options(AnalysisOptions(decision_profile="nope"))


# ------------------------------------------------------------------------------------------ regression
def _saved_units():
    g = json.loads(GOLDEN.read_text("utf8"))
    try:
        from mrcd.optimize import data as OD

        feats = OD.load_features()
        recs = [OD._rec(u, feats, ("p0",), None) for u in g["units"]]
    except Exception as exc:  # noqa: BLE001 - corpus-derived features are git-ignored
        pytest.skip(f"saved campaign features not available ({type(exc).__name__})")
    return g, recs


def test_service_pipelines_reproduce_the_saved_campaign_decisions():
    g, recs = _saved_units()
    assert len(recs) == 40
    pipes = DP.o_pipelines()
    for c in DP.O_CLASSES:
        r = pipes[c].predict(recs)
        exp = g["classes"][c]
        assert [int(x) for x in r["decision"]] == exp["decision"]
        assert [str(x) for x in r["state"]] == exp["state"]
        assert np.allclose(r["score"], exp["score"], atol=1e-9)


@pytest.mark.skipif(not MODELS.is_dir(), reason="campaign artifacts not present")
def test_json_export_is_equivalent_to_the_pickle_pipelines():
    _, recs = _saved_units()
    pk = PP.load_pipelines(MODELS)
    js = DP.o_pipelines()
    for c in DP.O_CLASSES:
        a, b = pk[c].predict(recs), js[c].predict(recs)
        assert np.array_equal(a["decision"], b["decision"]) and list(
            a["state"]
        ) == list(b["state"])
        assert np.allclose(a["score"], b["score"], atol=1e-10)


# ------------------------------------------------------------------------------------------ unit records
class StubEngine:
    """Deterministic ``unit_features`` so the segmentation and event mapping are testable without models."""

    def __init__(self, fill=0.0):
        self.calls = []
        self.fill = fill

    def unit_features(self, clip, words):
        self.calls.append((len(clip), copy.deepcopy(words)))
        w = 3  # windows
        rng = np.random.default_rng(len(clip))
        return {
            "probs": rng.random((w, 8)).astype(np.float32),
            "pooled": rng.random((w, 128)).astype(np.float32),
            "pause": {
                "max_internal_pause_s": 0.1,
                "edge_silence_s": 0.2,
                "speech_frac": 0.8,
                "n_frames": len(clip) // 160,
            },
            "n_samples": len(clip),
        }


def _words(spec):
    return [
        {"text": t, "norm": t, "start": s, "end": e, "prob": 0.9, "punct_after": ""}
        for t, s, e in spec
    ]


def test_units_are_pause_delimited_clips_with_retimed_words():
    sr = 16000
    audio = np.zeros(sr * 10, np.float32)
    words = _words(
        [("hola", 1.0, 1.4), ("eh", 1.5, 1.9), ("bueno", 4.0, 4.5), ("si", 4.6, 5.0)]
    )
    eng = StubEngine()
    recs, spans = DP.unit_records(eng, audio, sr, words)
    assert [round(s["start"], 2) for s in spans] == [1.0, 4.0] and [
        round(s["end"], 2) for s in spans
    ] == [1.9, 5.0]
    assert len(recs) == 2 == len(eng.calls)
    n, rel = eng.calls[0]
    assert abs(n / sr - (1.9 + DP.MARGIN_S - (1.0 - DP.MARGIN_S))) < 1e-3
    assert abs(rel[0]["start"] - DP.MARGIN_S) < 1e-3  # times are relative to the clip
    assert recs[0]["text"]["p0"] == "hola eh" and recs[0]["words"]["p0"] == rel


def test_no_words_means_no_units_and_no_events():
    assert DP.unit_records(StubEngine(), np.zeros(32000, np.float32), 16000, []) == (
        [],
        [],
    )
    assert DP.profile_events(DP.o_pipelines(), [], []) == []


def test_profile_events_cover_only_the_pipelined_classes_and_are_scored_in_unit_range():
    sr = 16000
    audio = np.zeros(sr * 40, np.float32)
    words = _words([("eh", 1.0 + 3 * i, 1.4 + 3 * i) for i in range(10)])
    eng = StubEngine()
    recs, spans = DP.unit_records(eng, audio, sr, words)
    ev = DP.profile_events(DP.o_pipelines(), recs, spans)
    assert ev, "ten 'eh' units must trigger the filler rule"
    assert {e["class"] for e in ev} <= set(DP.O_CLASSES)
    for e in ev:
        assert 0.0 <= e["score"] <= 1.0 and e["end_s"] > e["start_s"]
        assert e["decision"]["calibration_claim"] == "none" and e["decision"][
            "state"
        ] in ("present", "absent", "uncertain")


# ------------------------------------------------------------------------------------------ documents
def _result_with(events=(), words=None):
    from mrcd.engine import AnalysisResult

    return AnalysisResult(
        events=list(events),
        transcript="eh",
        duration=20.0,
        mode="audio_text",
        warning="weak supervision",
        metadata={
            "asr_profile": {
                "name": "p0",
                "experimental": False,
                "model": {},
                "anomalies": [],
            },
            "window_plan": {},
            "environment": {},
        },
        words=words,
    )


def _v0_events():
    from mrcd.engine import Event

    mods = ("acoustic", "linguistic")
    return [
        Event("filler_word", 1.0, 2.0, 0.8, 2, modalities=mods),
        Event("prolongation", 3.0, 4.0, 0.7, 2, modalities=mods),
        Event("repetition", 5.0, 6.0, 0.6, 2, modalities=mods),
        Event("block", 7.0, 8.0, 0.5, 2, modalities=mods),
    ]


def _doc(profile, profile_events=None, **kw):
    opts = AnalysisOptions(decision_profile=profile, **kw)
    return build_analysis(
        _result_with(_v0_events()),
        options=opts,
        audio=None,
        engine_info={},
        profile_events=profile_events,
    )


def test_v0_document_keeps_every_engine_event_and_declares_its_profile():
    d = _doc("v0")
    assert validate_document(d) == []
    assert [(e["class"], e["start_s"], e["score"]["kind"]) for e in d["events"]] == [
        ("filler_word", 1.0, "max_window_softmax_score"),
        ("prolongation", 3.0, "max_window_softmax_score"),
        ("repetition", 5.0, "max_window_softmax_score"),
    ]
    assert (
        d["provenance"]["decision_profile"]["name"] == "v0"
        and d["request"]["decision_profile"] == "v0"
    )
    assert "decision_profile_exploratory" not in [w["code"] for w in d["warnings"]]


def test_o_profile_replaces_only_filler_and_prolongation_events():
    pe = [
        {
            "class": "filler_word",
            "start_s": 10.0,
            "end_s": 11.0,
            "score": 0.9,
            "n_windows": 3,
            "text": "eh",
            "decision": {
                "state": "present",
                "estimate": 0.9,
                "lo": 0.36,
                "hi": 0.73,
                "basis": "x",
                "calibration_claim": "none",
            },
        },
    ]
    d = _doc("o2026_11", pe, decision=True)
    assert validate_document(d) == []
    got = [(e["class"], e["start_s"], e["score"]["kind"]) for e in d["events"]]
    assert got == [
        ("repetition", 5.0, "max_window_softmax_score"),
        ("filler_word", 10.0, SCORE_KIND_UNIT_PIPELINE),
    ]
    fe = d["events"][1]
    assert (
        fe["score"]["calibrated"] is False
        and fe["decision"]["state"] == "present"
        and fe["experimental"] is False
    )
    assert d["summary"]["counts_by_class"] == {"repetition": 1, "filler_word": 1}
    prov = d["provenance"]["decision_profile"]
    assert prov["classes"]["repetition"]["route"] == "engine_v0_window_events"
    assert (
        prov["classes"]["filler_word"]["artifact_sha256"]
        == DP.PINNED_SHA256["filler_word.json"]
    )
    assert prov["calibration_claim"] == "none" and prov["status"] == "exploratory"
    assert "decision_profile_exploratory" in [w["code"] for w in d["warnings"]]


def test_o_profile_with_other_asr_profile_warns_about_the_mismatch():
    d = _doc("o2026_11", [], asr_profile="p1_disfluent_prompt")
    assert "decision_profile_asr_mismatch" in [w["code"] for w in d["warnings"]]


def test_documents_without_the_rev_1_4_fields_still_validate():
    d = _doc("v0")
    old = copy.deepcopy(d)
    del old["request"]["decision_profile"]
    del old["provenance"]["decision_profile"]
    old["provenance"]["service"]["contract_revision"] = "1.3"
    assert validate_document(old) == []


# ------------------------------------------------------------------------------------------ frozen v0 document
FROZEN_V0 = ROOT / "tests" / "data" / "v0_frozen_document.json"
FROZEN_V0_SHA256 = "22f5b0080cf0e2bf9894ba3baff9b8e5ac28a0ffc642468d93feb9c82282f0dc"
REV_1_4_FIELDS = (("request", "decision_profile"), ("provenance", "decision_profile"))


def _frozen_input_doc():
    """Deterministic v0 document: fixed engine result (frozen ASR text/events, no model, no Whisper)."""
    opts = AnalysisOptions(decision_profile="v0", include_text=True, decision=True, output_scope="all")
    d = build_analysis(_result_with(_v0_events()), options=opts, audio=None, engine_info={}, profile_events=None)
    d.pop("generated_at")
    for a, b in REV_1_4_FIELDS:
        d[a].pop(b)
    d["provenance"]["service"].pop("contract_revision")
    d["provenance"]["service"].pop("mrcd_version")
    return d


def test_v0_document_equals_the_frozen_reference():
    blob = FROZEN_V0.read_bytes()
    assert DP.sha256_hex(blob) == FROZEN_V0_SHA256, "frozen reference was edited"
    assert json.loads(blob) == _frozen_input_doc()


def test_missing_words_warn_instead_of_silently_returning_nothing():
    d = _doc("o2026_11", [])
    w = [x for x in d["warnings"] if x["code"] == "decision_profile_no_words"]
    assert w and w[0]["message"].startswith("Not available")
