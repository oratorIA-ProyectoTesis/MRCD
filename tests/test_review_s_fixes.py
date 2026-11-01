"""Review S (2026-11-01) fixes: pins for behaviour that mutation testing showed to be untested.

Covers: server default output scope, zlib branch of ``chunk_looped``, experimental flags of the rep_ctx_p1 route,
unresolved re-decoded chunk keeps the ORIGINAL text, chunk offset / non-reproducibility marks of the re-decode,
service handling of the opt-in ASR profiles and the ``device='auto'`` GPU warning.
"""

from __future__ import annotations

import numpy as np
import pytest
from starlette.testclient import TestClient

from mrcd.asr_profiles import (
    LOOP_REDECODE_LADDER,
    chunk_looped,
    get_profile,
    loop_redecode_warning,
)
from mrcd.engine import DisfluencyEngine, Event
from mrcd.extractors.linguistic import VerbatimASR
from mrcd.service import AnalysisOptions, MrcdAnalyzer, validate_document
from mrcd.service.server import _query_options, create_app
from tests.test_asr_noloop_profile import AUDIO, FakeModel, _Seg, _asr
from tests.test_service_analyzer import FakeEngine, wav
from tests.test_service_contract import _doc, _result


# ------------------------------------------------------------------ HTTP server: default scope (M1c)
def test_server_query_defaults_to_the_evaluated_scope():
    assert _query_options({}).output_scope == "evaluated"
    assert _query_options({"include_out_of_scope": "false"}).output_scope == "evaluated"
    assert _query_options({"include_out_of_scope": "true"}).output_scope == "all"


def test_server_job_without_the_flag_runs_with_the_evaluated_scope():
    analyzer = MrcdAnalyzer("u.pt", engine_factory=lambda o: FakeEngine())
    client = TestClient(create_app(analyzer))
    queued = client.post("/v1/jobs", content=wav()).json()
    import time

    for _ in range(500):
        job = client.get(f"/v1/jobs/{queued['job_id']}").json()
        if job["status"] in ("completed", "failed"):
            break
        time.sleep(0.02)
    assert job["result"]["request"]["output_scope"] == "evaluated"


# ------------------------------------------------------------------ zlib branch of chunk_looped (M2a)
def test_chunk_looped_zlib_branch_has_its_threshold_at_2_4():
    # no back-to-back n-gram run (the n-gram scan does not fire) but a zlib ratio of ~3.0 on >= 80 characters
    repetitive = (
        "el perro come pan el gato come pan el loro come pan el pez come pan " * 2
    )
    assert len(repetitive) >= 80
    assert chunk_looped(repetitive)
    # ratio ~1.95 (< 2.4): ordinary repeated phrasing is not flagged
    ordinary = "el perro come pan el gato come pan el loro come pan el pez come pan la vaca come pan el toro come pan"
    assert len(ordinary) >= 80 and not chunk_looped(ordinary)
    # the 80-character floor: a compressible short text is not flagged by the ratio
    assert not chunk_looped("el perro come pan el gato come pan")


# ------------------------------------------------------------------ rep_ctx_p1 experimental flags (M4a/b/c)
def test_rep_ctx_p1_route_info_is_marked_experimental_and_not_validated_outside_dihana():
    from mrcd.service.s3_features import route_info

    info = route_info()
    assert info["experimental"] is True
    assert info["validated_outside_dihana"] is False
    assert "turn-level model" in info["unit"]


def test_rep_ctx_p1_events_are_nonempty_experimental_and_say_so():
    words = [
        {
            "text": t,
            "norm": t,
            "start": 0.2 * i,
            "end": 0.2 * i + 0.15,
            "prob": 0.9,
            "punct_after": "",
        }
        for i, t in enumerate("la la la casa casa de de mi mi tia".split())
    ]
    r = _result(
        events=[Event("repetition", 0.0, 1.0, 0.8, 2, modalities=("acoustic",))]
    )
    r.words = words
    doc = _doc(r, text_route="rep_ctx_p1", asr_profile="p1_disfluent_prompt")
    assert validate_document(doc) == []
    reps = [e for e in doc["events"] if e["class"] == "repetition"]
    assert reps, (
        "the route must emit at least one event for this input, otherwise the flags below are not exercised"
    )
    for e in reps:
        assert e["experimental"] is True
        assert (
            "experimental text route" in e["validation"]
            and "not validated outside DIHANA" in e["validation"]
        )
    route = doc["provenance"]["text_route"]
    assert route["experimental"] is True and route["validated_outside_dihana"] is False


# ------------------------------------------------------------------ re-decode: original kept, offset, reproducibility
class _TwoLoopModel(FakeModel):
    """Original decode loops on 'no'; every re-decode loops on 'si', so the two texts are distinguishable."""

    def transcribe(self, audio, **kw):
        self.calls.append(dict(kw))
        looped_word = "no" if kw.get("initial_prompt") is not None else "si"
        return [_Seg([looped_word] * 10)], None


def test_unresolved_chunk_keeps_the_original_text_not_the_last_redecode():
    a = _asr("p0_noloop", _TwoLoopModel())
    words = a.transcribe(AUDIO)
    assert [w.norm for w in words] == ["no"] * 10  # NOT ["si"] * 10
    (ev,) = a.redecode_events
    assert ev["attempt"] is None and ev["kept_original"] is True


def test_redecode_events_record_the_chunk_offset_and_reproducibility():
    class Rung2Only(FakeModel):
        def transcribe(self, audio, **kw):
            self.calls.append(dict(kw))
            if (
                kw.get("initial_prompt") is not None
                or kw.get("temperature", 0.0) == 0.0
            ):
                return [_Seg(["no"] * 10)], None
            return [
                _Seg(["hola", "buenos", "dias"])
            ], None  # only the sampled rung (T=0.4) is clean

    a = _asr("p0_noloop", Rung2Only())
    audio = np.zeros(16000 * 20, np.float32)
    words = a.transcribe_chunked(audio, chunk_s=8.0, overlap_s=2.0)
    assert words
    offsets = [e["chunk_offset_s"] for e in a.redecode_events]
    assert offsets[:3] == [
        0.0,
        6.0,
        12.0,
    ]  # chunk starts every chunk_s - overlap_s seconds
    assert all(
        e["attempt"] == len(LOOP_REDECODE_LADDER) and e["reproducible"] is False
        for e in a.redecode_events
    )
    assert a._chunk_offset_s == 0.0  # reset after the call


def test_deterministic_rung_is_marked_reproducible_and_single_pass_offset_is_zero():
    a = _asr(
        "p0_noloop", FakeModel()
    )  # loops only while a prompt is present: rung 1 (T=0) is clean
    a.transcribe_chunked(AUDIO)  # short audio: single pass
    (ev,) = a.redecode_events
    assert (
        ev["attempt"] == 1
        and ev["reproducible"] is True
        and ev["chunk_offset_s"] == 0.0
    )


def test_warning_and_metadata_state_non_reproducibility():
    msg = loop_redecode_warning(
        [{"attempt": 2, "reproducible": False}, {"attempt": 1, "reproducible": True}]
    )
    assert "NOT bit-reproducible" in msg and "1 replaced chunk(s)" in msg
    assert "NOT bit-reproducible" not in loop_redecode_warning(
        [{"attempt": 1, "reproducible": True}]
    )

    class Rung2Only(FakeModel):
        def transcribe(self, audio, **kw):
            self.calls.append(dict(kw))
            clean = kw.get("temperature", 0.0) > 0
            return [_Seg(["hola", "mundo"] if clean else ["no"] * 10)], None

    a = _asr("p0_noloop", Rung2Only())
    a.model_size, a.device, a.compute_type = "small", "cpu", "int8"
    a._model_info_done, a._model_bin_sha256, a._model_dir = True, "b" * 64, "x"
    words = a.transcribe_chunked(AUDIO)
    eng = DisfluencyEngine.__new__(DisfluencyEngine)
    eng._asr = a
    meta = DisfluencyEngine._asr_profile_meta(
        eng, AUDIO, " ".join(w.text for w in words), 8.0
    )
    lr = meta["loop_redecode"]
    assert lr["n_non_reproducible"] == 1 and lr["reproducible"] is False


# ------------------------------------------------------------------ device='auto' ending on GPU (metadata warning)
def _meta_for(profile, device, compute_type):
    a = VerbatimASR.__new__(VerbatimASR)
    a.model, a.language, a.profile = FakeModel(), "es", get_profile(profile)
    a.model_size, a.device, a.compute_type = "small", device, compute_type
    a._model_info_done, a._model_bin_sha256, a._model_dir = True, "b" * 64, "x"
    eng = DisfluencyEngine.__new__(DisfluencyEngine)
    eng._asr = a
    return DisfluencyEngine._asr_profile_meta(eng, AUDIO, "hola", 8.0)


def test_p0_on_gpu_through_auto_carries_a_device_warning_and_cpu_does_not():
    assert "device_warning" not in _meta_for(
        "p0", "cpu", "int8"
    )  # default CPU output unchanged
    w = _meta_for("p0", "cuda", "float16")["device_warning"]
    assert "CPU int8" in w and "device='auto'" in w
    # a profile that DEMANDS cuda is not a surprise: no warning
    assert "device_warning" not in _meta_for("p0_cuda", "cuda", "float16")


# ------------------------------------------------------------------ service: opt-in profiles are accepted AND marked
@pytest.mark.parametrize("name", ["p0_noloop", "p1_noloop", "p0_cuda", "p1_cuda"])
def test_service_accepts_the_opt_in_profiles_as_experimental_with_a_specific_warning(
    name,
):
    MrcdAnalyzer.validate_options(AnalysisOptions(asr_profile=name))
    r = _result(asr_experimental=True)
    r.metadata["asr_profile"]["name"] = name
    doc = _doc(r, asr_profile=name)
    assert validate_document(doc) == []
    assert (
        doc["experimental"]["active"]
        and f"asr_profile:{name}" in doc["experimental"]["components"]
    )
    codes = {w["code"] for w in doc["warnings"]}
    assert "experimental_components" in codes
    assert ("asr_gpu_text_unvalidated" in codes) == name.endswith("_cuda")
    assert ("asr_redecode_experimental" in codes) == name.endswith("_noloop")


def test_service_default_profiles_get_no_profile_specific_warning():
    codes = {w["code"] for w in _doc()["warnings"]}
    assert not codes & {
        "asr_gpu_text_unvalidated",
        "asr_redecode_experimental",
        "asr_device_auto_gpu",
    }


def test_service_reports_non_reproducible_chunks_and_the_auto_gpu_warning():
    r = _result(asr_experimental=True)
    r.metadata["asr_profile"].update(
        name="p0_noloop",
        loop_redecode={"n_non_reproducible": 2},
        device_warning="ran on cuda by auto",
    )
    doc = _doc(r, asr_profile="p0_noloop")
    msgs = {w["code"]: w["message"] for w in doc["warnings"]}
    assert (
        "2 chunk(s)" in msgs["asr_redecode_experimental"]
        and "not bit-reproducible" in msgs["asr_redecode_experimental"]
    )
    assert msgs["asr_device_auto_gpu"] == "ran on cuda by auto"


def test_cuda_profile_without_a_device_is_a_structured_error_that_says_so():
    from mrcd.asr_profiles import ASRDeviceUnavailable

    def factory(options):
        raise ASRDeviceUnavailable(
            "profile 'p0_cuda' requires CUDA but it is unavailable (secret path C:\\x)"
        )

    doc = MrcdAnalyzer("u.pt", engine_factory=factory).analyze_bytes(
        wav(), AnalysisOptions(asr_profile="p0_cuda")
    )
    assert validate_document(doc) == []
    err = doc["error"]
    assert err["code"] == "asr_unavailable"
    assert (
        "CUDA" in err["message"]
        and "never fall back" in err["message"]
        and "local cache" not in err["message"]
    )
    assert err["details"] == {
        "exception_type": "ASRDeviceUnavailable",
        "asr_profile": "p0_cuda",
    }
    assert "secret" not in str(doc)
