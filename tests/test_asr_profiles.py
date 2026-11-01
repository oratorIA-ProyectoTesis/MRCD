"""Opt-in ASR profiles: p0 is the historical default, p1_disfluent_prompt is experimental and fully recorded."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest

from mrcd.asr_profiles import (
    EXPERIMENTAL_NOTE,
    P1_PROMPT,
    PROFILES,
    asr_cache_key,
    detect_asr_anomalies,
    get_profile,
)
from mrcd.constants import WHISPER_INITIAL_PROMPT
from mrcd.engine import DisfluencyEngine
from mrcd.extractors.linguistic import VerbatimASR, Word

REPO = Path(__file__).resolve().parents[1]
CKPT = REPO / "models" / "ckpt_audio_text.pt"
RECOVERY = REPO / "results" / "asr_recovery_20261031"
P0_CACHE = REPO / "results" / "mejoras_20261030" / "asr_cache" / "dihana"
WAV = REPO / "results" / "baselines_20261030" / "dihana" / "wav16k"
WHISPER = (
    Path.home()
    / ".cache/huggingface/hub/models--Systran--faster-whisper-small/snapshots/536b0662742c02347bc0e980a01041f333bce120"
)
P1_PROMPT_SHA256 = "fe49e930ccc9f94847b7bedaec65f8c347cecaccbcd7283545618bc194c6a8b2"

# kwargs the pre-profile VerbatimASR.transcribe passed to WhisperModel.transcribe (historical, hard-coded on purpose)
HISTORICAL_KWARGS = dict(
    language="es",
    word_timestamps=True,
    condition_on_previous_text=False,
    initial_prompt="eh, este, o sea, mmm, bueno, tipo, ah, este...",
    suppress_tokens=[],
    vad_filter=False,
    beam_size=5,
    temperature=0.0,
)


class _Seg:
    def __init__(self):
        self.words = []


class RecordingModel:
    def __init__(self):
        self.calls = []

    def transcribe(self, audio, **kw):
        self.calls.append(kw)
        return [], None


def _asr_with(profile):
    asr = VerbatimASR.__new__(VerbatimASR)
    asr.model = RecordingModel()
    asr.language = "es"
    asr.profile = get_profile(profile)
    return asr


# ----------------------------------------------------------------- configuration
def test_p0_is_the_default_and_its_kwargs_are_the_historical_ones():
    assert get_profile(None).name == "p0"
    assert get_profile("p0").decode_kwargs("es") == HISTORICAL_KWARGS
    assert get_profile("p0").initial_prompt == WHISPER_INITIAL_PROMPT
    asr = _asr_with(None)
    asr.transcribe(np.zeros(16000, np.float32))
    assert asr.model.calls == [HISTORICAL_KWARGS]


def test_p1_changes_only_the_initial_prompt():
    p0, p1 = get_profile("p0"), get_profile("p1_disfluent_prompt")
    k0, k1 = p0.decode_kwargs(), p1.decode_kwargs()
    assert k1.pop("initial_prompt") == P1_PROMPT
    k0.pop("initial_prompt")
    assert k0 == k1
    assert (p1.chunk_s, p1.overlap_s) == (8.0, 2.0) == (p0.chunk_s, p0.overlap_s)
    assert p1.experimental and not p0.experimental
    assert "not validated outside DIHANA" in p1.validation


def test_p1_prompt_is_literal_and_hash_is_pinned():
    expected = (
        "Eh... yo quería, quería preguntar por el, por el... horario. Es que no, no sé si... si es el mismo. "
        "Perdone, digo, perdón. La mesa está en la ca... en la casa, o sea, en la casa de mi tía. "
        "Mmm, este... bueno, sí, sí."
    )
    assert P1_PROMPT == expected
    assert (
        get_profile("p1_disfluent_prompt").prompt_sha256
        == hashlib.sha256(expected.encode("utf8")).hexdigest()
    )
    assert get_profile("p1_disfluent_prompt").prompt_sha256 == P1_PROMPT_SHA256


def test_p1_matches_the_experiment_decode_parameters():
    from mrcd.asr_recovery.decode import P1_PROMPT as EXP_PROMPT
    from mrcd.asr_recovery.decode import decode_kwargs

    assert EXP_PROMPT == P1_PROMPT
    exp = decode_kwargs("P1")
    assert exp == get_profile("p1_disfluent_prompt").decode_kwargs("es")
    asr = _asr_with("p1_disfluent_prompt")
    asr.transcribe(np.zeros(16000, np.float32))
    assert asr.model.calls == [exp]


def test_unknown_profile_is_rejected():
    for bad in ("P1", "p2", "", "p1"):
        with pytest.raises(ValueError, match="unknown ASR profile"):
            get_profile(bad)
    with pytest.raises(ValueError, match="unknown ASR profile"):
        DisfluencyEngine.load(
            CKPT, asr_profile="nope"
        )  # rejected before any model is loaded


# ----------------------------------------------------------------- cache key
def _key(profile, **over):
    base = dict(
        audio_sha256="a" * 64,
        model_bin_sha256="b" * 64,
        compute_type="int8",
        device="cpu",
        omp_num_threads="4",
    )
    base.update(over)
    return asr_cache_key(profile, **base)


def test_cache_key_differs_between_profiles_and_is_deterministic():
    assert _key("p0") != _key("p1_disfluent_prompt")
    assert _key("p0") == _key("p0")
    assert (
        get_profile("p0").cache_namespace()
        != get_profile("p1_disfluent_prompt").cache_namespace()
    )
    assert _key("p0") != _key("p0", audio_sha256="c" * 64)
    assert _key("p0") != _key("p0", omp_num_threads="20")
    assert _key("p0") != _key("p0", model_bin_sha256="d" * 64)


def test_cache_key_depends_on_prompt_hash_even_with_same_name():
    from dataclasses import replace

    # two custom (experimental) profiles with the SAME name and different prompts; a registered name can no
    # longer be reused with other parameters (REVISION7 F-7), so the registered P1 is not used here
    a = replace(PROFILES["p1_disfluent_prompt"], name="variant", initial_prompt="x")
    b = replace(a, initial_prompt="y")
    assert _key(a) != _key(b)


# ----------------------------------------------------------------- anomalies
def test_prompt_echo_is_flagged():
    a = detect_asr_anomalies(
        "hola yo quería quería preguntar por el, por el horario", P1_PROMPT
    )
    assert any(x["kind"] == "prompt_echo" for x in a)
    assert not detect_asr_anomalies(
        "buenos días quiero un billete a valencia", P1_PROMPT
    )


def test_hallucinated_loops_are_flagged_but_genuine_repetitions_are_not():
    loop1 = "gracias " + "si " * 8 + "adios"
    assert any(
        x["kind"] == "repeated_ngram_loop" and x["ngram"] == "si"
        for x in detect_asr_anomalies(loop1, "")
    )
    loop2 = "vale " + "por favor " * 5
    assert any(
        x["kind"] == "repeated_ngram_loop" and x["ngram"] == "por favor"
        for x in detect_asr_anomalies(loop2, "")
    )
    assert not detect_asr_anomalies("quiero quiero un billete la la la ida", "")
    comp = detect_asr_anomalies("de la casa " * 30, "")
    assert any(x["kind"] == "high_compression_ratio" for x in comp)


# ----------------------------------------------------------------- engine metadata
class StubASR:
    compute_type, device, model_size = "int8", "cpu", "stub"
    language = "es"

    def __init__(self, words, profile=None):
        self.words = words
        if profile is not None:
            self.profile = get_profile(profile)

    def transcribe_chunked(self, audio, chunk_s=8.0):
        return self.words

    def transcribe(self, audio):
        return self.words


def _engine(words, profile="p0"):
    from mrcd.inference import load_ckpt

    net, norm, labels, prov = load_ckpt(CKPT, "cpu")
    eng = DisfluencyEngine(net, norm, list(labels), StubASR(words, profile), prov)
    eng._device = "cpu"
    return eng


def _audio():
    rng = np.random.default_rng(0)
    return (0.01 * rng.standard_normal(16000 * 18)).astype(np.float32)


needs_ckpt = pytest.mark.skipif(not CKPT.exists(), reason="base checkpoint not present")
WORDS = [
    Word("bueno", "bueno", 1.0, 1.4, 0.9, ","),
    Word("eh", "eh", 3.0, 3.3, 0.8, ""),
]


@needs_ckpt
def test_default_result_records_p0_and_keeps_warning_unchanged():
    eng = _engine(WORDS)
    r = eng.analyze(_audio())
    m = r.metadata["asr_profile"]
    assert m["name"] == "p0" and m["experimental"] is False
    assert (
        m["initial_prompt_sha256"]
        == hashlib.sha256(WHISPER_INITIAL_PROMPT.encode("utf8")).hexdigest()
    )
    assert r.warning == eng.warning
    assert r.to_dict()["metadata"]["asr_profile"] == m
    json.dumps(r.to_dict())  # serialisable


@needs_ckpt
def test_p1_result_records_everything_and_is_marked_experimental():
    r = _engine(WORDS, "p1_disfluent_prompt").analyze(_audio())
    m = r.metadata["asr_profile"]
    assert m["name"] == "p1_disfluent_prompt" and m["experimental"] is True
    assert (
        m["initial_prompt"] == P1_PROMPT
        and m["initial_prompt_sha256"] == P1_PROMPT_SHA256
    )
    for k in (
        "temperature",
        "beam_size",
        "condition_on_previous_text",
        "vad_filter",
        "chunk_s",
        "overlap_s",
    ):
        assert m[k] == get_profile("p1_disfluent_prompt").describe()[k]
    assert m["model"]["compute_type"] == "int8" and "model_bin_sha256" in m["model"]
    assert "omp_num_threads_effective" in m
    assert m["asr_cache"]["key"] and m["asr_cache"]["engine_cache"] is None
    assert m["asr_cache"]["namespace"].startswith("p1_disfluent_prompt-")
    assert (
        "NOT validated outside DIHANA" in r.warning and EXPERIMENTAL_NOTE in r.warning
    )
    p0 = _engine(WORDS).analyze(_audio()).metadata["asr_profile"]["asr_cache"]["key"]
    assert p0 != m["asr_cache"]["key"]


@needs_ckpt
def test_p1_anomalies_are_reported_and_the_transcript_is_not_filtered():
    echo = [
        Word(t, t, 1.0 + i * 0.3, 1.2 + i * 0.3, 0.9, "")
        for i, t in enumerate("quería quería preguntar por el".split())
    ]
    r = _engine(echo, "p1_disfluent_prompt").analyze(_audio())
    kinds = [a["kind"] for a in r.metadata["asr_profile"]["anomalies"]]
    assert "prompt_echo" in kinds
    assert r.transcript == "quería quería preguntar por el"  # reported, never removed
    assert "anomalies detected and NOT filtered" in r.warning


# ----------------------------------------------------------------- slow: reproduces stored decodes
SLOW_IDS = [
    "B101_BA0c0_U_002",
    "B104_BA0c3_U_002",
    "V110_VC4a1_U_003",
]  # last one > 8 s (chunked)
#: REVISION7 F-3: the slow decode tests compare floats at 1e-6 and are LOAD-SENSITIVE. With the CPU oversubscribed
#: (two concurrent suites, or burners) 6 of them failed (prob 0.256 vs 0.272, "tarde," vs "tarde") while the
#: same tests pass alone. "Bit-identical" is therefore only claimed for an UNLOADED machine with
#: OMP_NUM_THREADS=4 exported before numpy is imported. The fixture below skips (never fails) under high load;
#: set MRCD_SKIP_LOAD_CHECK=1 to force the run anyway.
MAX_BUSY_FRACTION = 0.5


def cpu_busy_fraction(interval: float = 0.5):
    """Fraction of CPU time busy over ``interval`` seconds (Windows: GetSystemTimes; POSIX: /proc/stat or
    loadavg), or ``None`` when it cannot be measured."""
    import time

    try:
        if os.name == "nt":
            import ctypes
            from ctypes import wintypes

            class FT(ctypes.Structure):
                _fields_ = [("lo", wintypes.DWORD), ("hi", wintypes.DWORD)]

            def snap():
                i, k, u = FT(), FT(), FT()
                if not ctypes.windll.kernel32.GetSystemTimes(
                    ctypes.byref(i), ctypes.byref(k), ctypes.byref(u)
                ):
                    raise OSError
                v = lambda f: (f.hi << 32) | f.lo
                return v(i), v(k) + v(u)  # kernel time includes idle

            i0, t0 = snap()
            time.sleep(interval)
            i1, t1 = snap()
            return 1.0 - (i1 - i0) / max(t1 - t0, 1)
        return min(os.getloadavg()[0] / (os.cpu_count() or 1), 1.0)
    except Exception:
        return None


def _load_skip_reason():
    if os.environ.get("MRCD_SKIP_LOAD_CHECK") == "1":
        return None
    busy = cpu_busy_fraction()
    if busy is not None and busy > MAX_BUSY_FRACTION:
        return f"CPU busy {busy:.0%} > {MAX_BUSY_FRACTION:.0%}: float-exact decode comparison is load-sensitive"
    return None


slow = pytest.mark.skipif(
    os.environ.get("MRCD_SLOW") != "1"
    or os.environ.get("OMP_NUM_THREADS") != "4"
    or not WHISPER.exists(),
    reason="set MRCD_SLOW=1 and OMP_NUM_THREADS=4 (before python starts); needs the faster-whisper small snapshot",
)


def _words_equal(got, stored):
    assert [w.text for w in got] == [w["text"] for w in stored]
    for g, s in zip(got, stored):
        assert (
            abs(g.start - s["start"]) < 1e-6
            and abs(g.end - s["end"]) < 1e-6
            and abs(g.prob - s["prob"]) < 1e-6
        )


@pytest.fixture(scope="module")
def real_asr():
    import soundfile as sf  # noqa: F401

    reason = _load_skip_reason()
    if reason:
        pytest.skip(reason)
    return {
        "p0": VerbatimASR(str(WHISPER), device="cpu", compute_type="int8"),
        "p1": VerbatimASR(
            str(WHISPER),
            device="cpu",
            compute_type="int8",
            profile="p1_disfluent_prompt",
        ),
    }


@slow
@pytest.mark.parametrize("uid", SLOW_IDS)
def test_slow_p1_reproduces_the_stored_experiment_decode(real_asr, uid):
    import soundfile as sf

    audio, sr = sf.read(WAV / f"{uid}.wav", dtype="float32")
    stored = json.loads((RECOVERY / "asr" / "P1" / f"{uid}.json").read_text("utf8"))
    assert (
        real_asr["p1"].model_info()["model_bin_sha256"]
        == "3e305921506d8872816023e4c273e75d2419fb89b24da97b4fe7bce14170d671"
    )
    _words_equal(real_asr["p1"].transcribe_chunked(audio), stored["words"])


@slow
@pytest.mark.parametrize("uid", SLOW_IDS)
def test_slow_p0_default_still_reproduces_the_frozen_cache(real_asr, uid):
    import soundfile as sf

    audio, sr = sf.read(WAV / f"{uid}.wav", dtype="float32")
    stored = json.loads((P0_CACHE / f"{uid}.json").read_text("utf8"))
    _words_equal(real_asr["p0"].transcribe_chunked(audio), stored["words"])


# ----------------------------------------------------------------- REVISION7 F-1/F-2: heuristic limits
def test_echo_ignores_function_word_only_overlap_reviewer_case():
    # 'si si es el' matches the prompt only through function words (REVISION7 F-1: genuine repetition)
    assert not [
        x
        for x in detect_asr_anomalies("Pues... sí, sí es el único tren", P1_PROMPT)
        if x["kind"] == "prompt_echo"
    ]
    assert not [
        x
        for x in detect_asr_anomalies("no, no sé si es", P1_PROMPT)
        if x["kind"] == "prompt_echo"
    ]


def test_echo_still_flags_content_overlap_and_documents_in_domain_false_positive():
    assert any(
        x["kind"] == "prompt_echo"
        for x in detect_asr_anomalies("yo quería, quería preguntar por el", P1_PROMPT)
    )
    # reviewer false positives that the function-word rule removes
    assert not [
        x
        for x in detect_asr_anomalies("quiero preguntar por el horario", P1_PROMPT)
        if x["kind"] == "prompt_echo"
    ]
    # known REMAINING false positives, kept visible on purpose: honest speech that shares >= 2 content words
    # with a prompt 4-gram ('casa de mi tia' is the prompt's own tail)
    for t in ("la mesa esta en la casa de mi tia", "quería preguntar por el horario"):
        assert any(x["kind"] == "prompt_echo" for x in detect_asr_anomalies(t, P1_PROMPT))


def test_loop_with_filler_between_repetitions_is_detected():
    a = detect_asr_anomalies("no se no se eh no se no se eh", "")
    assert any(
        x["kind"] == "repeated_ngram_loop" and x["ngram"] == "no se" and x.get("skipping_fillers")
        for x in a
    )
    assert any(x["ngram"] == "eh" for x in detect_asr_anomalies("eh eh eh eh eh eh", ""))
    assert not detect_asr_anomalies("no no no", "")
    assert not detect_asr_anomalies("no se eh no se", "")


def test_known_false_negative_loop_interleaved_with_other_words_is_not_detected():
    # documented limitation (REVISION7 F-2)
    assert not detect_asr_anomalies("si claro si bueno si claro si bueno", "")


def test_cpu_busy_fraction_is_a_probability_or_none():
    b = cpu_busy_fraction(0.1)
    assert b is None or 0.0 <= b <= 1.0


# ----------------------------------------------------------------- REVISION7 F-7/F-8
def test_custom_non_experimental_profile_other_than_p0_is_rejected():
    from dataclasses import replace

    p0 = PROFILES["p0"]
    assert get_profile(p0) is p0
    for bad in (
        replace(p0, name="p0", initial_prompt=P1_PROMPT),  # poses as p0 with the P1 prompt
        replace(p0, name="mine", beam_size=1),  # non-experimental custom
        replace(PROFILES["p1_disfluent_prompt"], initial_prompt="x"),  # reuses the registered P1 name
    ):
        with pytest.raises(ValueError):
            get_profile(bad)
    custom = replace(PROFILES["p1_disfluent_prompt"], name="mine", initial_prompt="x")
    assert get_profile(custom) is custom  # an honest experimental custom profile is fine


@needs_ckpt
def test_asr_without_profile_is_reported_unknown_not_p0():
    r = _engine(WORDS, None).analyze(_audio())
    m = r.metadata["asr_profile"]
    assert m["name"] == "unknown" and m["experimental"] is None and "not verified" in m["validation"]
    assert "initial_prompt_sha256" not in m and m["asr_cache"] is None
    assert r.warning == _engine(WORDS).warning
    json.dumps(r.to_dict())


@needs_ckpt
def test_p1_events_carry_the_experimental_mark_and_p0_events_do_not():
    words = [Word(t, t, 1.0 + i * 0.4, 1.3 + i * 0.4, 0.9, "") for i, t in enumerate("eh eh eh bueno".split())]
    audio = _audio()
    p1 = _engine(words, "p1_disfluent_prompt").analyze(audio)
    p0 = _engine(words).analyze(audio)
    assert all(e.asr_profile == "p1_disfluent_prompt" for e in p1.events)
    assert all(d["asr_profile"] == "p1_disfluent_prompt" and d["asr_experimental"] is True for d in p1.to_dict()["events"])
    assert all(e.asr_profile == "" for e in p0.events)
    assert all("asr_profile" not in d and "asr_experimental" not in d for d in p0.to_dict()["events"])


@needs_ckpt
def test_fingerprint_is_reported_and_cache_key_is_a_documented_alias():
    m = _engine(WORDS, "p1_disfluent_prompt").analyze(_audio()).metadata["asr_profile"]
    fp = m["asr_fingerprint"]["profile_fingerprint"]
    assert fp and m["asr_cache"]["key"] == fp and m["asr_cache"]["engine_cache"] is None
    assert "deprecated" in m["asr_cache"] and "NO ASR cache" in m["asr_fingerprint"]["note"]


def test_model_info_failure_is_logged_and_exposed_not_swallowed(caplog, monkeypatch):
    import logging

    import mrcd.reproducibility as rep

    def boom(_):
        raise OSError("disk exploded")

    monkeypatch.setattr(rep, "file_sha256", boom)
    asr = VerbatimASR.__new__(VerbatimASR)
    asr.model_size, asr.compute_type, asr.device = str(Path(__file__).parent), "int8", "cpu"
    asr._model_bin_sha256 = asr._model_dir = None
    asr._model_info_done = False
    (Path(__file__).parent / "model.bin").exists()  # the tests dir has no model.bin: force the hash path below
    monkeypatch.setattr(Path, "exists", lambda self: True)
    with caplog.at_level(logging.WARNING):
        info = asr.model_info()
    assert info["model_bin_sha256"] is None and "disk exploded" in info["model_info_error"]
    assert any("model_info" in r.message for r in caplog.records)


def test_live_rows_are_marked_only_for_non_default_asr_profiles():
    from mrcd.live.session import LiveScorer
    from mrcd.live.streaming_asr import StreamingASR

    def mk(profile):
        sa = StreamingASR.__new__(StreamingASR)
        sa.asr = VerbatimASR.__new__(VerbatimASR)
        if profile is not None:
            sa.asr.profile = get_profile(profile)
        return sa

    assert mk("p0").asr_mark() is None
    assert mk("p1_disfluent_prompt").asr_mark() == {"asr_profile": "p1_disfluent_prompt", "asr_experimental": True}
    assert mk(None).asr_mark() == {"asr_profile": "unknown", "asr_experimental": None}
    sc = LiveScorer(lambda t: np.array([1.0]), ["a"], asr_mark=mk("p1_disfluent_prompt").asr_mark(), ling_dim=12)
    assert sc.asr_mark == {"asr_profile": "p1_disfluent_prompt", "asr_experimental": True}
    assert LiveScorer(lambda t: np.array([1.0]), ["a"], ling_dim=12).asr_mark is None


@needs_ckpt
def test_engine_fingerprint_changes_with_chunk_s_REVISION7_mutant_b():
    """Pins that the engine passes chunk_s to the fingerprint (mutation: dropping it left all tests green)."""
    eng = _engine(WORDS, "p1_disfluent_prompt")
    a = eng.analyze(_audio(), asr_chunk_s=8.0).metadata["asr_profile"]["asr_fingerprint"]["profile_fingerprint"]
    b = eng.analyze(_audio(), asr_chunk_s=4.0).metadata["asr_profile"]["asr_fingerprint"]["profile_fingerprint"]
    assert a != b
    assert a == eng.analyze(_audio(), asr_chunk_s=8.0).metadata["asr_profile"]["asr_fingerprint"]["profile_fingerprint"]
