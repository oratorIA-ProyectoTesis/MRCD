"""MrcdAnalyzer: structured errors, defaults left intact, isolation from the app, no audio on disk."""

from __future__ import annotations

import io
import os
import struct
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np
import pytest

from mrcd.engine import AnalysisResult, DisfluencyEngine, Event
from mrcd.service import AnalysisOptions, MrcdAnalyzer, validate_document

REPO = Path(__file__).resolve().parents[1]


def wav(seconds=20.0, rate=16000, channels=1, width=2):
    n = int(seconds * rate)
    t = np.arange(n) / rate
    x = (0.1 * np.sin(2 * np.pi * 180 * t) * 32767).astype("<i2")
    if channels == 2:
        x = np.repeat(x, 2)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(width)
        w.setframerate(rate)
        w.writeframes(x.tobytes())
    return buf.getvalue()


class FakeEngine:
    mode = "audio_text"
    labels = ["fluent", "filler_word"]
    warning = "weak supervision"
    checkpoint_name = "fake.pt"
    checkpoint_sha256 = "d" * 64

    def __init__(self):
        self.calls = []

    def analyze(self, audio, **kw):
        self.calls.append((len(audio), kw))
        return AnalysisResult(
            events=[
                Event(
                    "filler_word",
                    1.0,
                    2.0,
                    0.8,
                    2,
                    modalities=("acoustic", "linguistic"),
                )
            ],
            transcript="eh",
            duration=len(audio) / 16000,
            mode="audio_text",
            warning=self.warning,
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
        )


def analyzer(**kw):
    seen = []

    def factory(options):
        seen.append(options)
        return FakeEngine()

    a = MrcdAnalyzer("unused.pt", engine_factory=factory, **kw)
    a.seen = seen
    return a


def test_defaults_are_p0_and_v0_and_the_engine_is_called_the_normal_way():
    a = analyzer()
    doc = a.analyze_bytes(wav())
    assert validate_document(doc) == [] and doc["status"] == "completed"
    assert a.seen == [AnalysisOptions(asr_profile="p0", head=None, include_text=False)]
    ((engine, _info),) = a._engines.values()
    # only sample_rate is passed: asr_chunk_s / terminal_window / span keep the engine defaults
    assert engine.calls == [(20 * 16000, {"sample_rate": 16000})]
    assert doc["experimental"]["active"] is False and doc["request"]["head"] is None


def test_engine_load_defaults_were_not_changed_by_the_service():
    import inspect

    sig = inspect.signature(DisfluencyEngine.load)
    assert (
        sig.parameters["asr_profile"].default == "p0"
        and sig.parameters["head"].default is None
    )
    sig = inspect.signature(DisfluencyEngine.analyze)
    assert (
        sig.parameters["terminal_window"].default is False
        and sig.parameters["asr_chunk_s"].default == 8.0
    )


def test_default_factory_loads_p0_without_head(monkeypatch, tmp_path):
    ckpt = tmp_path / "c.pt"
    ckpt.write_bytes(b"x")
    recorded = {}

    def fake_load(cls, checkpoint, **kw):
        recorded.update(kw)
        return FakeEngine()

    monkeypatch.setattr(DisfluencyEngine, "load", classmethod(fake_load))
    MrcdAnalyzer(ckpt).analyze_bytes(wav())
    assert recorded["asr_profile"] == "p0" and recorded["head"] is None


def test_engines_are_cached_per_profile_and_head():
    a = analyzer()
    a.analyze_bytes(wav())
    a.analyze_bytes(wav())
    a.analyze_bytes(wav(), AnalysisOptions(asr_profile="p1_disfluent_prompt"))
    assert [o.asr_profile for o in a.seen] == ["p0", "p1_disfluent_prompt"]


def float_wav(seconds=20.0, rate=16000):
    data = np.zeros(int(seconds * rate), "<f4").tobytes()
    fmt = struct.pack("<HHIIHH", 3, 1, rate, rate * 4, 4, 32)
    body = (
        b"WAVEfmt "
        + struct.pack("<I", len(fmt))
        + fmt
        + b"data"
        + struct.pack("<I", len(data))
        + data
    )
    return b"RIFF" + struct.pack("<I", len(body)) + body


CASES = {
    "empty": (lambda: b"", "invalid_audio_file"),
    "garbage": (lambda: b"not audio at all", "invalid_audio_file"),
    "stereo": (lambda: wav(channels=2), "audio_not_mono"),
    "rate_44100": (lambda: wav(rate=44100), "unsupported_sample_rate"),
    "float_wav": (lambda: float_wav(), "unsupported_audio_format"),
    "too_short": (lambda: wav(seconds=2), "audio_too_short"),
}


@pytest.mark.parametrize("case", sorted(CASES))
def test_input_problems_return_structured_errors_without_loading_the_engine(case):
    make, code = CASES[case]
    a = analyzer()
    doc = a.analyze_bytes(make())
    assert validate_document(doc) == []
    assert doc["status"] == "failed" and doc["error"]["code"] == code
    assert doc["events"] == [] and a.seen == []  # the engine was never touched
    assert "retryable" in doc["error"]


def test_error_documents_do_not_echo_audio_content_or_file_names():
    doc = analyzer().analyze_bytes(b"secret-looking payload 12345")
    assert "secret" not in str(doc)
    assert doc["request"]["audio_duration_s"]["available"] is False  # unknown, never 0


def test_too_long_audio_is_rejected():
    doc = analyzer(max_duration_s=10).analyze_bytes(wav(seconds=20))
    assert doc["error"]["code"] == "audio_too_long"


def test_non_finite_array_is_rejected():
    arr = np.zeros(16000 * 20, np.float32)
    arr[5] = np.inf
    doc = analyzer().analyze_array(arr, 16000)
    assert doc["error"]["code"] == "audio_not_finite" and validate_document(doc) == []


def test_unknown_profile_and_head_are_invalid_requests():
    a = analyzer()
    assert (
        a.analyze_bytes(wav(), AnalysisOptions(asr_profile="p9"))["error"]["code"]
        == "invalid_request"
    )
    assert (
        a.analyze_bytes(wav(), AnalysisOptions(head="zz"))["error"]["code"]
        == "invalid_request"
    )
    assert a.seen == []


def test_missing_checkpoint_is_reported_without_a_path():
    doc = MrcdAnalyzer(Path("C:/Users/x/definitely/missing.pt")).analyze_bytes(wav())
    assert doc["error"]["code"] == "checkpoint_unavailable"
    assert doc["error"]["details"] == {"checkpoint_file": "missing.pt"}
    assert "Users" not in str(doc)


def _raise_in(filename: str, exc: Exception):
    code = compile("def boom(e):\n    raise e\n", filename, "exec")
    ns: dict = {}
    exec(code, ns)  # noqa: S102 - synthetic frame to emulate a failure inside a library
    ns["boom"](exc)


def test_asr_load_failure_is_classified_as_asr_unavailable():
    def factory(options):
        _raise_in(
            "site-packages/faster_whisper/utils.py", OSError("model not in cache")
        )

    doc = MrcdAnalyzer("u.pt", engine_factory=factory).analyze_bytes(wav())
    assert doc["error"]["code"] == "asr_unavailable"
    assert doc["error"]["details"] == {"exception_type": "OSError"}
    assert "model not in cache" not in str(doc)


def test_head_load_failure_is_classified_as_head_unavailable():
    def factory(options):
        _raise_in("mrcd/adaptation.py", FileNotFoundError("nope"))

    doc = MrcdAnalyzer("u.pt", engine_factory=factory).analyze_bytes(
        wav(), AnalysisOptions(head="vc")
    )
    assert doc["error"]["code"] == "head_unavailable"


def test_unknown_load_failure_and_analysis_failure_are_structured():
    def bad_factory(options):
        raise RuntimeError("boom")

    assert (
        MrcdAnalyzer("u.pt", engine_factory=bad_factory).analyze_bytes(wav())["error"][
            "code"
        ]
        == "engine_load_failed"
    )

    class Exploding(FakeEngine):
        def analyze(self, audio, **kw):
            raise RuntimeError("secret detail")

    doc = MrcdAnalyzer("u.pt", engine_factory=lambda o: Exploding()).analyze_bytes(
        wav()
    )
    assert (
        doc["error"]["code"] == "analysis_failed" and doc["error"]["retryable"] is True
    )
    assert "secret" not in str(doc) and validate_document(doc) == []


def test_nothing_is_written_to_disk(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    a = analyzer()
    a.analyze_bytes(wav())
    a.analyze_bytes(b"bad")
    assert list(tmp_path.iterdir()) == []


def test_service_does_not_import_the_app_and_the_engine_does_not_import_the_service():
    code = (
        "import sys, mrcd.service, mrcd.service.server, mrcd.service.cli;"
        "bad=[m for m in sys.modules if m.split('.')[0].lower() in ('oratoria','scripts')];"
        "print(bad)"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO,
        capture_output=True,
        text=True,
        env={**os.environ, "OMP_NUM_THREADS": "4"},
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "[]"
    forbidden = (
        "ORATORIA-PROYECTO",
        "ORATORIA-ENGINE",
        "from oratoria",
        "import oratoria",
        "from scripts",
    )
    for p in (REPO / "mrcd" / "service").glob("*.py"):
        text = p.read_text(encoding="utf-8")
        assert not any(f in text for f in forbidden), (p, forbidden)
    for p in (REPO / "mrcd").glob("*.py"):  # engine core never depends on the service
        assert "mrcd.service" not in p.read_text(encoding="utf-8"), p


@pytest.mark.skipif(
    os.environ.get("MRCD_SLOW") != "1" or os.environ.get("OMP_NUM_THREADS") != "4",
    reason="set MRCD_SLOW=1 and OMP_NUM_THREADS=4 before python starts; needs models/ckpt_audio_text.pt and the cached Whisper small",
)
def test_real_engine_service_events_equal_the_direct_engine_path(tmp_path):
    clip = REPO / "data" / "annotation_batch_01" / "DNrrpE-sYnU_s00.wav"
    ckpt = REPO / "models" / "ckpt_audio_text.pt"
    if not (clip.exists() and ckpt.exists()):
        pytest.skip("clip or checkpoint not present")
    engine = DisfluencyEngine.load(ckpt)
    direct = engine.analyze_file(clip)
    a = MrcdAnalyzer(ckpt, engine_factory=lambda o: engine)
    doc = a.analyze_file(clip)
    assert validate_document(doc) == [] and doc["status"] == "completed"
    got = [
        (e["class"], e["start_s"], e["end_s"], e["score"]["value"])
        for e in doc["events"]
    ]
    want = [
        (e.label, round(e.start, 3), round(e.end, 3), round(e.confidence, 6))
        for e in direct.events
    ]
    assert got == want
