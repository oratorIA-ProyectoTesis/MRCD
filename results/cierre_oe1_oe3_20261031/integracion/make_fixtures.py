"""Generate the contract fixtures in ../fixtures by running the REAL service code (no hand-written JSON).

Usage (repo root, venv):
    OMP_NUM_THREADS=4 .venv/Scripts/python.exe results/cierre_oe1_oe3_20261031/integracion/make_fixtures.py

* response_ok_*.json come from `python -m mrcd.service analyze ...` (real engine; see logs/fixture_commands.txt).
* error_*.json are produced here by feeding deliberately bad bytes to `MrcdAnalyzer.analyze_bytes`; the engine is
  never reached for these (they fail in the input checks), so no model is loaded.
* request_example.txt documents the HTTP request that produced job_*.json.
"""

from __future__ import annotations

import io
import json
import struct
import sys
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from mrcd.service import AnalysisOptions, MrcdAnalyzer, validate_document  # noqa: E402

OUT = Path(__file__).resolve().parent / "fixtures"
OUT.mkdir(exist_ok=True)


def wav_bytes(
    samples: np.ndarray, rate: int = 16000, channels: int = 1, width: int = 2
) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(width)
        w.setframerate(rate)
        w.writeframes((np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes())
    return buf.getvalue()


def float_wav(seconds: float, rate: int = 16000) -> bytes:
    """WAV with IEEE-float samples (format tag 3): Python's `wave` cannot read it."""
    n = int(seconds * rate)
    data = np.zeros(n, "<f4").tobytes()
    fmt = struct.pack("<HHIIHH", 3, 1, rate, rate * 4, 4, 32)
    body = (
        b"WAVE"
        + b"fmt "
        + struct.pack("<I", len(fmt))
        + fmt
        + b"data"
        + struct.pack("<I", len(data))
        + data
    )
    return b"RIFF" + struct.pack("<I", len(body)) + body


def main() -> None:
    tone = 0.1 * np.sin(2 * np.pi * 180 * np.arange(16000 * 20) / 16000)
    cases = {
        "error_not_a_wav": (b"this is not audio", AnalysisOptions()),
        "error_empty_body": (b"", AnalysisOptions()),
        "error_stereo": (wav_bytes(np.repeat(tone, 2), channels=2), AnalysisOptions()),
        "error_sample_rate_44100": (wav_bytes(tone, rate=44100), AnalysisOptions()),
        "error_float_wav": (float_wav(20.0), AnalysisOptions()),
        "error_audio_too_short": (wav_bytes(tone[: 16000 * 2]), AnalysisOptions()),
        "error_unknown_asr_profile": (
            wav_bytes(tone),
            AnalysisOptions(asr_profile="p9"),
        ),
    }
    analyzer = MrcdAnalyzer(max_duration_s=600)
    for name, (data, options) in cases.items():
        doc = analyzer.analyze_bytes(data, options)
        assert validate_document(doc) == [], name
        (OUT / f"{name}.json").write_text(
            json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(name, "->", doc["error"]["code"])
    # not finite: library entry point (a PCM16 WAV cannot carry NaN)
    bad = tone.astype(np.float32).copy()
    bad[100] = np.nan
    doc = analyzer.analyze_array(bad, 16000)
    assert validate_document(doc) == []
    (OUT / "error_audio_not_finite.json").write_text(
        json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print("error_audio_not_finite ->", doc["error"]["code"])
    # checkpoint unavailable: real analyzer pointed at a missing file
    missing = MrcdAnalyzer(ROOT / "models" / "__does_not_exist__.pt")
    doc = missing.analyze_bytes(wav_bytes(tone), AnalysisOptions())
    assert validate_document(doc) == []
    (OUT / "error_checkpoint_unavailable.json").write_text(
        json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print("error_checkpoint_unavailable ->", doc["error"]["code"])


if __name__ == "__main__":
    main()
