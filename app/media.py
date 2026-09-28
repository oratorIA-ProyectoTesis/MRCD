"""Media ingestion: the original stays untouched; a 16 kHz mono PCM copy is derived
with a recorded command. No denoising, loudness normalization or silence trimming."""

from __future__ import annotations

import hashlib
import io
import json
import subprocess
import wave
from pathlib import Path

import numpy as np
import soundfile as sf

SR = 16_000
PEAK_LEVELS = (10, 100)  # peaks per second, coarse overview and 10 ms detail
DERIVE_ARGS = ["-vn", "-ac", "1", "-ar", str(SR), "-c:a", "pcm_s16le"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def probe(path: Path) -> dict:
    """Detect container/codec from content, not from the file extension."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)],
        capture_output=True,
        text=True,
    )
    info = json.loads(out.stdout or "{}")
    audio = [s for s in info.get("streams", []) if s.get("codec_type") == "audio"]
    if out.returncode or not audio:
        raise ValueError("El archivo no contiene una pista de audio decodificable")
    a = audio[0]
    return {
        "format": info["format"].get("format_name"),
        "codec": a.get("codec_name"),
        "channels": a.get("channels"),
        "sample_rate": int(a.get("sample_rate", 0)),
        "duration_s": float(info["format"].get("duration") or a.get("duration") or 0),
        "has_video": any(s.get("codec_type") == "video" for s in info["streams"]),
    }


def derive(original: Path, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    source = probe(original)
    wav = out_dir / "analysis.wav"
    cmd = ["ffmpeg", "-nostdin", "-y", "-v", "error", "-i", "<original>", *DERIVE_ARGS, "<analysis.wav>"]
    run = subprocess.run(
        [str(original) if a == "<original>" else str(wav) if a == "<analysis.wav>" else a for a in cmd],
        capture_output=True,
        text=True,
    )
    if run.returncode:
        raise ValueError(f"ffmpeg falló: {run.stderr.strip()[-300:]}")
    version = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True).stdout.split("\n")[0]
    audio = load(wav)
    for pps in PEAK_LEVELS:
        np.save(out_dir / f"peaks_{pps}.npy", peaks(audio, pps))
    return {
        "original": source,
        "analysis": {
            "sha256": sha256(wav),
            "samples": len(audio),
            "sample_rate": SR,
            "channels": 1,
            "duration_ms": round(len(audio) * 1000 / SR),
        },
        "transform": {"command": cmd, "ffmpeg": version},
    }


def load(path: Path, start_ms: int = 0, end_ms: int | None = None) -> np.ndarray:
    a = round(start_ms * SR / 1000)
    b = None if end_ms is None else round(end_ms * SR / 1000)
    audio, sr = sf.read(str(path), start=a, stop=b, dtype="float32", always_2d=True)
    if sr != SR:
        raise ValueError(f"se esperaba {SR} Hz y el derivado tiene {sr}")
    return audio.mean(1)


def peaks(audio: np.ndarray, pps: int) -> np.ndarray:
    hop = SR // pps
    n = int(np.ceil(len(audio) / hop))
    padded = np.zeros(n * hop, np.float32)
    padded[: len(audio)] = np.abs(audio)
    return padded.reshape(n, hop).max(1).astype(np.float16)


def clip_wav(path: Path, start_ms: int, end_ms: int) -> bytes:
    pcm = (np.clip(load(path, start_ms, end_ms), -1, 1) * 32767).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


def segments(duration_ms: int, core_ms: int = 30_000, context_ms: int = 5_000) -> list[dict]:
    """Analysis regions: a central interval that owns events plus context on each side."""
    return [
        {
            "id": f"s{i:03d}",
            "core_start_ms": a,
            "core_end_ms": min(a + core_ms, duration_ms),
            "audio_start_ms": max(0, a - context_ms),
            "audio_end_ms": min(duration_ms, a + core_ms + context_ms),
        }
        for i, a in enumerate(range(0, duration_ms, core_ms))
    ]


def owner(segs: list[dict], start_ms: int, end_ms: int) -> str | None:
    mid = (start_ms + end_ms) / 2
    return next((s["id"] for s in segs if s["core_start_ms"] <= mid < s["core_end_ms"]), None)
