"""Decode an uploaded WAV into the mono 16 kHz float32 array the engine takes, with structured errors.

Nothing is written to disk: the bytes are decoded in memory. The service never downmixes or resamples
silently (the engine itself refuses that, see ``DisfluencyEngine.analyze``); a stereo or 44.1 kHz file is
reported as an error so the caller fixes its pipeline.
"""

from __future__ import annotations

import hashlib
import io
import wave
from dataclasses import dataclass

import numpy as np

from mrcd.constants import SAMPLE_RATE


class ServiceError(Exception):
    """A failure with a stable contract code. ``details`` must contain no audio content or personal data."""

    def __init__(self, code: str, message: str, details: dict | None = None):
        super().__init__(message)
        self.code, self.message, self.details = code, message, details or {}


@dataclass(frozen=True)
class DecodedAudio:
    samples: np.ndarray  # mono float32 in [-1, 1]
    sample_rate: int
    channels: int
    duration_s: float
    sha256: str  # of the raw uploaded bytes


def audio_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def inspect_header(data: bytes) -> dict:
    """Best-effort header facts for error reports (``{}`` when the bytes are not a readable WAV)."""
    try:
        with wave.open(io.BytesIO(data), "rb") as w:
            return {
                "channels": w.getnchannels(),
                "sample_rate_hz": w.getframerate(),
                "duration_s": w.getnframes() / w.getframerate()
                if w.getframerate()
                else None,
            }
    except (wave.Error, EOFError):
        return {}


def check_array(samples: np.ndarray, sample_rate: int) -> np.ndarray:
    """Validate an in-memory array (library entry point) and return it as float32."""
    arr = np.asarray(samples)
    if arr.ndim != 1:
        raise ServiceError(
            "audio_not_mono",
            "A one-dimensional mono array is required.",
            {"ndim": int(arr.ndim)},
        )
    arr = arr.astype(np.float32, copy=False)
    if not np.isfinite(arr).all():
        raise ServiceError(
            "audio_not_finite", "The audio contains NaN or infinite samples."
        )
    if sample_rate != SAMPLE_RATE:
        raise ServiceError(
            "unsupported_sample_rate",
            f"The engine takes {SAMPLE_RATE} Hz and received {sample_rate} Hz; resample before sending.",
            {"expected_hz": SAMPLE_RATE, "received_hz": int(sample_rate)},
        )
    return arr


def decode_wav(data: bytes) -> DecodedAudio:
    if not data:
        raise ServiceError("invalid_audio_file", "The request body is empty.")
    digest = audio_sha256(data)
    try:
        with wave.open(io.BytesIO(data), "rb") as w:
            channels, width, rate = w.getnchannels(), w.getsampwidth(), w.getframerate()
            comptype, frames = w.getcomptype(), w.getnframes()
            if channels != 1:
                raise ServiceError(
                    "audio_not_mono",
                    f"Mono audio is required; the file has {channels} channels. "
                    "Downmix before sending (the service never mixes channels silently).",
                    {"channels": channels},
                )
            if width != 2 or comptype != "NONE":
                raise ServiceError(
                    "unsupported_audio_format",
                    "Only 16-bit PCM WAV is supported.",
                    {"sample_width_bytes": width, "compression": comptype},
                )
            if rate != SAMPLE_RATE:
                raise ServiceError(
                    "unsupported_sample_rate",
                    f"The engine takes {SAMPLE_RATE} Hz and the file is {rate} Hz; "
                    "resample before sending.",
                    {"expected_hz": SAMPLE_RATE, "received_hz": rate},
                )
            pcm = np.frombuffer(w.readframes(frames), dtype="<i2")
    except ServiceError:
        raise
    except (wave.Error, EOFError) as exc:
        # `wave` reports non-PCM WAV (e.g. float, format tag 3) as "unknown format"
        text = str(exc)
        if "unknown format" in text:
            raise ServiceError(
                "unsupported_audio_format",
                "Only 16-bit PCM WAV is supported.",
                {"reason": text},
            ) from exc
        raise ServiceError(
            "invalid_audio_file", "The body is not a readable WAV file."
        ) from exc
    samples = pcm.astype(np.float32) / 32768.0
    return DecodedAudio(samples, rate, channels, len(samples) / rate, digest)
