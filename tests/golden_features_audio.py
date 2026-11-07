"""Synthetic, licence-safe clip + words for the audio->features golden (no corpus audio is ever committed)."""

from __future__ import annotations

import numpy as np

SR = 16000


def synthetic_clip() -> np.ndarray:
    """2.4 s of a gliding voiced tone with a 0.5 s gap and light noise; fully determined by the seed."""
    rng = np.random.default_rng(20261107)
    t = np.arange(int(2.4 * SR)) / SR
    f0 = 120.0 + 25.0 * np.sin(2 * np.pi * 0.7 * t)
    phase = 2 * np.pi * np.cumsum(f0) / SR
    voice = sum(np.sin(k * phase) / k for k in range(1, 6))
    env = 0.5 * (1 + np.sin(2 * np.pi * 3.0 * t - np.pi / 2))
    x = 0.25 * voice * env + 0.003 * rng.standard_normal(len(t))
    x[int(1.0 * SR) : int(1.5 * SR)] = 0.003 * rng.standard_normal(int(0.5 * SR))
    return x.astype(np.float32)


def _w(text, start, end, punct=""):
    return {"text": text, "norm": text, "start": start, "end": end, "prob": 0.9, "punct_after": punct}


WORDS = [_w("eh", 0.10, 0.40, ","), _w("hola", 0.45, 0.95, "."), _w("este", 1.55, 1.90), _w("proyecto", 1.95, 2.30, ".")]
