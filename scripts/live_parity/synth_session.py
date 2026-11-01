"""Deterministic long-session audio builder shared by tests and live-parity scripts.

``build_session(minutes, seed)`` interleaves REAL speech (AMERESCO clips, when present locally) with
synthetic stretches that stress the live path: digital silence, white noise, mains hum, amplitude
ramps, short pauses, clicks and a clipped burst. Without the AMERESCO clips the whole session is
synthetic (``real_clips == 0`` in the returned info) and callers that need real speech must skip.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

SR = 16_000
REPO = Path(__file__).resolve().parents[2]
AMERESCO = REPO / "results" / "baselines_20261030" / "ameresco"


def real_clips() -> list[Path]:
    man = AMERESCO / "manifest.json"
    if not man.exists():
        return []
    paths = [Path(m["wav"]) for m in json.loads(man.read_text("utf8"))]
    return [p if p.is_absolute() else REPO / p for p in paths if (p if p.is_absolute() else REPO / p).exists()]


def _voice(rng, seconds, amp=0.2):
    t = np.arange(int(seconds * SR)) / SR
    f0 = 120 + 50 * np.sin(2 * np.pi * rng.uniform(0.2, 0.5) * t)
    ph = 2 * np.pi * np.cumsum(f0) / SR
    v = sum((0.6 / k) * np.sin(k * ph) for k in range(1, 6))
    return (amp * v).astype(np.float32)


def synthetic_stretch(kind: str, rng, seconds: float) -> np.ndarray:
    n = int(seconds * SR)
    if kind == "silence":
        return np.zeros(n, np.float32)
    if kind == "noise":
        return (0.02 * rng.standard_normal(n)).astype(np.float32)
    if kind == "hum":
        t = np.arange(n) / SR
        return (0.05 * np.sin(2 * np.pi * 60 * t) + 0.003 * rng.standard_normal(n)).astype(np.float32)
    if kind == "ramp":
        return (_voice(rng, seconds) * np.linspace(0.0, 1.0, n)).astype(np.float32)
    if kind == "voice":
        return _voice(rng, seconds)
    if kind == "clicks":
        x = np.zeros(n, np.float32)
        x[:: SR // 3] = 0.9
        return x
    if kind == "clipped":
        return np.clip(4.0 * _voice(rng, seconds), -1.0, 1.0).astype(np.float32)
    raise ValueError(kind)


KINDS = ("silence", "noise", "hum", "ramp", "voice", "clicks", "clipped")


def build_session(minutes: float, seed: int = 0, *, use_real: bool = True):
    """Return ``(audio float32, info)``; length >= ``minutes`` minutes."""
    import soundfile as sf

    rng = np.random.default_rng(seed)
    clips = real_clips() if use_real else []
    target = int(minutes * 60 * SR)
    parts, total, n_real, layout = [], 0, 0, []
    i = 0
    while total < target:
        if clips and i % 3 != 2:
            p = clips[int(rng.integers(len(clips)))]
            a, _ = sf.read(str(p), dtype="float32")
            parts.append(a)
            n_real += 1
            layout.append(("real", p.name, len(a) / SR))
        else:
            kind = KINDS[int(rng.integers(len(KINDS)))]
            secs = float(rng.uniform(0.3, 4.0)) if kind != "silence" else float(rng.uniform(0.5, 30.0))
            parts.append(synthetic_stretch(kind, rng, secs))
            layout.append((kind, "", secs))
        total += len(parts[-1])
        i += 1
    audio = np.concatenate(parts).astype(np.float32)
    return audio, {"real_clips": n_real, "n_parts": len(parts), "duration_s": len(audio) / SR, "layout": layout}
