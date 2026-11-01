"""Explicit handling of short units and window planning (new module, 2026-10-30).

Why: 62 % of DIHANA user turns last < 3 s. The pilot recipe zero-pads them at the END to one 3 s
window. Digital zeros are not a realistic silence (log-RMS hits its floor, Silero sees a perfect gap),
so the padded tail can itself look like a pause to the model. This module makes the choice explicit:

* ``pad_mode="zero"``   : V0 behaviour (kept for comparison).
* ``pad_mode="noise"``  : pad with Gaussian noise at the unit's own noise floor (10th percentile of the
                          25 ms frame RMS), deterministic per ``seed_key``.
* ``abstain_below_s``   : a unit shorter than this is not scored at all (decision layer returns
                          "uncertain"); no features are changed.

Nothing here reads labels.
"""

from __future__ import annotations

import hashlib

import numpy as np

SR = 16000
WIN_S = 3.0
STRIDE_S = 0.5
PAD_MODES = ("zero", "noise")


def noise_floor_rms(
    audio: np.ndarray, sr: int = SR, frame_s: float = 0.025, pct: float = 10.0
) -> float:
    """Noise floor = ``pct`` percentile of the frame RMS (never below 1e-5 so the pad is not digital zero)."""
    n = int(round(frame_s * sr))
    if len(audio) < n:
        return 1e-5
    m = len(audio) // n
    fr = audio[: m * n].reshape(m, n).astype(np.float64)
    rms = np.sqrt((fr**2).mean(1))
    return float(max(np.percentile(rms, pct), 1e-5))


def _rng(seed_key: str) -> np.random.Generator:
    return np.random.default_rng(
        int(hashlib.sha256(seed_key.encode()).hexdigest()[:8], 16)
    )


def pad_audio(
    audio: np.ndarray, n_extra: int, mode: str = "zero", seed_key: str = ""
) -> np.ndarray:
    """Append ``n_extra`` samples to ``audio`` following ``mode``. ``n_extra <= 0`` returns the input unchanged."""
    if mode not in PAD_MODES:
        raise ValueError(f"pad mode must be one of {PAD_MODES}")
    if n_extra <= 0:
        return audio
    if mode == "zero":
        tail = np.zeros(n_extra, np.float32)
    else:
        tail = (
            _rng(seed_key).standard_normal(n_extra) * noise_floor_rms(audio)
        ).astype(np.float32)
    return np.concatenate([audio.astype(np.float32), tail])


def plan_unit(
    n_samples: int,
    a: float,
    b: float,
    *,
    win_s: float = WIN_S,
    stride_s: float = STRIDE_S,
    sr: int = SR,
):
    """Window plan of the pilot recipe. Returns ``(starts, b_eff, n_extra, padded)``.

    Spans shorter than one window are extended to ``a + win_s`` (``padded=True``) and need ``n_extra``
    samples appended to the audio; otherwise windows tile ``[a, b]`` every ``stride_s`` plus a terminal
    window at ``b - win_s`` when the grid does not end there.
    """
    padded = False
    n_extra = 0
    if b - a < win_s - 1e-8:
        n_extra = max(int(round((a + win_s) * sr)) - n_samples, 0)
        b = a + win_s
        padded = True
    starts = np.arange(a, b - win_s + 1e-9, stride_s)
    if b - win_s - starts[-1] > 1e-6:
        starts = np.append(starts, b - win_s)
    return starts, b, n_extra, padded


def should_abstain(duration_s: float, abstain_below_s: float | None) -> bool:
    """True when the unit is too short to be scored (``None`` disables abstention)."""
    return abstain_below_s is not None and duration_s < abstain_below_s


def internal_pause_stats(
    vad: np.ndarray, hop_s: float = 0.01, n_valid: int | None = None
) -> dict:
    """Pause descriptors from a 0/1 VAD track over the ORIGINAL (unpadded) frames.

    ``max_internal_pause_s``: longest silent run between two speech frames. ``edge_silence_s``: silence
    before the first / after the last speech frame. ``speech_frac``: speech frames / valid frames.
    """
    v = np.asarray(vad)[: n_valid if n_valid is not None else len(vad)] > 0
    n = len(v)
    if n == 0 or not v.any():
        return {
            "max_internal_pause_s": 0.0,
            "edge_silence_s": n * hop_s,
            "speech_frac": 0.0,
            "n_frames": n,
        }
    first, last = int(np.argmax(v)), int(n - 1 - np.argmax(v[::-1]))
    inner = ~v[first : last + 1]
    best = cur = 0
    for x in inner:
        cur = cur + 1 if x else 0
        best = max(best, cur)
    return {
        "max_internal_pause_s": best * hop_s,
        "edge_silence_s": (first + (n - 1 - last)) * hop_s,
        "speech_frac": float(v.mean()),
        "n_frames": n,
    }
