"""Acoustic features of an ASR word interval from 8 kHz mu-law telephone audio."""

from __future__ import annotations

import audioop
from pathlib import Path

import numpy as np

SR = 8000
HOP = 80  # 10 ms


def load_ulaw(path: str | Path) -> np.ndarray:
    raw = Path(path).read_bytes()
    pcm = audioop.ulaw2lin(raw, 2)
    return np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0


def turn_tracks(x: np.ndarray) -> dict:
    """Frame RMS (dB) and F0 (Hz, NaN unvoiced) at 10 ms."""
    import librosa

    n = len(x) // HOP
    if n < 4:
        return {"rms_db": np.full(1, -80.0), "f0": np.full(1, np.nan), "median_f0": np.nan}
    frames = x[: n * HOP].reshape(n, HOP)
    rms_db = 20 * np.log10(np.sqrt((frames**2).mean(1)) + 1e-5)
    f0, voiced, _ = librosa.pyin(x, fmin=70, fmax=400, sr=SR, frame_length=640, hop_length=HOP)
    f0 = np.where(voiced, f0, np.nan)[:n]
    if len(f0) < n:
        f0 = np.pad(f0, (0, n - len(f0)), constant_values=np.nan)
    med = float(np.nanmedian(f0)) if np.isfinite(f0).any() else float("nan")
    return {"rms_db": rms_db, "f0": f0, "median_f0": med}


def word_acoustics(
    tr: dict,
    start: float,
    end: float,
    n_chars: int,
    prob: float,
    prev_end: float | None,
    next_start: float | None,
) -> dict[str, float]:
    dur = max(end - start, 0.0)
    a, b = int(start * 100), max(int(end * 100), int(start * 100) + 1)
    rms = tr["rms_db"][a:b]
    med_rms = float(np.median(tr["rms_db"]))
    f0 = tr["f0"][a:b]
    ok = np.isfinite(f0).sum() >= 3 and np.isfinite(tr["median_f0"])
    if ok:
        st = 12 * np.log2(f0[np.isfinite(f0)] / tr["median_f0"])
        f0m = float(st.mean())
        f0s = float(np.polyfit(np.arange(len(st)), st, 1)[0] * 100) if len(st) > 2 else 0.0
    else:
        f0m = f0s = 0.0
    return {
        "a_dur": dur,
        "a_dur_pc": dur / max(n_chars, 1),
        "a_gap_b": float(start - prev_end) if prev_end is not None else 0.0,
        "a_gap_a": float(next_start - end) if next_start is not None else 0.0,
        "a_prob": float(prob),
        "a_rms": float(rms.mean() - med_rms) if len(rms) else 0.0,
        "a_f0m": f0m,
        "a_f0s": f0s,
        "a_f0_ok": float(ok),
    }
