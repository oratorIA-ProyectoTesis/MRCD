"""Extractor acústico (DSP): Silero VAD, F0 (pYIN) y envolvente RMS.

Salida a 100 fps (hop de 10 ms), alineada al inicio del archivo (t = 0).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from core.constants import ACOUSTIC_HOP_S, SAMPLE_RATE


@dataclass
class AcousticTrack:
    times: np.ndarray          # (T,) s
    rms: np.ndarray            # (T,) amplitud RMS lineal
    f0: np.ndarray             # (T,) Hz, NaN si no sonoro
    voiced_prob: np.ndarray    # (T,) probabilidad de sonoridad pYIN
    vad: np.ndarray            # (T,) 0/1 habla
    speech_segments: list[tuple[float, float]] = field(default_factory=list)
    silence_segments: list[tuple[float, float]] = field(default_factory=list)
    vad_backend: str = "silero"
    hf_ratio: np.ndarray | None = None   # (T,) energía > 3 kHz / energía total (fricativas)

    def frame_matrix(self) -> np.ndarray:
        """Matriz (T, 5) para el modelo: [log_rms, f0_st, voiced_prob, vad, df0]."""
        log_rms = np.log10(self.rms + 1e-6)
        f0 = np.where(np.isnan(self.f0), 0.0, self.f0)
        # semitonos relativos a 100 Hz (0 si no sonoro)
        f0_st = np.where(f0 > 0, 12 * np.log2(np.maximum(f0, 1e-3) / 100.0), 0.0)
        df0 = np.concatenate([[0.0], np.diff(f0_st)])
        df0[f0 == 0] = 0.0
        return np.stack([log_rms, f0_st, self.voiced_prob, self.vad.astype(float), df0], 1).astype(np.float32)


# ------------------------------------------------------------------ VAD
def _silero_vad(audio: np.ndarray, sr: int) -> list[tuple[float, float]]:
    from silero_vad import get_speech_timestamps, load_silero_vad  # type: ignore
    import torch  # noqa: F401  (silero-vad usa torch o onnx)

    model = load_silero_vad(onnx=True)
    ts = get_speech_timestamps(
        audio.astype(np.float32), model, sampling_rate=sr,
        min_silence_duration_ms=150, min_speech_duration_ms=120, speech_pad_ms=30,
        return_seconds=True,
    )
    return [(float(t["start"]), float(t["end"])) for t in ts]


def _energy_vad(rms: np.ndarray, times: np.ndarray) -> list[tuple[float, float]]:
    """Fallback: umbral adaptativo sobre RMS (solo si Silero no está disponible)."""
    db = 20 * np.log10(rms + 1e-8)
    thr = max(np.percentile(db, 20) + 12, db.max() - 45)
    speech = db > thr
    return _mask_to_segments(speech, times, min_len=0.12, merge_gap=0.15)


def _mask_to_segments(mask, times, min_len=0.0, merge_gap=0.0):
    segs, start = [], None
    hop = times[1] - times[0] if len(times) > 1 else ACOUSTIC_HOP_S
    for i, m in enumerate(mask):
        if m and start is None:
            start = times[i]
        elif not m and start is not None:
            segs.append([start, times[i]]); start = None
    if start is not None:
        segs.append([start, times[-1] + hop])
    merged: list[list[float]] = []
    for s in segs:
        if merged and s[0] - merged[-1][1] <= merge_gap:
            merged[-1][1] = s[1]
        else:
            merged.append(s)
    return [(a, b) for a, b in merged if b - a >= min_len]


def _complement(segs, total):
    out, cur = [], 0.0
    for a, b in segs:
        if a > cur:
            out.append((cur, a))
        cur = max(cur, b)
    if cur < total:
        out.append((cur, total))
    return out


# ------------------------------------------------------------------ main
def extract_acoustic(audio: np.ndarray, sr: int = SAMPLE_RATE, use_silero: bool = True,
                     fmin: float = 60.0, fmax: float = 400.0) -> AcousticTrack:
    import librosa

    if audio.dtype != np.float32:
        audio = audio.astype(np.float32)
    if np.abs(audio).max() > 1.5:        # int16 -> float
        audio = audio / 32768.0
    hop = int(sr * ACOUSTIC_HOP_S)
    frame = 4 * hop                       # 40 ms
    rms = librosa.feature.rms(y=audio, frame_length=frame, hop_length=hop, center=True)[0]
    f0, _, vprob = librosa.pyin(audio, fmin=fmin, fmax=fmax, sr=sr,
                                frame_length=1024, hop_length=hop, center=True)
    S = np.abs(librosa.stft(audio, n_fft=512, hop_length=hop, center=True)) ** 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=512)
    hf = S[freqs >= 3000].sum(0) / (S.sum(0) + 1e-10)
    n = min(len(rms), len(f0), len(hf))
    rms, f0, vprob, hf = rms[:n], f0[:n], np.nan_to_num(vprob[:n]), hf[:n]
    times = np.arange(n) * ACOUSTIC_HOP_S
    total = len(audio) / sr

    backend = "silero"
    speech = None
    if use_silero:
        try:
            speech = _silero_vad(audio, sr)
        except Exception as exc:  # pragma: no cover - depende del entorno
            print(f"[acoustic] Silero VAD no disponible ({exc}); usando VAD por energía")
    if speech is None:
        speech, backend = _energy_vad(rms, times), "energy"

    vad = np.zeros(n, dtype=np.int8)
    for a, b in speech:
        vad[(times >= a) & (times < b)] = 1
    return AcousticTrack(times, rms.astype(np.float32), f0.astype(np.float32),
                         vprob.astype(np.float32), vad, speech,
                         _complement(speech, total), backend, hf.astype(np.float32))


def load_wav(path: str) -> tuple[np.ndarray, int]:
    import soundfile as sf

    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(1)
    if sr != SAMPLE_RATE:
        import librosa
        audio, sr = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE), SAMPLE_RATE
    return audio, sr
