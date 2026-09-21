"""Buffer de sincronización audio + cinésica por timestamp de CAPTURA.

Principio: el orden de llegada por red es irrelevante; cada muestra/frame se
indexa por su timestamp de captura (reloj del cliente). Una ventana solo se
emite cuando el reloj de audio superó su fin + `jitter_ms`.

Entregables por ventana:
  audio     (48 000,) float32  -> 3.0 s @ 16 kHz
  kinesic   (30, 12)  float32  -> rejilla fija @ 10 fps, interpolación lineal
  k_mask    (30,)     bool     -> True = frame real cercano (no extrapolado)
  has_video bool               -> >= MIN_KINESIC_FRAMES frames reales en la ventana
"""
from __future__ import annotations

import bisect
from dataclasses import dataclass

import numpy as np

from core.constants import (KINESIC_DIM, KINESIC_STEPS, MIN_KINESIC_FRAMES,
                            SAMPLE_RATE, STRIDE_S, VIDEO_FPS, WINDOW_S)


@dataclass
class SyncedWindow:
    start_ms: int
    end_ms: int
    audio: np.ndarray
    kinesic: np.ndarray
    k_mask: np.ndarray
    real_frames: int
    has_video: bool
    max_visual_gap_ms: float


class SyncBuffer:
    def __init__(self, sr: int = SAMPLE_RATE, window_s: float = WINDOW_S, stride_s: float = STRIDE_S,
                 fps: int = VIDEO_FPS, jitter_ms: int = 400, av_offset_ms: float = 0.0,
                 max_interp_gap_ms: float = 500.0, retain_s: float = 15.0):
        """
        av_offset_ms: desfase intra-dispositivo a compensar (video retrasado respecto
                      al audio => valor positivo; se RESTA a los timestamps de video).
                      Calibrar con la prueba del aplauso; por defecto 0 (sin supuestos).
        max_interp_gap_ms: huecos visuales mayores no se interpolan (se marcan como faltantes).
        retain_s: historia retenida (permite RESUME dentro de 15 s).
        """
        self.sr, self.win, self.stride, self.fps = sr, window_s, stride_s, fps
        self.jitter_ms, self.av_offset_ms = jitter_ms, av_offset_ms
        self.max_gap = max_interp_gap_ms / 1000.0
        self.retain_s = retain_s
        self._chunks: dict[int, np.ndarray] = {}     # start_sample -> samples (tolera desorden)
        self._k_t: list[float] = []                    # s, ordenado
        self._k_v: list[np.ndarray] = []
        self._next_start = 0.0                         # s, próxima ventana a emitir

    # ------------------------------------------------------------ ingesta
    def push_audio(self, samples: np.ndarray, capture_ts_ms: float) -> None:
        if samples.dtype == np.int16:
            samples = samples.astype(np.float32) / 32768.0
        start = int(round(capture_ts_ms / 1000.0 * self.sr))
        self._chunks[start] = samples.astype(np.float32)

    def push_kinesic(self, vector: np.ndarray, capture_ts_ms: float) -> None:
        t = capture_ts_ms / 1000.0 - self.av_offset_ms / 1000.0
        v = np.asarray(vector, np.float32).reshape(KINESIC_DIM)
        i = bisect.bisect_left(self._k_t, t)
        if i < len(self._k_t) and abs(self._k_t[i] - t) < 1e-6:
            self._k_v[i] = v                 # duplicado -> reemplaza
            return
        self._k_t.insert(i, t); self._k_v.insert(i, v)

    # --------------------------------------------------------- utilidades
    def audio_horizon_s(self) -> float:
        """Fin del audio contiguo disponible desde t=0 (o desde el primer chunk)."""
        if not self._chunks:
            return 0.0
        keys = sorted(self._chunks)
        end = keys[0]
        for k in keys:
            if k > end + 1:        # hueco
                break
            end = max(end, k + len(self._chunks[k]))
        return end / self.sr

    def _audio_slice(self, t0: float, t1: float) -> np.ndarray:
        s0, s1 = int(round(t0 * self.sr)), int(round(t1 * self.sr))
        out = np.zeros(s1 - s0, np.float32)
        for k, v in self._chunks.items():
            a, b = max(k, s0), min(k + len(v), s1)
            if a < b:
                out[a - s0:b - s0] = v[a - k:b - k]
        return out

    def _kinesic_grid(self, t0: float) -> tuple[np.ndarray, np.ndarray, int, float]:
        grid = t0 + np.arange(KINESIC_STEPS) / self.fps
        out = np.zeros((KINESIC_STEPS, KINESIC_DIM), np.float32)
        mask = np.zeros(KINESIC_STEPS, bool)
        if not self._k_t:
            return out, mask, 0, self.win * 1000
        T = np.asarray(self._k_t); V = np.stack(self._k_v)
        inwin = (T >= t0 - 1e-9) & (T < t0 + self.win)
        real = int(inwin.sum())
        ts_in = np.concatenate([[t0], T[inwin], [t0 + self.win]])
        max_gap = float(np.diff(ts_in).max() * 1000) if len(ts_in) > 1 else self.win * 1000
        for j, g in enumerate(grid):
            i = bisect.bisect_left(self._k_t, g)
            left = i - 1 if i > 0 else None
            right = i if i < len(T) else None
            if right is not None and abs(T[right] - g) < 1e-9:
                v, ok = V[right], True
            elif left is not None and right is not None and T[right] - T[left] <= self.max_gap:
                w = (g - T[left]) / (T[right] - T[left])       # interpolación lineal
                a, b = V[left], V[right]
                v = np.where(np.isnan(a), b, np.where(np.isnan(b), a, (1 - w) * a + w * b))
                ok = True
            else:
                nearest = left if right is None else right if left is None else (left if g - T[left] <= T[right] - g else right)
                ok = abs(T[nearest] - g) <= self.max_gap / 2
                v = V[nearest] if ok else np.zeros(KINESIC_DIM, np.float32)
            out[j] = np.nan_to_num(v)
            mask[j] = ok and not np.all(np.isnan(v))
        return out, mask, real, max_gap

    # --------------------------------------------------------------- emisión
    def ready_windows(self, flush: bool = False) -> list[SyncedWindow]:
        horizon = self.audio_horizon_s()
        limit = horizon if flush else horizon - self.jitter_ms / 1000.0
        out = []
        while self._next_start + self.win <= limit + 1e-9:
            out.append(self.window_at(self._next_start))
            self._next_start = round(self._next_start + self.stride, 6)
        self._gc()
        return out

    def window_at(self, t0: float) -> SyncedWindow:
        audio = self._audio_slice(t0, t0 + self.win)
        kin, mask, real, gap = self._kinesic_grid(t0)
        return SyncedWindow(int(round(t0 * 1000)), int(round((t0 + self.win) * 1000)), audio, kin,
                            mask, real, real >= MIN_KINESIC_FRAMES, gap)

    def _gc(self):
        cut = self._next_start - self.retain_s
        if cut <= 0:
            return
        for k in [k for k, v in self._chunks.items() if (k + len(v)) / self.sr < cut]:
            del self._chunks[k]
        i = bisect.bisect_left(self._k_t, cut)
        del self._k_t[:i]; del self._k_v[:i]


def offline_windows(audio: np.ndarray, k_times: np.ndarray, k_vectors: np.ndarray,
                    sr: int = SAMPLE_RATE, **kw) -> list[SyncedWindow]:
    """Atajo offline: alimenta el buffer con archivos completos y devuelve todas las ventanas."""
    buf = SyncBuffer(sr=sr, **kw)
    chunk = int(0.5 * sr)
    for s in range(0, len(audio), chunk):
        buf.push_audio(audio[s:s + chunk], s / sr * 1000)
    for t, v in zip(k_times, k_vectors):
        if not np.all(np.isnan(v)):
            buf.push_kinesic(v, t * 1000)
    return buf.ready_windows(flush=True)
