"""Microphone capture and an in-memory, thread-safe audio ring buffer."""
from __future__ import annotations

import threading
from collections import deque

import numpy as np

from core.constants import CONTEXT_S, SAMPLE_RATE, WINDOW_S


class AudioRingBuffer:
    def __init__(self, sample_rate: int = SAMPLE_RATE, retain_s: float = CONTEXT_S + WINDOW_S):
        self.sample_rate = sample_rate
        self.capacity = int(round(sample_rate * retain_s))
        self._chunks: deque[np.ndarray] = deque()
        self._samples = 0
        self._total_samples = 0
        self._lock = threading.Lock()

    def append(self, audio: np.ndarray) -> None:
        chunk = np.asarray(audio, dtype=np.float32).reshape(-1).copy()
        if not len(chunk):
            return
        with self._lock:
            self._chunks.append(chunk); self._samples += len(chunk); self._total_samples += len(chunk)
            while self._samples > self.capacity:
                excess = self._samples - self.capacity
                first = self._chunks[0]
                if len(first) <= excess:
                    self._chunks.popleft(); self._samples -= len(first)
                else:
                    self._chunks[0] = first[excess:]; self._samples -= excess

    def clear(self) -> None:
        with self._lock:
            self._chunks.clear(); self._samples = 0; self._total_samples = 0

    @property
    def duration_s(self) -> float:
        with self._lock:
            return self._total_samples / self.sample_rate

    def latest(self, seconds: float | None = None) -> tuple[np.ndarray, float]:
        """Return a copy and the absolute session timestamp of its first sample."""
        with self._lock:
            audio = np.concatenate(tuple(self._chunks)) if self._chunks else np.zeros(0, np.float32)
            if seconds is not None:
                audio = audio[-int(round(seconds * self.sample_rate)):]
            start = (self._total_samples - len(audio)) / self.sample_rate
        return audio.copy(), start


class MicrophoneStream:
    def __init__(self, device: int | str | None = None, sample_rate: int = SAMPLE_RATE,
                 retain_s: float = CONTEXT_S + WINDOW_S):
        self.device, self.sample_rate = device, sample_rate
        self.buffer = AudioRingBuffer(sample_rate, retain_s)
        self._stream = None

    def start(self) -> None:
        try:
            import sounddevice as sd
            sd.check_input_settings(device=self.device, samplerate=self.sample_rate, channels=1)
            self._stream = sd.InputStream(device=self.device, samplerate=self.sample_rate, channels=1,
                                          dtype="float32", callback=self._callback)
            self._stream.start()
        except Exception as exc:
            try:
                import sounddevice as sd
                devices = sd.query_devices()
                listing = "\n".join(f"  [{i}] {d['name']} (inputs={d['max_input_channels']})"
                                    for i, d in enumerate(devices) if d['max_input_channels'])
            except Exception:
                listing = "  (no se pudieron consultar dispositivos)"
            raise RuntimeError("No se pudo abrir un micrófono mono a " + str(self.sample_rate)
                               + " Hz. Conecta/selecciona un dispositivo de entrada y lista los disponibles con "
                               "`python -c \"import sounddevice as sd; print(sd.query_devices())\"`.\n" + listing) from exc

    def _callback(self, indata, frames, time_info, status) -> None:
        if status:
            return
        self.buffer.append(indata[:, 0])

    def latest_audio(self, seconds: float | None = None) -> tuple[np.ndarray, float]:
        return self.buffer.latest(seconds)

    def reset(self) -> None:
        self.buffer.clear()

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop(); self._stream.close(); self._stream = None
