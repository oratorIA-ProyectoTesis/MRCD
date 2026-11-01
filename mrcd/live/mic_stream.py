"""Microphone capture and an in-memory, thread-safe audio ring buffer."""

from __future__ import annotations

import threading
from collections import deque

import numpy as np

from mrcd.constants import CONTEXT_S, SAMPLE_RATE, WINDOW_S


class AudioRingBuffer:
    def __init__(
        self, sample_rate: int = SAMPLE_RATE, retain_s: float = CONTEXT_S + WINDOW_S
    ):
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
            self._chunks.append(chunk)
            self._samples += len(chunk)
            self._total_samples += len(chunk)
            while self._samples > self.capacity:
                excess = self._samples - self.capacity
                first = self._chunks[0]
                if len(first) <= excess:
                    self._chunks.popleft()
                    self._samples -= len(first)
                else:
                    self._chunks[0] = first[excess:]
                    self._samples -= excess

    def clear(self) -> None:
        with self._lock:
            self._chunks.clear()
            self._samples = 0
            self._total_samples = 0

    @property
    def duration_s(self) -> float:
        with self._lock:
            return self._total_samples / self.sample_rate

    def read_since(self, abs_sample: int) -> tuple[np.ndarray, int, int]:
        """Samples from absolute index ``abs_sample`` on: ``(audio, start_sample, lost)``.

        ``lost`` counts samples that were already evicted from the ring (the consumer fell more than
        ``retain_s`` behind); ``start_sample = abs_sample + lost``.
        """
        with self._lock:
            buffered_start = self._total_samples - self._samples
            start = max(int(abs_sample), buffered_start)
            audio = (
                np.concatenate(tuple(self._chunks))
                if self._chunks
                else np.zeros(0, np.float32)
            )
            audio = audio[start - buffered_start :]
        return audio.copy(), start, start - int(abs_sample)

    def latest(self, seconds: float | None = None) -> tuple[np.ndarray, float]:
        """Return a copy and the absolute session timestamp of its first sample."""
        with self._lock:
            audio = (
                np.concatenate(tuple(self._chunks))
                if self._chunks
                else np.zeros(0, np.float32)
            )
            if seconds is not None:
                audio = audio[-int(round(seconds * self.sample_rate)) :]
            start = (self._total_samples - len(audio)) / self.sample_rate
        return audio.copy(), start


class SessionRecorder:
    """Acumula la sesion COMPLETA, no solo la ventana de contexto.

    `AudioRingBuffer` retiene 13 s porque es lo que el modelo necesita para una
    ventana. Para auditar el sistema hace falta el audio entero: sin el no se puede
    distinguir «el modelo no detecto la disfluencia» de «el ASR la borro antes de
    que el modelo la viera», que son fallos en capas distintas y con arreglos
    distintos.

    A 16 kHz float32 son ~3.8 MB por minuto de sesion.
    """

    def __init__(self, sample_rate: int = SAMPLE_RATE):
        self.sample_rate = sample_rate
        self._chunks: list[np.ndarray] = []
        self._lock = threading.Lock()

    def append(self, audio: np.ndarray) -> None:
        chunk = np.asarray(audio, dtype=np.float32).reshape(-1).copy()
        if len(chunk):
            with self._lock:
                self._chunks.append(chunk)

    def clear(self) -> None:
        with self._lock:
            self._chunks.clear()

    @property
    def duration_s(self) -> float:
        with self._lock:
            return sum(len(c) for c in self._chunks) / self.sample_rate

    def samples(self) -> np.ndarray:
        with self._lock:
            return (
                np.concatenate(tuple(self._chunks))
                if self._chunks
                else np.zeros(0, np.float32)
            )

    def to_wav_bytes(self) -> bytes:
        """WAV PCM 16-bit mono. Se recorta a [-1, 1] antes de escalar: un pico que
        desborde se envolveria y sonaria a chasquido, y el audio es la evidencia."""
        import io
        import wave

        x = np.clip(self.samples(), -1.0, 1.0)
        pcm = (x * 32767.0).astype(np.int16)
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(self.sample_rate)
            w.writeframes(pcm.tobytes())
        return buf.getvalue()


class MicrophoneStream:
    def __init__(
        self,
        device: int | str | None = None,
        sample_rate: int = SAMPLE_RATE,
        retain_s: float = CONTEXT_S + WINDOW_S,
        record: bool = True,
        fill_dropped: bool = False,
    ):
        self.device, self.sample_rate = device, sample_rate
        #: ``fill_dropped=False`` (historic): a block that PortAudio flags (``status``) is discarded and the
        #: timeline silently shortens. ``True``: it is replaced by zeros so the timeline keeps its length.
        #: Either way the event is counted in ``dropped_blocks`` / ``dropped_samples``.
        self.fill_dropped = fill_dropped
        self.dropped_blocks = 0
        self.dropped_samples = 0
        self.buffer = AudioRingBuffer(sample_rate, retain_s)
        self.recorder = SessionRecorder(sample_rate) if record else None
        self._stream = None

    def start(self) -> None:
        try:
            import sounddevice as sd

            sd.check_input_settings(
                device=self.device, samplerate=self.sample_rate, channels=1
            )
            self._stream = sd.InputStream(
                device=self.device,
                samplerate=self.sample_rate,
                channels=1,
                dtype="float32",
                callback=self._callback,
            )
            self._stream.start()
        except Exception as exc:
            try:
                import sounddevice as sd

                devices = sd.query_devices()
                listing = "\n".join(
                    f"  [{i}] {d['name']} (inputs={d['max_input_channels']})"
                    for i, d in enumerate(devices)
                    if d["max_input_channels"]
                )
            except Exception:
                listing = "  (no se pudieron consultar dispositivos)"
            raise RuntimeError(
                "No se pudo abrir un micrófono mono a "
                + str(self.sample_rate)
                + " Hz. Conecta/selecciona un dispositivo de entrada y lista los disponibles con "
                '`python -c "import sounddevice as sd; print(sd.query_devices())"`.\n'
                + listing
            ) from exc

    def _callback(self, indata, frames, time_info, status) -> None:
        block = indata[:, 0]
        if status:
            self.dropped_blocks += 1
            self.dropped_samples += len(block)
            if not self.fill_dropped:
                return
            block = np.zeros(len(block), np.float32)
        self.buffer.append(block)
        if self.recorder is not None:
            self.recorder.append(block)

    def latest_audio(self, seconds: float | None = None) -> tuple[np.ndarray, float]:
        return self.buffer.latest(seconds)

    def reset(self) -> None:
        self.dropped_blocks = self.dropped_samples = 0
        self.buffer.clear()
        if self.recorder is not None:
            self.recorder.clear()

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
