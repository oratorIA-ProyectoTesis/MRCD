"""ASR wrapper that answers from the per-chunk cache and otherwise decodes exactly like the offline path.

It subclasses ``VerbatimASR`` only to inherit the UNCHANGED ``transcribe_chunked`` (planning and overlap merge),
so the offline code path is the one that assembles the final transcript. Only ``transcribe`` is replaced.
"""

from __future__ import annotations

import os
from typing import Callable

import numpy as np

from mrcd.constants import SAMPLE_RATE
from mrcd.extractors.linguistic import VerbatimASR
from mrcd.service.incremental.cache import ChunkCache, segment_key


def asr_namespace(real) -> str:
    """Everything that can change the text of a chunk (profile, prompt, parameters, model, device, threads)."""
    profile = getattr(real, "profile", None)
    if profile is None:
        return f"unprofiled:{id(real)}"
    from mrcd.asr_profiles import asr_cache_key

    info = real.model_info() if hasattr(real, "model_info") else {}
    return asr_cache_key(
        profile,
        audio_sha256="-",
        model_bin_sha256=info.get("model_bin_sha256"),
        compute_type=getattr(real, "compute_type", None),
        device=getattr(real, "device", None),
        omp_num_threads=os.environ.get("OMP_NUM_THREADS"),
        language=getattr(real, "language", "es"),
    )


def decode_with(real, seg: np.ndarray, start_sample: int) -> tuple[list, list]:
    """One real decode of ``seg`` (a chunk starting at ``start_sample``) with the same per-chunk state the offline
    ``transcribe_chunked`` sets. Returns ``(words, redecode_events)``; the shared ASR state is restored afterwards."""
    saved = getattr(real, "_chunk_offset_s", 0.0)
    real.redecode_events = []
    real._chunk_offset_s = start_sample / SAMPLE_RATE
    try:
        words = real.transcribe(seg)
        events = [dict(e) for e in getattr(real, "redecode_events", [])]
    finally:
        real.redecode_events = []
        real._chunk_offset_s = saved
    return words, events


class CachedASR(VerbatimASR):
    def __init__(  # noqa: D107
        self,
        real,
        cache: ChunkCache,
        namespace: str | None = None,
        should_stop: Callable[[], bool] | None = None,
    ) -> None:
        # ``should_stop`` is probed only BEFORE a real decode (a safe point between chunks); it never alters a result.
        # Share the loaded model and every attribute of the real ASR; it is never modified.
        if hasattr(real, "model_info"):
            real.model_info()  # resolve (and memoise on ``real``) before copying its state
        self.__dict__.update(real.__dict__)
        self._real = real
        self._cache = cache
        self._namespace = namespace or asr_namespace(real)
        self.decode_calls = 0
        self._should_stop = should_stop

    def transcribe(self, audio):  # noqa: D102
        start = int(round(getattr(self, "_chunk_offset_s", 0.0) * SAMPLE_RATE))
        key = segment_key(self._namespace, start, audio)
        hit = self._cache.get(key)
        if hit is not None:
            words, events = hit
        else:
            if self._should_stop is not None and self._should_stop():
                from mrcd.service.incremental.session import SessionAbandoned

                raise SessionAbandoned()
            self.decode_calls += 1
            words, events = decode_with(self._real, np.asarray(audio, dtype=np.float32), start)
        if events:
            if not hasattr(self, "redecode_events"):
                self.redecode_events = []
            self.redecode_events.extend(events)
        return words
