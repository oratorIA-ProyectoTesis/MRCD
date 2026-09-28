"""Build audio_text tensors through the exact offline feature functions."""
from __future__ import annotations

import numpy as np

from core.constants import SAMPLE_RATE, WINDOW_S
from core.engine import window_tensors
from core.extractors.acoustic import extract_acoustic
from core.extractors.linguistic import Word


def window_text(words: list[Word], start_s: float) -> str:
    return " ".join(word.text for word in words if start_s <= (word.start + word.end) / 2 < start_s + WINDOW_S)


def build_window(audio: np.ndarray, words: list[Word], start_s: float, *, word_offset_s: float = 0.0,
                 use_silero: bool = True, context: bool = True,
                 include_pros: bool = True, ling_dim: int | None = None) -> dict[str, np.ndarray]:
    """One window over a fresh buffer; offline analysis uses core.engine to extract once."""
    frames = extract_acoustic(audio, use_silero=use_silero).frame_matrix()
    local = [Word(w.text, w.norm, w.start - word_offset_s, w.end - word_offset_s, w.prob, w.punct_after)
             for w in words]
    return window_tensors(frames, local, [start_s], context=context, include_pros=include_pros, ling_dim=ling_dim)


def build_latest_contextual_window(audio: np.ndarray, buffer_start_s: float, words: list[Word],
                                   *, use_silero: bool = True, context: bool = True,
                                   include_pros: bool = True,
                                   ling_dim: int | None = None) -> tuple[dict[str, np.ndarray], float, str]:
    """Emit the newest window whose 10-second linguistic context is complete.

    `linguistic_windows` looks five seconds ahead, therefore this window starts
    five seconds before the current end of the buffer.
    """
    now_s = buffer_start_s + len(audio) / SAMPLE_RATE
    start_s = now_s - (5.0 if context else WINDOW_S)
    tensors = build_window(audio, words, start_s - buffer_start_s, word_offset_s=buffer_start_s,
                           use_silero=use_silero, context=context, include_pros=include_pros, ling_dim=ling_dim)
    return tensors, start_s, window_text(words, start_s)
