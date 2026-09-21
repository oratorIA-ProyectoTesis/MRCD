"""Build audio_text tensors through the exact offline feature functions."""
from __future__ import annotations

import numpy as np

from core.constants import SAMPLE_RATE, WINDOW_S
from core.dataset import acoustic_windows, center_f0, linguistic_windows
from core.extractors.acoustic import extract_acoustic
from core.extractors.linguistic import Word, word_features
from core.extractors.prosody import prosody_windows, speaker_f0_center


def _word_dicts(words: list[Word], offset_s: float = 0.0) -> list[dict]:
    return [{**word.to_dict(), "start": word.start - offset_s, "end": word.end - offset_s} for word in words]


def window_text(words: list[Word], start_s: float) -> str:
    return " ".join(word.text for word in words if start_s <= (word.start + word.end) / 2 < start_s + WINDOW_S)


def build_window(audio: np.ndarray, words: list[Word], start_s: float, *, word_offset_s: float = 0.0,
                 use_silero: bool = True, context: bool = True,
                 include_pros: bool = True, ling_dim: int | None = None) -> dict[str, np.ndarray]:
    """Build one window in the identical offline ordering: pros, ac, ling, lpos, lmask."""
    track = extract_acoustic(audio, use_silero=use_silero)
    frames = track.frame_matrix()
    center = speaker_f0_center(frames)
    starts = np.asarray([start_s], np.float64)
    local_words = _word_dicts(words, word_offset_s)
    feats = word_features([Word(**word) for word in local_words])
    ling, lpos, lmask = linguistic_windows(local_words, feats, starts, context=context)
    if ling_dim is not None:
        if ling_dim > ling.shape[-1]:
            raise ValueError(f"El checkpoint espera ling={ling_dim}, pero el extractor produce {ling.shape[-1]}")
        ling = ling[:, :, :ling_dim]
    out = {"ac": acoustic_windows(center_f0(frames, center), starts),
           "ling": ling, "lpos": lpos, "lmask": lmask}
    if include_pros:
        out["pros"] = prosody_windows(frames, starts, center)
    return out


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
