"""Overlapped Faster-Whisper transcription with session-absolute word timing."""
from __future__ import annotations

from core.extractors.linguistic import VerbatimASR, Word


def dedupe_words(existing: list[Word], incoming: list[Word], tolerance_s: float = 0.12) -> list[Word]:
    """Merge retranscribed overlap without duplicating the same timestamped token."""
    merged: list[Word] = []
    for word in sorted([*existing, *incoming], key=lambda item: (item.start, item.end)):
        same = next((old for old in merged if old.norm == word.norm
                     and abs(old.start - word.start) <= tolerance_s
                     and abs(old.end - word.end) <= tolerance_s), None)
        if same is None:
            merged.append(word)
        elif word.prob > same.prob:
            merged[merged.index(same)] = word
    return merged


class StreamingASR:
    def __init__(self, model_size: str = "small", variety: str = "es-PE", asr: VerbatimASR | None = None):
        self.asr = asr or VerbatimASR(model_size)
        self.variety = variety
        self.words: list[Word] = []

    def transcribe_recent(self, audio, buffer_start_s: float) -> list[Word]:
        local = self.asr.transcribe(audio)
        absolute = [Word(w.text, w.norm, w.start + buffer_start_s, w.end + buffer_start_s,
                         w.prob, w.punct_after) for w in local]
        # All timestamps inside the retranscribed interval are provisional and are replaced.
        retained = [w for w in self.words if w.end < buffer_start_s - 0.12]
        self.words = dedupe_words(retained, absolute)
        return list(self.words)
