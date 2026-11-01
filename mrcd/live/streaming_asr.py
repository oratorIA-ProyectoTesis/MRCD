"""Overlapped Faster-Whisper transcription with session-absolute word timing."""
from __future__ import annotations

from dataclasses import dataclass, field

from mrcd.constants import SAMPLE_RATE
from mrcd.extractors.linguistic import VerbatimASR, Word


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


@dataclass
class ChunkAlignedState:
    """Per-session state of the OPT-IN chunk-aligned streaming mode (see ``StreamingASR.step_chunk_aligned``)."""

    chunk_s: float = 8.0
    overlap_s: float = 2.0
    #: words of the chunks that are complete and therefore final (never rewritten)
    frozen: list[Word] = field(default_factory=list)
    #: number of chunks already frozen (chunk ``k`` starts at ``k * (chunk_s - overlap_s)``)
    n_frozen: int = 0
    #: chunks that could not be decoded because they had already left the audio ring (consumer lag), recorded not hidden
    lost_chunks: list[int] = field(default_factory=list)
    decode_calls: int = 0


class StreamingASR:
    def __init__(self, model_size: str = "small", variety: str = "es-PE", asr: VerbatimASR | None = None):
        self.asr = asr or VerbatimASR(model_size)
        self.variety = variety
        self.words: list[Word] = []

    def asr_mark(self) -> dict | None:
        """Row mark for the live path: ``None`` for the validated default p0 (rows keep their historical
        schema), a small dict for any other / unknown profile (see ``LiveScorer(asr_mark=...)``)."""
        profile = getattr(self.asr, "profile", None)
        if profile is None:
            return {"asr_profile": "unknown", "asr_experimental": None}
        if not profile.experimental and profile.name == "p0":
            return None
        return {"asr_profile": profile.name, "asr_experimental": bool(profile.experimental)}

    def merge_recent(self, previous: list[Word], audio, buffer_start_s: float) -> list[Word]:
        """Re-transcribe ``audio`` and merge it into ``previous`` WITHOUT touching ``self.words``.

        The live engine keeps the words of each session itself, so a worker of an older session can
        never overwrite the words of a newer one (quick stop -> start).
        """
        local = self.asr.transcribe(audio)
        absolute = [Word(w.text, w.norm, w.start + buffer_start_s, w.end + buffer_start_s,
                         w.prob, w.punct_after) for w in local]
        # All timestamps inside the retranscribed interval are provisional and are replaced.
        retained = [w for w in previous if w.end < buffer_start_s - 0.12]
        return dedupe_words(retained, absolute)

    def transcribe_recent(self, audio, buffer_start_s: float) -> list[Word]:
        self.words = self.merge_recent(self.words, audio, buffer_start_s)
        return list(self.words)

    # ------------------------------------------------------------------ opt-in: chunk-aligned streaming
    def step_chunk_aligned(
        self, state: ChunkAlignedState, audio, buffer_start_s: float
    ) -> list[Word]:
        """Streaming that uses the SAME chunk grid as the engine's offline path (``transcribe_chunked``). OPT-IN.

        Why: ``merge_recent`` re-decodes the last 13 s in ONE Whisper pass, but the engine decodes 8 s chunks with 2 s overlap.
        A single long pass loses repetitions and can fall into repetition loops; the two paths therefore disagree.
        Here chunk ``k`` covers ``[6k, 6k + 8)`` s of the session. A chunk that is complete is decoded ONCE and frozen
        (only its region after the 2 s overlap is kept, exactly like the offline path); the incomplete chunks (at most
        two) are decoded again at every tick as provisional words. On the last tick (when the audio stops) the result equals ``transcribe_chunked`` on the whole session word for word, because the same samples
        are decoded with the same parameters. ``audio`` must hold the samples ``[buffer_start_s, now)``; the live ring
        (13 s) is enough while a tick comes at least every ~5 s, otherwise the chunk is recorded in ``state.lost_chunks``.
        Nothing here is used unless the caller asks for it; the live driver keeps ``merge_recent``.
        """
        sr = SAMPLE_RATE
        step_s = state.chunk_s - state.overlap_s
        now = buffer_start_s + len(audio) / sr

        def decode(k: int, end_s: float) -> list[Word] | None:
            start_s = k * step_s
            if start_s < buffer_start_s - 1e-9:
                return None  # already out of the ring
            a = int(round((start_s - buffer_start_s) * sr))
            seg = audio[a : a + int(round(min(end_s - start_s, state.chunk_s) * sr))]
            if len(seg) < sr:  # < 1 s adds nothing (as offline)
                return []
            state.decode_calls += 1
            new_from = 0.0 if k == 0 else state.overlap_s
            return [
                Word(w.text, w.norm, w.start + start_s, w.end + start_s, w.prob, w.punct_after)
                for w in self.asr.transcribe(seg)
                if (w.start + w.end) / 2 >= new_from
            ]

        # 1) freeze every chunk that is complete (its full chunk_s of audio has been heard)
        while (state.n_frozen) * step_s + state.chunk_s <= now + 1e-9:
            got = decode(state.n_frozen, state.n_frozen * step_s + state.chunk_s)
            if got is None:
                state.lost_chunks.append(state.n_frozen)
            else:
                state.frozen.extend(got)
            state.n_frozen += 1
        # 2) provisional: chunks that have started (>= 1 s of audio) but are not complete
        provisional: list[Word] = []
        k = state.n_frozen
        while k * step_s + 1.0 <= now + 1e-9:
            got = decode(k, now)
            if got:
                provisional.extend(got)
            k += 1
        out = sorted([*state.frozen, *provisional], key=lambda w: w.start)
        return out
