"""O1: the three fixed decoding variants (P0 control, P1 disfluent prompt, P2 prompt + temperature fallback)."""

from __future__ import annotations

import numpy as np

from mrcd.constants import SAMPLE_RATE, WHISPER_INITIAL_PROMPT
from mrcd.extractors.linguistic import Word, _punct, normalize

# Written once, contains no text from any DIHANA turn (protocol section 3.1).
P1_PROMPT = (
    "Eh... yo quería, quería preguntar por el, por el... horario. Es que no, no sé si... si es el mismo. "
    "Perdone, digo, perdón. La mesa está en la ca... en la casa, o sea, en la casa de mi tía. "
    "Mmm, este... bueno, sí, sí."
)

# Control prompt of PROTOCOL_ANNEX_20261031_3 (attribution control, EXPLORATORY): fluent Spanish, same domain words as
# P1 (horario, preguntar), comparable length, NO repetitions, fillers, ellipses or self-corrections, no DIHANA text.
C1_PROMPT = (
    "Buenos días, quería preguntar por el horario de las salidas de mañana. Me interesa conocer la duración del "
    "trayecto hasta la ciudad vecina y las tarifas para estudiantes. Además, quisiera elegir un asiento junto a la "
    "ventanilla, en un vagón silencioso. Muchas gracias por su ayuda."
)

VARIANTS: dict[str, dict] = {
    "P0": dict(initial_prompt=WHISPER_INITIAL_PROMPT, temperature=0.0),
    "P1": dict(initial_prompt=P1_PROMPT, temperature=0.0),
    "C1": dict(initial_prompt=C1_PROMPT, temperature=0.0),
    "P2": dict(
        initial_prompt=P1_PROMPT,
        condition_on_previous_text=False,
        temperature=(0.0, 0.2, 0.4),
        compression_ratio_threshold=2.4,
        log_prob_threshold=-1.0,
        no_speech_threshold=0.6,
    ),
}
COMMON = dict(
    word_timestamps=True,
    condition_on_previous_text=False,
    suppress_tokens=[],
    vad_filter=False,
    beam_size=5,
)


def decode_kwargs(variant: str, language: str = "es") -> dict:
    if variant not in VARIANTS:
        raise KeyError(f"unknown variant {variant!r}; the protocol fixes P0/P1/P2")
    kw = dict(COMMON)
    kw.update(VARIANTS[variant])
    kw["language"] = language
    return kw


def transcribe(asr, audio: np.ndarray, variant: str) -> list[Word]:
    """Same word construction as ``VerbatimASR.transcribe`` but with the variant's decoding parameters."""
    segments, _ = asr.model.transcribe(
        audio.astype(np.float32), **decode_kwargs(variant, asr.language)
    )
    words: list[Word] = []
    for seg in segments:
        for w in seg.words or []:
            words.append(
                Word(
                    w.word.strip(),
                    normalize(w.word),
                    float(w.start),
                    float(w.end),
                    float(w.probability),
                    _punct(w.word),
                )
            )
    return [w for w in words if w.norm]


def transcribe_chunked(
    asr, audio, variant: str, chunk_s: float = 8.0, overlap_s: float = 2.0
) -> list[Word]:
    """Mirror of ``VerbatimASR.transcribe_chunked`` (only turns longer than ``chunk_s`` are chunked)."""
    if chunk_s <= 0 or len(audio) <= chunk_s * SAMPLE_RATE:
        return transcribe(asr, audio, variant)
    step = int((chunk_s - overlap_s) * SAMPLE_RATE)
    size = int(chunk_s * SAMPLE_RATE)
    out: list[Word] = []
    for k, start in enumerate(range(0, len(audio), step)):
        seg = audio[start : start + size]
        if len(seg) < SAMPLE_RATE:
            break
        offset = start / SAMPLE_RATE
        new_from = 0.0 if k == 0 else overlap_s
        for w in transcribe(asr, seg, variant):
            if (w.start + w.end) / 2 < new_from:
                continue
            out.append(
                Word(
                    w.text,
                    w.norm,
                    w.start + offset,
                    w.end + offset,
                    w.prob,
                    w.punct_after,
                )
            )
    out.sort(key=lambda w: w.start)
    return out


def words_text(words) -> str:
    """Same join as the parent ``asr_text_from_cache``."""
    return " ".join(w["text"] if isinstance(w, dict) else w.text for w in words)
