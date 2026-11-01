"""S2 (2026-11-01): loop metrics and the experimental decoders A/B/C of ``results/mejora_objetivos_20261101/S2/PROTOCOL.md``.

Nothing here is a default. The decoders mirror ``VerbatimASR.transcribe_chunked`` but let a branch change the decoding kwargs, the chunk
grid or re-decode a looped chunk; every re-decode is RECORDED (the returned diagnostics), never silent.
"""

from __future__ import annotations

import numpy as np

from mrcd.asr_profiles import LOOP_REDECODE_LADDER, chunk_looped
from mrcd.constants import SAMPLE_RATE
from mrcd.extractors.linguistic import Word, _punct, normalize

LOOP_RUN_MIN_REPEATS = (
    4  # runs with fewer back-to-back repeats are ordinary speech and are not counted
)
LOOP_CLIP_EXCESS = 20  # hypothesis loop excess (tokens) over the reference from which a clip is a "loop clip"


def run_excess(tokens: list[str], min_repeats: int = LOOP_RUN_MIN_REPEATS) -> int:
    """Tokens inside back-to-back n-gram runs (n = 1..4) with >= ``min_repeats`` repeats, counting every repeat after the first."""
    i, total = 0, 0
    while i < len(tokens):
        best_cov, best_k, best_n = 0, 0, 0
        for n in (1, 2, 3, 4):
            unit = tokens[i : i + n]
            if len(unit) < n:
                break
            k = 1
            while tokens[i + k * n : i + (k + 1) * n] == unit:
                k += 1
            if k >= min_repeats and k * n > best_cov:
                best_cov, best_k, best_n = k * n, k, n
        if best_cov:
            total += (best_k - 1) * best_n
            i += best_cov
        else:
            i += 1
    return total


def loop_excess(hyp_tokens: list[str], ref_tokens: list[str]) -> int:
    """Hypothesis loop tokens beyond what the human reference itself contains (floored at 0)."""
    return max(run_excess(hyp_tokens) - run_excess(ref_tokens), 0)


def is_loop_clip(hyp_tokens: list[str], ref_tokens: list[str]) -> bool:
    return loop_excess(hyp_tokens, ref_tokens) >= LOOP_CLIP_EXCESS


def looped_text(text: str) -> bool:
    """Chunk-level detector of the B branch; the SAME function the opt-in profiles use (``mrcd.asr_profiles.chunk_looped``)."""
    return chunk_looped(text)


# ---------------------------------------------------------------------------------------------- decoding
P2_FALLBACK = dict(
    temperature=(0.0, 0.2, 0.4),
    compression_ratio_threshold=2.4,
    log_prob_threshold=-1.0,
    no_speech_threshold=0.6,
)
#: B-branch re-decode ladder (protocol section 2): attempt kwargs overriding the profile kwargs
B_LADDER = tuple(LOOP_REDECODE_LADDER)
SEED = 13


def _decode(asr, audio: np.ndarray, kwargs: dict) -> list[Word]:
    segments, _ = asr.model.transcribe(audio.astype(np.float32), **kwargs)
    words = []
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


def _text(words: list[Word]) -> str:
    return " ".join(w.text for w in words)


def transcribe_chunked_branch(
    asr,
    audio: np.ndarray,
    base_kwargs: dict,
    chunk_s: float = 8.0,
    overlap_s: float = 2.0,
    ladder: tuple | None = None,
) -> tuple[list[Word], dict]:
    """``VerbatimASR.transcribe_chunked`` with explicit kwargs; optional per-chunk loop ladder. Returns (words, diagnostics)."""
    import ctranslate2  # type: ignore

    diag = {"chunks": 0, "looped_chunks": 0, "redecoded": [], "unresolved": 0}
    if chunk_s <= 0 or len(audio) <= chunk_s * SAMPLE_RATE:
        spans = [(0, len(audio), 0.0)]
    else:
        step = int((chunk_s - overlap_s) * SAMPLE_RATE)
        size = int(chunk_s * SAMPLE_RATE)
        spans = []
        for k, start in enumerate(range(0, len(audio), step)):
            if len(audio[start : start + size]) < SAMPLE_RATE:
                break
            spans.append((start, start + size, 0.0 if k == 0 else overlap_s))
    out: list[Word] = []
    for k, (a, b, new_from) in enumerate(spans):
        seg = audio[a:b]
        words = _decode(asr, seg, base_kwargs)
        diag["chunks"] += 1
        if ladder and looped_text(_text(words)):
            diag["looped_chunks"] += 1
            for j, override in enumerate(ladder, start=1):
                ctranslate2.set_random_seed(SEED)
                alt = _decode(asr, seg, {**base_kwargs, **override})
                if not looped_text(_text(alt)):
                    diag["redecoded"].append(
                        {
                            "chunk": k,
                            "attempt": j,
                            "words_before": len(words),
                            "words_after": len(alt),
                        }
                    )
                    words = alt
                    break
            else:
                diag["unresolved"] += 1
                diag["redecoded"].append(
                    {"chunk": k, "attempt": None, "kept_original": True}
                )
        offset = a / SAMPLE_RATE
        for w in words:
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
    return out, diag
