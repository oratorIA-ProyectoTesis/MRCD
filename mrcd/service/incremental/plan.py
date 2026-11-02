"""Pure integer chunk planning for the incremental ASR (mirror of ``VerbatimASR.transcribe_chunked``).

The plan only decides WHICH chunks may be decoded early. The final assembly is always done by the untouched
``transcribe_chunked``; a wrong plan can therefore only cause a cache miss (more latency), never a different result.
"""

from __future__ import annotations

SAMPLE_RATE = 16_000
CHUNK_SAMPLES = 8 * SAMPLE_RATE  # int(8.0 * 16000), see linguistic.py transcribe_chunked
STEP_SAMPLES = 6 * SAMPLE_RATE  # int((8.0 - 2.0) * 16000)
MIN_SEGMENT_SAMPLES = SAMPLE_RATE  # a segment shorter than 1 s stops the loop


def complete_chunks(n_samples: int) -> list[tuple[int, int]]:
    """``(k, start)`` of every chunk that is entirely inside ``n_samples`` (its content cannot change any more)."""
    out: list[tuple[int, int]] = []
    k = 0
    while k * STEP_SAMPLES + CHUNK_SAMPLES <= n_samples:
        out.append((k, k * STEP_SAMPLES))
        k += 1
    return out


def final_plan(n_samples: int) -> list[tuple[int, int, int]]:
    """``(k, start, length)`` of the segments the offline path decodes for ``n_samples`` of audio."""
    if n_samples <= CHUNK_SAMPLES:
        return [(0, 0, n_samples)]
    plan: list[tuple[int, int, int]] = []
    for k, start in enumerate(range(0, n_samples, STEP_SAMPLES)):
        length = min(CHUNK_SAMPLES, n_samples - start)
        if length < MIN_SEGMENT_SAMPLES:
            break
        plan.append((k, start, length))
    return plan
