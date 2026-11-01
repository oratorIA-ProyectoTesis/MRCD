"""Transcripcion por trozos: no debe borrar las repeticiones que intenta preservar.

El riesgo concreto de esta funcion es sutil. Al transcribir en trozos solapados
aparecen palabras duplicadas en la zona de solape, y la forma obvia de quitarlas
—comparar texto— borraria tambien las repeticiones REALES del hablante, que es
justo el fenomeno que se quiere detectar. Estas pruebas fijan que la
deduplicacion sea por region y nunca por coincidencia de texto.
"""

from __future__ import annotations

import numpy as np
import pytest

from mrcd.constants import SAMPLE_RATE
from mrcd.extractors.linguistic import VerbatimASR, Word


class FakeASR(VerbatimASR):
    """ASR de prueba, sin cargar Whisper.

    El indice de cada token va codificado en la AMPLITUD del audio (segundo i ->
    amplitud i/1000), de modo que el falso ASR lee del audio que palabra le toca.
    Sin esto el falso devolveria siempre los primeros tokens sin importar que
    trozo recibe, y la prueba pasaria o fallaria por un artefacto del doble, no
    por el comportamiento real del troceado.
    """

    def __init__(self, tokens_per_second: list[str]):
        self.tokens = tokens_per_second
        self.model_size, self.device = "fake", "cpu"

    def transcribe(self, audio: np.ndarray) -> list[Word]:
        out = []
        for i in range(int(len(audio) / SAMPLE_RATE)):
            idx = int(round(float(audio[i * SAMPLE_RATE]) * 1000))
            if 0 <= idx < len(self.tokens):
                tok = self.tokens[idx]
                out.append(Word(tok, tok, float(i), float(i) + 0.5, 1.0, ""))
        return out


def _audio(tokens: list[str]) -> np.ndarray:
    """Un segundo por token, con el indice del token codificado en la amplitud."""
    a = np.zeros(len(tokens) * SAMPLE_RATE, np.float32)
    for i in range(len(tokens)):
        a[i * SAMPLE_RATE:(i + 1) * SAMPLE_RATE] = i / 1000.0
    return a


def test_chunking_preserves_a_real_repetition():
    """«la la la» debe sobrevivir: es el fenomeno a detectar, no un duplicado."""
    tokens = ["el", "sistema", "la", "la", "la", "primera", "es", "acustica"]
    asr = FakeASR(tokens)
    words = asr.transcribe_chunked(_audio(tokens), chunk_s=4.0, overlap_s=2.0)
    got = [w.norm for w in words]
    assert got == tokens, f"la transcripcion por trozos altero la secuencia: {got}"
    adjacent = sum(1 for a, b in zip(words, words[1:]) if a.norm == b.norm)
    assert adjacent == 2, "las dos repeticiones adyacentes de «la» deben conservarse"


def test_chunking_does_not_duplicate_across_the_overlap():
    """Una palabra en la zona de solape se cuenta UNA vez, no dos."""
    tokens = [f"w{i}" for i in range(12)]
    asr = FakeASR(tokens)
    words = asr.transcribe_chunked(_audio(tokens), chunk_s=5.0, overlap_s=2.0)
    got = [w.norm for w in words]
    assert got == tokens, f"solape duplico o perdio palabras: {got}"
    assert len(got) == len(set(got)), "ninguna palabra distinta debe repetirse"


def test_timestamps_stay_absolute_and_monotonic():
    """Los tiempos deben ser de la sesion, no del trozo: la cascada los usa para
    medir huecos y duraciones, y un reinicio por trozo falsearia esos rasgos."""
    tokens = [f"w{i}" for i in range(10)]
    words = FakeASR(tokens).transcribe_chunked(_audio(tokens), chunk_s=4.0, overlap_s=1.0)
    starts = [w.start for w in words]
    assert starts == sorted(starts), "los tiempos deben ser monotonos crecientes"
    assert starts[-1] > 5.0, "el ultimo tiempo debe ser absoluto, no relativo al trozo"


def test_short_audio_falls_back_to_single_pass():
    """Audio mas corto que un trozo no necesita trocearse."""
    tokens = ["hola", "mundo"]
    words = FakeASR(tokens).transcribe_chunked(_audio(tokens), chunk_s=8.0)
    assert [w.norm for w in words] == tokens


def test_chunk_zero_disables_chunking():
    """chunk_s=0 reproduce el comportamiento anterior (una sola pasada)."""
    tokens = [f"w{i}" for i in range(20)]
    asr = FakeASR(tokens)
    assert asr.transcribe_chunked(_audio(tokens), chunk_s=0) == asr.transcribe(_audio(tokens))


# ------------------------------------------------ opt-in chunk-aligned streaming (Task 3.2, 2026-10-31)
class LoopingASR(FakeASR):
    """Mimics the observed Whisper defect: a single pass over more than 8 s of audio degenerates into a repetition loop
    (measured on AMERESCO clips: ``y que se entramo perfecto`` x15 on a 13 s single pass; the 8 s chunks were fine)."""

    def transcribe(self, audio: np.ndarray) -> list[Word]:
        words = super().transcribe(audio)
        if len(audio) / SAMPLE_RATE > 8.0 + 1e-9 and words:
            last = words[-1]
            words = words + [Word("bucle", "bucle", last.start + 0.1 * i, last.end + 0.1 * i, 0.5, "") for i in range(1, 6)]
        return words


def _stream_aligned(asr, audio, tick_s):
    from mrcd.live.streaming_asr import ChunkAlignedState, StreamingASR

    s, st, keep = StreamingASR(asr=asr), ChunkAlignedState(), int(13 * SAMPLE_RATE)
    dur = len(audio) / SAMPLE_RATE
    ticks = list(np.arange(tick_s, dur, tick_s)) + [dur]
    words = []
    for T in ticks:
        n = int(round(T * SAMPLE_RATE))
        seg = audio[max(n - keep, 0) : n]
        words = s.step_chunk_aligned(st, seg, (n - len(seg)) / SAMPLE_RATE)
    return words, st


@pytest.mark.parametrize("n_tokens,tick", [(5, 1.0), (8, 2.0), (9, 1.0), (14, 2.0), (23, 0.5), (31, 3.0)])
def test_chunk_aligned_streaming_equals_offline_chunking_on_the_last_tick(n_tokens, tick):
    tokens = [f"t{i}" for i in range(n_tokens)]
    tokens[3 % n_tokens] = tokens[2 % n_tokens]  # keep a real adjacent repetition in the data
    asr = FakeASR(tokens)
    audio = _audio(tokens)
    offline = asr.transcribe_chunked(audio, chunk_s=8.0, overlap_s=2.0)
    words, state = _stream_aligned(asr, audio, tick)
    assert [(w.norm, round(w.start, 3)) for w in words] == [(w.norm, round(w.start, 3)) for w in offline]
    assert state.lost_chunks == []


def test_chunk_aligned_avoids_the_single_pass_loop_that_merge_recent_has():
    from mrcd.live.streaming_asr import StreamingASR

    tokens = [f"t{i}" for i in range(13)]
    asr = LoopingASR(tokens)
    audio = _audio(tokens)
    offline = [w.norm for w in asr.transcribe_chunked(audio, chunk_s=8.0, overlap_s=2.0)]
    assert "bucle" not in offline  # the 8 s chunks never exceed the length that triggers the defect
    # default path (unchanged): one 13 s pass -> the loop is in the final text (this is the reproduced defect)
    s = StreamingASR(asr=asr)
    words = s.merge_recent([], audio, 0.0)
    assert sum(w.norm == "bucle" for w in words) >= 3
    # opt-in path: same units, no loop, identical to offline
    aligned, _ = _stream_aligned(asr, audio, 2.0)
    assert [w.norm for w in aligned] == offline


def test_chunk_aligned_records_chunks_lost_to_consumer_lag():
    from mrcd.live.streaming_asr import ChunkAlignedState, StreamingASR

    tokens = [f"t{i}" for i in range(30)]
    asr, audio = FakeASR(tokens), _audio(tokens)
    s, st = StreamingASR(asr=asr), ChunkAlignedState()
    # first tick arrives only at 26 s with a 13 s ring: chunks 0..2 (start 0, 6, 12) already left the ring
    n = 26 * SAMPLE_RATE
    s.step_chunk_aligned(st, audio[n - 13 * SAMPLE_RATE : n], 13.0)
    assert st.lost_chunks[:2] == [0, 1]
