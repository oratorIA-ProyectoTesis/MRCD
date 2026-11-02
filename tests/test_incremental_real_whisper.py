"""Real Whisper (p0, CPU int8): decoding each 8 s fragment on its own, out of order and from bytes received in parts
equals the offline ``transcribe_chunked`` bit for bit (DESIGN_B_INCREMENTAL cases 1-10 at engine level).

Slow: set MRCD_SLOW=1 and OMP_NUM_THREADS=4 before Python starts. The whole-service comparison with real models is
``scripts/s8_incremental/equivalence_real.py``.
"""

from __future__ import annotations

import os
import wave
from pathlib import Path

import numpy as np
import pytest

from mrcd.service.incremental import plan
from mrcd.service.incremental.cache import ChunkCache, segment_key
from mrcd.service.incremental.cached_asr import CachedASR, asr_namespace, decode_with

REPO = Path(__file__).resolve().parents[1]
C60 = REPO / "results/cierre_oe1_oe3_20261031/3b/cost/inputs/c60s_RkP96R.wav"

pytestmark = [
    pytest.mark.timeout(1800),
    pytest.mark.skipif(
    os.environ.get("MRCD_SLOW") != "1"
    or os.environ.get("OMP_NUM_THREADS") != "4"
    or not C60.exists(),
    reason="set MRCD_SLOW=1 and OMP_NUM_THREADS=4 before python starts; needs the cached Whisper small",
),
]


@pytest.fixture(scope="module")
def asr():
    from mrcd.extractors.linguistic import VerbatimASR

    return VerbatimASR(device="cpu")


def pcm_float(seconds: float) -> np.ndarray:
    with wave.open(str(C60)) as w:
        raw = w.readframes(w.getnframes())
    n = int(round(seconds * 16000))
    return np.frombuffer(raw[: n * 2], dtype="<i2").astype(np.float32) / 32768.0


def words(ws):
    return [
        (w.text, w.norm, repr(w.start), repr(w.end), repr(w.prob), w.punct_after)
        for w in ws
    ]


@pytest.mark.parametrize("seconds", [7.5, 8.0, 13.0, 14.0, 31.0, 36.999, 37.0])
def test_cached_chunked_decode_equals_offline_with_real_whisper(asr, seconds):
    audio = pcm_float(seconds)
    offline = asr.transcribe_chunked(audio, chunk_s=8.0)
    cache, ns = ChunkCache(), asr_namespace(asr)
    # decode only the complete fragments, in reverse order, as if they had been received earlier
    for _k, start in reversed(plan.complete_chunks(len(audio))):
        seg = audio[start : start + plan.CHUNK_SAMPLES]
        cache.put(segment_key(ns, start, seg), *decode_with(asr, seg, start))
    cached = CachedASR(asr, cache, ns)
    assert words(cached.transcribe_chunked(audio, chunk_s=8.0)) == words(offline)


def test_torch_thread_count_depends_on_load_order_which_is_why_startup_order_is_fixed():
    """Reproduces the finding of DESIGN_B_INCREMENTAL 1.A.1: importing Silero sets torch to 1 thread and every new
    WhisperModel puts the shared pool back to 4, so the value depends on the order engines were loaded in."""
    import subprocess
    import sys

    code = (
        "import torch\n"
        "a = torch.get_num_threads()\n"
        "import silero_vad\n"
        "b = torch.get_num_threads()\n"
        "from faster_whisper import WhisperModel\n"
        "WhisperModel('small', device='cpu', compute_type='int8')\n"
        "c = torch.get_num_threads()\n"
        "print(a, b, c)\n"
    )
    env = {**os.environ, "HF_HUB_OFFLINE": "1"}
    out = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, env=env, cwd=REPO, timeout=300)
    start, after_silero, after_whisper = (int(x) for x in out.stdout.split()[-3:])
    assert (start, after_silero, after_whisper) == (4, 1, 4), out.stdout + out.stderr


def _tiled(start_s: float, piece_s: float, seconds: int = 8) -> np.ndarray:
    """A short piece of the repo's own clip repeated until it fills ``seconds``: real Whisper loops on it."""
    audio = pcm_float(60.0)
    piece = audio[int(start_s * 16000) : int((start_s + piece_s) * 16000)]
    return np.tile(piece, int(seconds / piece_s) + 1)[: seconds * 16000]


@pytest.mark.parametrize("profile", ["p0_noloop", "p1_noloop"])
def test_real_whisper_loop_redecode_divergence_is_always_a_flagged_sampled_rung(profile):
    """H8 (S8 review). Chunks that REALLY loop (pieces of the repo clip tiled to 8 s) run the re-decode ladder.

    Measured: with ``p0_noloop`` the early-decoded + cached assembly equals the offline one bit for bit; with
    ``p1_noloop`` an early decode of one chunk was REPLACED by the sampled rung 2 (T=0.4) while the offline decode of
    the same audio and seed was not, so the documents differ. That rung is declared not bit-reproducible
    (``reproducible: False``). Hence the service refuses ``*_noloop`` profiles in incremental sessions (422
    ``profile_not_supported_incremental``); this test pins the claim underneath: whenever the two assemblies differ,
    a sampled-rung event explains it, and when no such event exists they are identical."""
    from mrcd.extractors.linguistic import VerbatimASR

    asr = VerbatimASR(device="cpu", profile=profile)
    audio = np.concatenate([_tiled(2, 0.8), _tiled(14, 0.8), _tiled(9, 0.8), _tiled(5, 1.2)])
    offline = asr.transcribe_chunked(audio, chunk_s=8.0)
    events = [dict(e) for e in asr.redecode_events]
    assert events, "the fixture must make Whisper loop so the ladder runs"
    cache, ns = ChunkCache(), asr_namespace(asr)
    for _k, start in reversed(plan.complete_chunks(len(audio))):
        seg = audio[start : start + plan.CHUNK_SAMPLES]
        chunk_words, chunk_events = decode_with(asr, seg, start)
        events += chunk_events
        cache.put(segment_key(ns, start, seg), chunk_words, chunk_events)
    cached = CachedASR(asr, cache, ns)
    same = words(cached.transcribe_chunked(audio, chunk_s=8.0)) == words(offline)
    sampled_replacement = any(
        e.get("attempt") is not None and e.get("reproducible") is False for e in events
    )
    assert same or sampled_replacement
    if not sampled_replacement:
        assert same
