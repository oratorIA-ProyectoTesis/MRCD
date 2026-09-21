from __future__ import annotations

import asyncio

import numpy as np
import pytest

from core.constants import SAMPLE_RATE
from core.dataset import acoustic_windows, center_f0, linguistic_windows
from core.extractors.acoustic import extract_acoustic
from core.extractors.linguistic import Word, word_features
from core.extractors.prosody import prosody_windows, speaker_f0_center
from core.live.mic_stream import AudioRingBuffer
from core.live.streaming_asr import dedupe_words
from core.live.window_builder import build_window


def _words() -> list[Word]:
    return [Word("bueno", "bueno", 7.0, 7.3, .9, ","), Word("eh", "eh", 8.0, 8.2, .8, ""),
            Word("seguimos", "seguimos", 9.0, 9.4, .95, ".")]


def test_streaming_window_matches_offline_feature_by_feature():
    seconds = 16
    time = np.arange(seconds * SAMPLE_RATE) / SAMPLE_RATE
    audio = (.15 * np.sin(2 * np.pi * 180 * time)).astype(np.float32)
    ring = AudioRingBuffer(retain_s=seconds)
    for begin in range(0, len(audio), SAMPLE_RATE // 2):
        ring.append(audio[begin:begin + SAMPLE_RATE // 2])
    simulated, _ = ring.latest()
    words = _words()
    start = 8.0
    streamed = build_window(simulated, words, start, use_silero=False)
    track = extract_acoustic(audio, use_silero=False)
    frames = track.frame_matrix(); center = speaker_f0_center(frames)
    word_dicts = [word.to_dict() for word in words]
    features = word_features(words)
    ling, lpos, lmask = linguistic_windows(word_dicts, features, np.array([start]))
    offline = {"pros": prosody_windows(frames, np.array([start]), center),
               "ac": acoustic_windows(center_f0(frames, center), np.array([start])),
               "ling": ling, "lpos": lpos, "lmask": lmask}
    for key in offline:
        np.testing.assert_allclose(streamed[key], offline[key], rtol=1e-4, atol=1e-5)


def test_asr_overlap_deduplicates_absolute_words():
    first = [Word("bueno", "bueno", 10.0, 10.3, .8, ""), Word("eh", "eh", 10.4, 10.6, .8, "")]
    retry = [Word("bueno", "bueno", 10.02, 10.31, .9, ""), Word("eh", "eh", 10.39, 10.61, .8, ""),
             Word("vamos", "vamos", 10.7, 11.0, .9, "")]
    merged = dedupe_words(first, retry)
    assert [word.norm for word in merged] == ["bueno", "eh", "vamos"]
    assert merged[0].prob == .9


def test_legacy_audio_text_contract_uses_ten_linguistic_features_without_prosody():
    audio = np.zeros(4 * SAMPLE_RATE, np.float32)
    tensors = build_window(audio, _words(), 0.0, use_silero=False, context=False,
                           include_pros=False, ling_dim=10)
    assert tensors["ling"].shape[-1] == 10
    assert "pros" not in tensors


def test_websocket_emits_prediction_schema():
    fastapi = pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from scripts.live_demo import create_app

    class FakeEngine:
        def __init__(self): self.queue = asyncio.Queue()
        def start(self, loop): loop.call_soon_threadsafe(self.queue.put_nowait, {
            "timestamp": 1.5, "texto_ventana": "eh bueno", "clase": "filler_word", "confianza": .8,
            "todas_las_probabilidades": {"fluent": .2, "filler_word": .8}})
        def stop(self): pass

    with TestClient(create_app(FakeEngine())) as client:
        with client.websocket_connect("/ws") as ws:
            message = ws.receive_json()
    assert set(message) == {"timestamp", "texto_ventana", "clase", "confianza", "todas_las_probabilidades"}
    assert message["clase"] == "filler_word"


def test_live_controls_start_and_stop_the_engine():
    from fastapi.testclient import TestClient
    from scripts.live_demo import create_app

    class FakeEngine:
        def __init__(self): self.queue, self.starts, self.stops = asyncio.Queue(), 0, 0
        def start(self, loop): self.starts += 1
        def stop(self): self.stops += 1

    engine = FakeEngine()
    with TestClient(create_app(engine)) as client:
        assert client.post("/control/start").json() == {"running": True}
        assert client.post("/control/stop").json() == {"running": False}
    assert engine.starts == 1 and engine.stops == 1


def test_live_page_exposes_session_analysis_panels():
    from scripts.live_demo import PAGE

    for element_id in ("metrics", "probabilities", "distribution", "events"):
        assert f'id={element_id}' in PAGE
