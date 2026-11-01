from __future__ import annotations

import asyncio

import numpy as np
import pytest

from mrcd.constants import SAMPLE_RATE
from mrcd.dataset import acoustic_windows, center_f0, linguistic_windows
from mrcd.extractors.acoustic import extract_acoustic
from mrcd.extractors.linguistic import Word, word_features
from mrcd.extractors.prosody import prosody_windows, speaker_f0_center
from mrcd.live.mic_stream import AudioRingBuffer
from mrcd.live.streaming_asr import dedupe_words
from mrcd.live.window_builder import build_window


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
        assert client.post("/control/start").json() == {"running": True}
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


# --------------------------------------------- grabacion de sesion para auditoria
def test_session_recorder_keeps_whole_session_not_just_the_ring_buffer():
    """El ring buffer retiene 13 s; la auditoria necesita la sesion entera.

    Sin esto no se puede distinguir «el modelo no detecto la disfluencia» de
    «el ASR la borro antes de que el modelo la viera».
    """
    import numpy as np

    from mrcd.constants import CONTEXT_S, SAMPLE_RATE, WINDOW_S
    from mrcd.live.mic_stream import MicrophoneStream

    mic = MicrophoneStream(record=True)
    total_s = (CONTEXT_S + WINDOW_S) * 3  # el triple de lo que retiene el buffer
    chunk = np.zeros(SAMPLE_RATE, np.float32)
    for i in range(int(total_s)):
        chunk[:] = (i % 7) / 10.0
        mic.buffer.append(chunk)
        mic.recorder.append(chunk)

    assert mic.buffer.duration_s == pytest.approx(total_s, abs=0.1)
    assert mic.recorder.duration_s == pytest.approx(total_s, abs=0.1)
    # el buffer solo CONSERVA la ventana de contexto; el grabador, todo
    retained, _ = mic.buffer.latest()
    assert len(retained) / SAMPLE_RATE <= CONTEXT_S + WINDOW_S + 1
    assert len(mic.recorder.samples()) / SAMPLE_RATE == pytest.approx(total_s, abs=0.1)


def test_recorder_emits_valid_wav_and_clips_overflow():
    import io
    import wave

    import numpy as np

    from mrcd.constants import SAMPLE_RATE
    from mrcd.live.mic_stream import SessionRecorder

    rec = SessionRecorder()
    rec.append(np.array([0.0, 0.5, -0.5, 3.0, -3.0], np.float32))  # dos muestras desbordan
    data = rec.to_wav_bytes()
    with wave.open(io.BytesIO(data), "rb") as w:
        assert w.getnchannels() == 1 and w.getsampwidth() == 2
        assert w.getframerate() == SAMPLE_RATE
        pcm = np.frombuffer(w.readframes(w.getnframes()), np.int16)
    assert len(pcm) == 5
    # recortado, no envuelto: un desbordamiento sonaria a chasquido y falsearia la evidencia
    assert pcm[3] == 32767 and pcm[4] == -32767


def test_recorder_clears_on_reset():
    import numpy as np

    from mrcd.live.mic_stream import MicrophoneStream

    mic = MicrophoneStream(record=True)
    mic.recorder.append(np.ones(1000, np.float32))
    assert mic.recorder.duration_s > 0
    mic.reset()
    assert mic.recorder.duration_s == 0.0, "una sesion nueva no debe arrastrar audio de la anterior"


def _upsert_js():
    import re

    from scripts.live_demo import PAGE

    m = re.search(r"// UPSERT-BEGIN.*?\n(function upsert.*?)\r?\n// UPSERT-END", PAGE, re.S)
    assert m, "the dashboard must keep the upsert block"
    return m.group(1)


def _run_upsert(rows_in):
    import json
    import shutil
    import subprocess

    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not available to execute the dashboard JS")
    prog = _upsert_js() + (
        f"\nconst rows=[];const added=[];for(const x of {json.dumps(rows_in)}){{added.push(upsert(rows,x))}}"
        "\nconsole.log(JSON.stringify({rows,added}))"
    )
    out = subprocess.run([node, "-e", prog], capture_output=True, text=True, check=True).stdout
    return json.loads(out)


def test_dashboard_counts_each_window_once_when_the_stop_pass_revises_rows():
    prov = [
        {"timestamp": t, "status": "provisional", "clase": "none", "confianza": 0.5} for t in (0.0, 0.5, 1.0)
    ]
    revised = [
        {"timestamp": t, "status": "final", "revises": t, "clase": "block", "confianza": 0.9} for t in (0.0, 0.5, 1.0)
    ]
    tail = {"timestamp": 1.5, "status": "final", "clase": "none", "confianza": 0.7}  # never had a provisional row
    res = _run_upsert(prov + revised + [tail])
    ts = [r["timestamp"] for r in res["rows"]]
    assert ts == [0.0, 0.5, 1.0, 1.5]  # each window once, original order
    assert [r["status"] for r in res["rows"]] == ["final"] * 4
    assert [r["clase"] for r in res["rows"]][:3] == ["block"] * 3  # the revision replaced the row
    assert res["added"] == [True] * 3 + [False] * 3 + [True]  # only new windows extend the transcript


def test_dashboard_v1_rows_without_status_are_all_kept():
    rows = [{"timestamp": t, "clase": "none", "confianza": 0.5} for t in (0.0, 0.5, 1.0)]
    assert len(_run_upsert(rows)["rows"]) == 3
