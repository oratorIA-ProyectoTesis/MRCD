"""LiveEngine end to end (fake microphone, cached words, real checkpoint) vs the offline pipeline.

With the offline F0 centre injected, every window the live demo emits must have EXACTLY the offline
probabilities for the same start: the live path's only non-equivalence is the causal centre.
"""

from __future__ import annotations

import json
import types
from pathlib import Path

import numpy as np
import pytest

from mrcd.constants import SAMPLE_RATE
from mrcd.extractors.acoustic import extract_acoustic
from mrcd.extractors.linguistic import Word, word_features
from mrcd.extractors.prosody import prosody_windows, speaker_f0_center
from mrcd.dataset import acoustic_windows, center_f0, linguistic_windows
from mrcd.live.mic_stream import AudioRingBuffer

REPO = Path(__file__).resolve().parents[1]
CKPT = REPO / "models" / "ckpt_audio_text.pt"
CLIP = REPO / "results" / "paridad_20261030" / "extra_audio" / "yD9Igp27VNE_30s_70s.wav"
WORDS = REPO / "results" / "paridad_20261030" / "asr_run1" / "extra_yD9_30_70_asr.json"

pytestmark = pytest.mark.skipif(
    not (CKPT.exists() and CLIP.exists() and WORDS.exists()),
    reason="checkpoint / 40 s clip / cached words not present",
)


class FakeMic:
    def __init__(self):
        self.buffer = AudioRingBuffer(retain_s=60.0)
        self.recorder = None

    def reset(self):
        self.buffer.clear()

    def start(self):
        pass

    def stop(self):
        pass

    def latest_audio(self, seconds=None):
        return self.buffer.latest(seconds)


class FakeASR:
    def __init__(self, words):
        self.words, self._all = [], words

    def transcribe_recent(self, audio, begin):
        return list(self._all)  # same words as the offline reference


def test_ring_buffer_read_since_reports_lost_samples():
    ring = AudioRingBuffer(retain_s=1.0)
    ring.append(np.arange(SAMPLE_RATE * 3, dtype=np.float32))
    audio, start, lost = ring.read_since(0)
    assert (
        len(audio) == SAMPLE_RATE
        and start == 2 * SAMPLE_RATE
        and lost == 2 * SAMPLE_RATE
    )
    audio, start, lost = ring.read_since(2 * SAMPLE_RATE + 100)
    assert (
        start == 2 * SAMPLE_RATE + 100 and lost == 0 and len(audio) == SAMPLE_RATE - 100
    )
    assert audio[0] == 2 * SAMPLE_RATE + 100
    audio, start, lost = ring.read_since(3 * SAMPLE_RATE)
    assert len(audio) == 0 and lost == 0


def test_live_demo_rows_equal_offline_probabilities_with_the_offline_centre(
    monkeypatch,
):
    import soundfile as sf

    import scripts.live_demo as live_demo
    from mrcd.inference import load_ckpt, predict

    audio, _ = sf.read(CLIP, dtype="float32")
    audio = audio[: 24 * SAMPLE_RATE]
    raw = json.loads(WORDS.read_text("utf8"))
    words = [Word(**w) for w in raw if w["end"] <= 24.0]
    frames = extract_acoustic(audio, use_silero=True).frame_matrix()
    c_off = speaker_f0_center(frames)
    net, norm, labels, _ = load_ckpt(CKPT, "cpu")

    mic = FakeMic()
    engine = live_demo.LiveEngine(
        CKPT,
        device="cpu",
        mic=mic,
        asr=FakeASR(words),
        f0_center=c_off,
        config="live_cfg_v1",  # this test pins the historic per-tick rows; v2 is covered in test_live_stream_partitions
    )
    state = engine._begin_session(None)  # resets the mic, no thread: _run is driven synchronously below
    pos = [8000]
    mic.buffer.append(audio[:8000])

    def feed(_seconds):
        if pos[0] >= len(audio):
            engine.stop()
            return
        mic.buffer.append(audio[pos[0] : pos[0] + 8000])
        pos[0] += 8000

    state.wake = types.SimpleNamespace(wait=feed, set=lambda: None)
    engine._run(state)
    rows = engine.session_rows
    assert len(rows) >= 8, "the demo must have emitted several windows"
    stamps = [r["timestamp"] for r in rows]
    assert stamps == sorted(set(stamps)), "each grid window is emitted once, in order"
    assert all(abs(t / 0.5 - round(t / 0.5)) < 1e-9 for t in stamps), (
        "windows are on the 0.5 s grid"
    )
    assert not engine.gaps

    starts = np.asarray(stamps)
    ling, lpos, lmask = linguistic_windows(
        [w.to_dict() for w in words], word_features(words), starts, context=True
    )
    off = {
        "ac": acoustic_windows(center_f0(frames, c_off), starts),
        "pros": prosody_windows(frames, starts, c_off),
        "ling": ling[:, :, :12],
        "lpos": lpos,
        "lmask": lmask,
    }
    p_off = predict(net, norm, off, "cpu", bs=1)
    for r, p in zip(rows, p_off):
        got = np.array([r["todas_las_probabilidades"][l] for l in labels])
        assert np.abs(got - p).max() == 0.0, f"window {r['timestamp']}"
        assert r["clase"] == labels[int(p.argmax())]
