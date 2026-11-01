"""Regresiones de la API de audio sin ASR ni modelos descargados."""
from types import SimpleNamespace
import wave

import numpy as np
import pytest

from mrcd.constants import SAMPLE_RATE, STRIDE_S, WINDOW_S
from mrcd.engine import DisfluencyEngine
from mrcd.extractors.linguistic import Word


def engine(mode="audio_text", ling_dim=12, pros=True):
    net = SimpleNamespace(
        mode=mode, ling=SimpleNamespace(proj=[SimpleNamespace(in_features=ling_dim)]),
        pros=object() if pros else None,
    )
    words = [Word("eh", "eh", 1.0, 1.2, .9, ""),
             Word("seguimos", "seguimos", 5.0, 5.4, .95, ".")]
    asr = SimpleNamespace(transcribe_chunked=lambda *a, **kw: words)
    return DisfluencyEngine(net, {}, ["fluent", "filler_word"], asr), words


@pytest.mark.parametrize("ling_dim,pros", [(12, True), (10, False), (0, False)])
def test_features_and_predictions_match_single_window_path(monkeypatch, ling_dim, pros):
    from mrcd.extractors import acoustic
    from mrcd import inference
    from mrcd.live import window_builder

    eng, words = engine(ling_dim=ling_dim, pros=pros)
    audio = np.zeros(13 * SAMPLE_RATE, np.float32)
    frames = np.random.default_rng(7).normal(size=(1300, 4)).astype(np.float32)
    frames[:, 1] = np.where(np.arange(1300) % 3, 8.0, 0.0)
    frames[:, 3] = (np.arange(1300) % 5 != 0).astype(np.float32)
    calls = []

    def extract(samples, **kw):
        calls.append(samples)
        return SimpleNamespace(frame_matrix=lambda: frames.copy())

    monkeypatch.setattr(acoustic, "extract_acoustic", extract)
    monkeypatch.setattr(window_builder, "extract_acoustic", extract)
    starts = np.arange(0, 13 - WINDOW_S + 1e-9, STRIDE_S)
    expected = [window_builder.build_window(
        audio, words, float(t), context=eng._context,
        include_pros=pros, ling_dim=ling_dim,
    ) for t in starts]
    calls.clear()

    def predict(net, norm, tensors, device, bs):
        assert bs == 1
        assert set(tensors) == set(expected[0])
        for i, single in enumerate(expected):
            for key in single:
                np.testing.assert_array_equal(tensors[key][i:i + 1], single[key])
        return np.tile([.25, .75], (len(starts), 1))

    monkeypatch.setattr(inference, "predict", predict)
    compact = eng.analyze(audio)
    assert len(calls) == 1
    detailed = eng.analyze(audio, keep_windows=True)
    assert len(calls) == 2
    assert compact.events == detailed.events
    assert compact.events[0].text == "eh"
    assert compact.events[0].confidence == .75
    assert not compact.windows
    assert len(detailed.windows) == len(starts)


@pytest.mark.parametrize("audio", [np.zeros((16000, 2)), np.zeros((1, 16000)),
                                    np.array([np.nan]), np.array([np.inf])])
def test_invalid_audio_rejected_before_asr(audio):
    eng, _ = engine()
    eng._asr = None
    with pytest.raises(ValueError, match="mono|finitas"):
        eng.analyze(audio)


@pytest.mark.parametrize("mode", ["trimodal", "kinesic_only"])
def test_visual_modes_rejected_before_asr(mode):
    eng, _ = engine(mode=mode)
    eng._asr = None
    with pytest.raises(ValueError, match="evidencia visual"):
        eng.analyze(np.zeros(13 * SAMPLE_RATE, np.float32))


@pytest.mark.parametrize("width", [1, 3, 4])
def test_non_pcm16_wav_rejected(tmp_path, width):
    path = tmp_path / "audio.wav"
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(width)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(bytes(width * SAMPLE_RATE))
    eng, _ = engine()
    with pytest.raises(ValueError, match="PCM de 16 bits"):
        eng.analyze_file(path)


def test_float_wav_rejected_explicitly(tmp_path):
    import struct

    path = tmp_path / "float.wav"
    fmt = struct.pack("<HHIIHH", 3, 1, SAMPLE_RATE, SAMPLE_RATE * 4, 4, 32)
    data = bytes(4 * SAMPLE_RATE)
    body = b"WAVEfmt " + struct.pack("<I", len(fmt)) + fmt
    body += b"data" + struct.pack("<I", len(data)) + data
    path.write_bytes(b"RIFF" + struct.pack("<I", len(body)) + body)
    eng, _ = engine()
    with pytest.raises(ValueError, match="PCM de 16 bits"):
        eng.analyze_file(path)
