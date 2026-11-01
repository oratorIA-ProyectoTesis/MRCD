"""Live path vs offline whole-file extraction (SessionFeatureTracker).

Declared tolerance (see results/paridad_live_20261030/PARIDAD_LIVE.md):

* acoustic frames (log-RMS, F0, voiced prob, VAD, dF0): EXACT (array_equal) for every committed frame;
* tensors with the offline F0 centre injected: EXACT (ac, pros, ling, lpos, lmask);
* tensors with the causal centre: every channel except centred F0 is exact, and centred F0 differs by
  exactly the centre difference (|d| <= |c_live - c_offline|), so the divergence is bounded and
  explained, not "small by luck".
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from mrcd.constants import SAMPLE_RATE, WINDOW_S
from mrcd.dataset import acoustic_windows, center_f0, linguistic_windows
from mrcd.extractors.acoustic import extract_acoustic
from mrcd.extractors.linguistic import Word, word_features
from mrcd.extractors.prosody import prosody_windows, speaker_f0_center
from mrcd.live.session_features import (
    IncrementalSegmenter,
    SessionFeatureTracker,
    WindowNotReady,
)

REPO = Path(__file__).resolve().parents[1]
AMERESCO = REPO / "results" / "baselines_20261030" / "ameresco"
ASR_CACHE = REPO / "results" / "mejoras_20261030" / "asr_cache" / "ameresco"
CKPT = REPO / "models" / "ckpt_audio_text.pt"
real_data = pytest.mark.skipif(
    not (AMERESCO / "manifest.json").exists() or not ASR_CACHE.exists(),
    reason="AMERESCO clips / frozen ASR cache not present",
)


def speech_like(seconds: float = 17.0, seed: int = 3) -> np.ndarray:
    """Harmonic 'voice' with gliding F0 and several pauses (so Silero produces real segments)."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    f0 = 130 + 40 * np.sin(2 * np.pi * 0.31 * t)
    ph = 2 * np.pi * np.cumsum(f0) / SAMPLE_RATE
    voice = sum((0.6 / k) * np.sin(k * ph) for k in range(1, 6))
    on = (np.sin(2 * np.pi * 0.23 * t) > -0.25).astype(float)
    on = np.convolve(on, np.ones(800) / 800, mode="same")
    return (0.25 * on * voice + 0.002 * rng.standard_normal(len(t))).astype(np.float32)


WORDS = [
    Word("bueno", "bueno", 1.0, 1.4, 0.9, ","),
    Word("eh", "eh", 3.0, 3.3, 0.8, ""),
    Word("seguimos", "seguimos", 5.2, 5.9, 0.95, "."),
    Word("entonces", "entonces", 8.0, 8.6, 0.9, ""),
    Word("este", "este", 10.0, 10.3, 0.7, ""),
    Word("de", "de", 12.0, 12.2, 0.9, ""),
]


def replay(audio, tracker, chunk=7919):
    """Feed the audio like a microphone (irregular chunks, not aligned to 512/160 samples)."""
    for i in range(0, len(audio), chunk):
        tracker.append(audio[i : i + chunk])
    return tracker


def offline_tensors(audio, words, starts, ling_dim=12):
    frames = extract_acoustic(audio, use_silero=True).frame_matrix()
    c = speaker_f0_center(frames)
    ling, lpos, lmask = linguistic_windows(
        [w.to_dict() for w in words], word_features(words), starts, context=True
    )
    return (
        frames,
        c,
        {
            "ac": acoustic_windows(center_f0(frames, c), starts),
            "pros": prosody_windows(frames, starts, c),
            "ling": ling[:, :, :ling_dim],
            "lpos": lpos,
            "lmask": lmask,
        },
    )


def grid_starts(tracker):
    last = int(np.floor((tracker.committed_s - WINDOW_S) / 0.5 + 1e-9))
    return np.arange(0, last + 1) * 0.5


# ------------------------------------------------------------------ segmenter (no model)
def test_incremental_segmenter_equals_one_shot_segmentation():
    """Freezing closed segments must not change the result (random speech/silence probability tracks)."""
    from silero_vad.utils_vad import get_speech_timestamps_from_probs

    rng = np.random.default_rng(7)
    for trial in range(6):
        runs, probs = [], []
        while len(probs) < 4000:
            speech = rng.random() < 0.5
            n = int(rng.integers(3, 150 if speech else 120))
            base = rng.uniform(0.6, 0.98) if speech else rng.uniform(0.0, 0.3)
            probs += list(np.clip(base + 0.15 * rng.standard_normal(n), 0, 1))
            runs.append(n)
        seg = IncrementalSegmenter()
        i = 0
        while i < len(probs):
            step = int(rng.integers(1, 40))
            seg.extend(probs[i : i + step])
            got = seg.segments_seconds()  # called every tick, as the tracker does
            i += step
        total = len(probs) * 512 / SAMPLE_RATE
        ref = get_speech_timestamps_from_probs(
            probs,
            sampling_rate=SAMPLE_RATE,
            return_seconds=True,
            min_silence_duration_ms=150,
            min_speech_duration_ms=120,
            speech_pad_ms=30,
        )
        assert got == [(s["start"], s["end"]) for s in ref], f"trial {trial}"
        assert total > 0


# ------------------------------------------------------------------ synthetic audio
@pytest.fixture(scope="module")
def synthetic():
    audio = speech_like()
    tracker = replay(audio, SessionFeatureTracker(f0_center="causal"))
    frames, c_off, off = offline_tensors(audio, WORDS, np.arange(0, 17 - 3 + 1e-9, 0.5))
    return audio, tracker, frames, c_off, off


def test_committed_frames_are_bit_identical_to_offline(synthetic):
    _, tracker, frames, _, _ = synthetic
    live = tracker.frame_matrix()
    assert len(live) > 1300
    np.testing.assert_array_equal(live, frames[: len(live)])


def test_commit_lags_the_audio_by_the_right_margin(synthetic):
    audio, tracker, *_ = synthetic
    assert tracker.duration_s - tracker.committed_s <= 3.0 + 0.02


def test_tensors_match_offline_exactly_with_the_offline_centre(synthetic):
    audio, _, frames, c_off, off = synthetic
    tracker = replay(audio, SessionFeatureTracker(f0_center=c_off))
    starts = grid_starts(tracker)
    assert len(starts) >= 20
    for k, s in enumerate(starts):
        got = tracker.window(float(s), WORDS, ling_dim=12)
        for key in off:
            np.testing.assert_array_equal(
                got[key][0], off[key][int(round(s / 0.5))], err_msg=f"{key}@{s}"
            )


def test_causal_centre_divergence_is_exactly_the_centre_shift(synthetic):
    """Only centred F0 (ac col 1) and pros out[4] move; ac shifts by the centre difference."""
    _, tracker, frames, c_off, off = synthetic
    c_live = tracker.f0_center_st()
    for s in grid_starts(tracker)[::3]:
        got = tracker.window(float(s), WORDS, ling_dim=12)
        i = int(round(s / 0.5))
        for key in ("ling", "lpos", "lmask"):
            np.testing.assert_array_equal(got[key][0], off[key][i])
        diff = np.abs(got["ac"][0].astype(np.float64) - off["ac"][i])
        assert diff[:, [0, 2, 3, 4]].max() == 0.0
        assert diff[:, 1].max() <= abs(c_live - c_off) + 1e-4
        assert (
            np.abs(got["pros"][0].astype(np.float64) - off["pros"][i]).max()
            <= abs(c_live - c_off) + 1e-4
        )


def test_window_not_ready_raises(synthetic):
    _, tracker, *_ = synthetic
    with pytest.raises(WindowNotReady):
        tracker.window(tracker.committed_s - 1.0, WORDS)


def test_latest_start_is_on_the_grid_and_ready(synthetic):
    _, tracker, *_ = synthetic
    s = tracker.latest_start()
    assert s is not None and abs(s / 0.5 - round(s / 0.5)) < 1e-9
    assert s + WINDOW_S <= tracker.committed_s + 1e-9
    assert (
        s <= tracker.duration_s - 5.0 + 1e-9
    )  # the 10 s linguistic context is complete


def test_margins_below_the_measured_minimum_are_rejected():
    with pytest.raises(ValueError):
        SessionFeatureTracker(left_margin_s=0.1)
    with pytest.raises(ValueError):
        SessionFeatureTracker(right_margin_s=0.5)


def test_local_words_equal_full_list_including_non_monotone_asr_timestamps():
    """Trimming the word list must not change linguistic_windows, also when ASR times go backwards."""
    from mrcd.live.session_features import _local_words

    rng = np.random.default_rng(1)
    t, words = 0.0, []
    for i in range(60):
        t += float(rng.uniform(0.1, 0.5))
        words.append(Word(f"w{i}", f"w{i}", t, t + float(rng.uniform(0.1, 0.4)), 0.9, ""))
    words[30] = Word("x", "x", words[29].start - 0.05, words[29].end - 0.02, 0.5, "")  # out of order
    for ws in (words[:30], words):  # sorted subset and the non-monotone list
        starts = np.arange(0.0, 25.0, 0.5)
        full = linguistic_windows([w.to_dict() for w in ws], word_features(ws), starts, context=True)
        for k, s in enumerate(starts):
            loc = _local_words(ws, float(s), True)
            part = linguistic_windows([w.to_dict() for w in loc], word_features(loc), np.array([s]), context=True)
            for a, b in zip(full, part):
                np.testing.assert_array_equal(a[k], b[0], err_msg=f"start {s}")


def test_growing_centre_of_the_live_loop_stays_within_the_centre_difference(synthetic):
    """Replay with 0.5 s ticks; each window is scored at the tick it becomes ready with the centre known
    THEN (not the final one). Every channel but centred F0 is exact; centred F0 moves by <= |c_now - c_off|."""
    audio, _, frames, c_off, off = synthetic
    tracker = SessionFeatureTracker(f0_center="causal")
    done, centres = set(), set()
    for i in range(0, len(audio), 8000):
        tracker.append(audio[i : i + 8000])
        latest = tracker.latest_start()
        if latest is None:
            continue
        c_now = tracker.f0_center_st()
        for s in np.arange(0.0, latest + 1e-9, 0.5):
            k = int(round(s / 0.5))
            if k in done:
                continue
            done.add(k)
            centres.add(round(c_now, 6))
            got = tracker.window(float(s), WORDS, ling_dim=12)
            for key in ("ling", "lpos", "lmask"):
                np.testing.assert_array_equal(got[key][0], off[key][k])
            d = np.abs(got["ac"][0].astype(np.float64) - off["ac"][k])
            assert d[:, [0, 2, 3, 4]].max() == 0.0
            assert d[:, 1].max() <= abs(c_now - c_off) + 1e-4, f"window {s}"
    assert len(done) >= 15 and len(centres) > 1, "the centre must actually grow during the replay"


# ------------------------------------------------------------------ real clips (frozen data)
def _clips(n):
    if not (AMERESCO / "manifest.json").exists():
        return []  # third-party clips are not redistributed: dependent tests skip cleanly
    man = json.loads((AMERESCO / "manifest.json").read_text("utf8"))
    return [m for m in man if m["in_pilot32"]][:n]


@real_data
@pytest.mark.parametrize("m", _clips(3), ids=lambda m: m["unit_id"])
def test_real_clip_frames_and_tensors_match_offline(m):
    import soundfile as sf

    audio, sr = sf.read(m["wav"], dtype="float32")
    assert sr == SAMPLE_RATE
    words = [
        Word(**w)
        for w in json.loads((ASR_CACHE / f"{m['unit_id']}.json").read_text("utf8"))[
            "words"
        ]
    ]
    starts = np.arange(0, len(audio) / sr - 3 + 1e-9, 0.5)
    frames, c_off, off = offline_tensors(audio, words, starts)
    tracker = replay(audio, SessionFeatureTracker(f0_center=c_off), chunk=8000)
    live = tracker.frame_matrix()
    np.testing.assert_array_equal(live, frames[: len(live)])
    for s in grid_starts(tracker):
        got = tracker.window(float(s), words, ling_dim=12)
        for key in off:
            np.testing.assert_array_equal(
                got[key][0], off[key][int(round(s / 0.5))], err_msg=f"{key}@{s}"
            )


@real_data
@pytest.mark.skipif(not CKPT.exists(), reason="base checkpoint not present")
def test_real_clip_probabilities_and_argmax_match_offline():
    import soundfile as sf

    from mrcd.inference import load_ckpt, predict

    net, norm, labels, _ = load_ckpt(CKPT, "cpu")
    m = _clips(1)[0]
    audio, _ = sf.read(m["wav"], dtype="float32")
    words = [
        Word(**w)
        for w in json.loads((ASR_CACHE / f"{m['unit_id']}.json").read_text("utf8"))[
            "words"
        ]
    ]
    starts = np.arange(0, len(audio) / SAMPLE_RATE - 3 + 1e-9, 0.5)
    _, c_off, off = offline_tensors(audio, words, starts)
    p_off = predict(net, norm, off, "cpu", bs=1)
    tracker = replay(audio, SessionFeatureTracker(f0_center=c_off), chunk=8000)
    ss = grid_starts(tracker)
    live = [tracker.window(float(s), words, ling_dim=12) for s in ss]
    t = {k: np.concatenate([w[k] for w in live]) for k in live[0]}
    p_live = predict(net, norm, t, "cpu", bs=1)
    idx = [int(round(s / 0.5)) for s in ss]
    assert np.abs(p_live - p_off[idx]).max() == 0.0
    assert (p_live.argmax(1) == p_off[idx].argmax(1)).all()


@real_data
def test_heldout_clip_a24_frames_exact_with_default_margins():
    """BAQ_001_03_16_a24 failed (36 frames, 9.22-9.57 s) with a 1 s LEFT margin: Viterbi dependence is two-sided."""
    import soundfile as sf

    m = next(x for x in json.loads((AMERESCO / "manifest.json").read_text("utf8")) if x["unit_id"] == "BAQ_001_03_16_a24")
    audio, _ = sf.read(m["wav"], dtype="float32")
    frames = extract_acoustic(audio, use_silero=True).frame_matrix()
    tracker = replay(audio, SessionFeatureTracker(f0_center=0.0), chunk=8000)
    live = tracker.frame_matrix()
    np.testing.assert_array_equal(live, frames[: len(live)])
    assert len(live) > 900


def test_defaults_are_the_margins_with_zero_differing_frames_and_smaller_ones_warn():
    t = SessionFeatureTracker()
    assert t._left == 300 and t._right == 300  # 3 s each (10 ms frames)
    with pytest.warns(RuntimeWarning, match="NOT frame-exact"):
        SessionFeatureTracker(left_margin_s=1.0)
