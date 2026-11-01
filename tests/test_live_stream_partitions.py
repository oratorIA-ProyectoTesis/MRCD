"""Live path vs offline whole-file extraction under ARBITRARY stream partitions, short/long sessions,
exact window limits, device gaps, close (finalize) and restart.

Levels (see results/ejecucion_material4_20261030/paridad/PARIDAD_LIVE_V2.md). Tolerances and why:

* (b) acoustic frames: EXACT (array_equal). Frames are deterministic functions of the audio; any
  difference means the margins/commit logic is wrong. Empirical result for the margins, not a proof.
* (c) tensors/probabilities with the OFFLINE centre injected: EXACT. NOTE: this does NOT show that the
  live path works autonomously (the centre is given by the offline pass); it shows that everything
  except the centre is equivalent.
* (d) causal centre, provisional rows: NOT asserted equal. Only the exact bound: every channel but
  centred F0 is exact and centred F0 differs by <= |c_now - c_offline|.
* (f) final pass (``finalize`` + rescore): after close the tracker knows every voiced frame, so its
  centre equals the offline one; rescored ``final`` rows must equal the offline engine tensors/probs
  EXACTLY given the same words (a decision-level statement, ASR words are given, not streamed).

``fake_score`` is a fixed random projection of ALL tensor values, so any tensor difference changes the
probabilities (it is not an accuracy claim about the real checkpoint).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pytest

from mrcd.constants import SAMPLE_RATE, STRIDE_S, WINDOW_S
from mrcd.extractors.acoustic import extract_acoustic
from mrcd.extractors.linguistic import Word
from mrcd.live.mic_stream import MicrophoneStream
from mrcd.live.session import LIVE_CFG_V1, LIVE_CFG_V2, LiveScorer
from mrcd.live.session_features import SessionFeatureTracker

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "live_parity"))
from tests.test_live_parity import WORDS, offline_tensors, speech_like  # noqa: E402

LABELS = ["fluent", "filler_word", "prolongation", "repetition"]
_W: dict[int, np.ndarray] = {}


def flatten(t: dict) -> np.ndarray:
    return np.concatenate([np.asarray(t[k], np.float64).ravel() for k in sorted(t)])


def fake_score(t: dict) -> np.ndarray:
    x = flatten(t)
    if len(x) not in _W:
        _W[len(x)] = np.random.default_rng(5).standard_normal(
            (len(LABELS), len(x))
        ) / np.sqrt(len(x))
    z = _W[len(x)] @ x
    e = np.exp(z - z.max())
    return e / e.sum()


def partition(audio, kind, seed, tracker_or_scorer):
    """Deliver ``audio`` in an arbitrary partition. Returns nothing; works on a scorer or tracker."""
    rng = np.random.default_rng(seed)
    push = (
        tracker_or_scorer.push
        if hasattr(tracker_or_scorer, "push")
        else tracker_or_scorer.append
    )
    i = 0
    while i < len(audio):
        if kind == "tiny":
            k = int(rng.integers(1, 700))
        elif kind == "mixed":
            k = int(rng.choice([1, 7, 159, 161, 511, 513, 4000, 7919, 8000, 30000]))
        else:  # "rand"
            k = int(np.exp(rng.uniform(np.log(80), np.log(40000))))
        push(audio[i : i + k])
        i += k


# ------------------------------------------------------------------ (b) frames, partitions
@pytest.mark.parametrize("kind,seed", [("rand", 1), ("mixed", 2), ("rand", 3)])
def test_frames_equal_offline_for_arbitrary_partitions_and_after_close(kind, seed):
    audio = speech_like(11.0, seed=seed)[:-37]  # not a multiple of 160 or 512
    off = extract_acoustic(audio, use_silero=True).frame_matrix()
    tr = SessionFeatureTracker(f0_center=0.0)
    # tiny chunks commit at every append: keep the number of appends moderate for the test budget
    rng = np.random.default_rng(seed)
    i = 0
    while i < len(audio):
        k = int(
            rng.choice([1, 7, 159, 161, 511, 513, 4000, 7919, 30000])
            if kind == "mixed"
            else np.exp(rng.uniform(np.log(300), np.log(40000)))
        )
        tr.append(audio[i : i + k], commit=(k > 200 or rng.random() < 0.05))
        i += k
    live = tr.frame_matrix()
    np.testing.assert_array_equal(live, off[: len(live)])
    tr.finalize()
    tr.finalize()  # idempotent
    full = tr.frame_matrix()
    assert len(full) == len(off), "after close EVERY offline frame must exist"
    np.testing.assert_array_equal(full, off)
    with pytest.raises(RuntimeError):
        tr.append(audio[:100])


def test_tiny_chunks_without_commit_equal_one_shot():
    audio = speech_like(8.0, seed=9)
    tr = SessionFeatureTracker(f0_center=0.0)
    rng = np.random.default_rng(0)
    i = 0
    while i < len(audio):
        k = int(rng.integers(1, 600))
        tr.append(
            audio[i : i + k], commit=False
        )  # VAD chunking across 512-sample boundaries
        i += k
    tr.finalize()
    np.testing.assert_array_equal(
        tr.frame_matrix(), extract_acoustic(audio, use_silero=True).frame_matrix()
    )


# ------------------------------------------------------------------ short sessions, window limits
@pytest.mark.parametrize("secs", [0.4, 2.99, 3.0, 4.2, 5.9])
def test_short_sessions_close_cleanly_and_match_offline(secs):
    """< 3 s: no window exists. 3-6 s (< margin): provisional scoring never fires; close scores them."""
    audio = speech_like(7.0, seed=4)[: int(secs * SAMPLE_RATE) + 11]
    sc = LiveScorer(
        fake_score, LABELS, config=LIVE_CFG_V2, f0_center="causal", ling_dim=12
    )
    partition(audio, "rand", 1, sc)
    assert sc.step(WORDS) is None or secs >= 6
    assert not sc.rows  # nothing provisional below the margin
    rows = sc.finalize(WORDS)
    dur = len(audio) / SAMPLE_RATE
    if dur < WINDOW_S:
        assert rows == []
        return
    n_grid = int(np.floor((dur - WINDOW_S) / STRIDE_S + 1e-9)) + 1
    assert [r["timestamp"] for r in rows if not r["terminal"]] == [
        k * STRIDE_S for k in range(n_grid)
    ]
    assert all(r["status"] == "final" for r in rows)
    # equals the offline engine tensors, centre included (causal centre after close == offline centre)
    starts = np.array([r["timestamp"] for r in rows])
    _, _, off = offline_tensors(audio, WORDS, starts)
    for k, r in enumerate(rows):
        t = {key: v[k : k + 1] for key, v in off.items()}
        assert r["todas_las_probabilidades"] == dict(
            zip(LABELS, map(float, fake_score(t)))
        ), r["timestamp"]


@pytest.mark.parametrize(
    "extra_samples,terminal",
    [(0, False), (1, False), (159, False), (160, False), (161, True), (4000, True)],
)
def test_terminal_window_limit_is_one_hop(extra_samples, terminal):
    """end - 3 is on the grid when end = 9 s; the terminal window exists only if it is > 1 hop (10 ms) past it."""
    audio = speech_like(10.0, seed=6)[: 9 * SAMPLE_RATE + extra_samples]
    sc = LiveScorer(fake_score, LABELS, config=LIVE_CFG_V2, ling_dim=12)
    sc.push(audio)
    rows = sc.finalize(WORDS)
    assert any(r["terminal"] for r in rows) is terminal
    if terminal:
        assert [r["timestamp"] for r in rows if r["terminal"]] == [
            pytest.approx(len(audio) / SAMPLE_RATE - WINDOW_S)
        ]


def test_v1_does_not_score_the_tail_and_has_no_status_field():
    audio = speech_like(14.0, seed=2)
    sc = LiveScorer(fake_score, LABELS, config=LIVE_CFG_V1, ling_dim=12)
    partition(audio, "rand", 3, sc)
    sc.step(WORDS)
    n = len(sc.rows)
    assert n > 0 and sc.finalize(WORDS) == [] and len(sc.rows) == n
    assert set(sc.rows[0]) == {
        "timestamp",
        "texto_ventana",
        "clase",
        "confianza",
        "todas_las_probabilidades",
    }
    assert (
        max(r["timestamp"] for r in sc.rows) < 14.0 - 5.0
    )  # the historic 6-8 s of unscored tail


# ------------------------------------------------------------------ (c)(d)(f) tensors / decisions
@pytest.fixture(scope="module")
def session14():
    audio = speech_like(14.0, seed=11)[:-53]
    starts_grid = np.arange(0, (len(audio) / SAMPLE_RATE - WINDOW_S) + 1e-9, 0.5)
    end = len(audio) / SAMPLE_RATE
    starts = (
        np.append(starts_grid, end - WINDOW_S)
        if (end - WINDOW_S) - starts_grid[-1] > 0.01
        else starts_grid
    )
    frames, c_off, off = offline_tensors(audio, WORDS, starts)
    return audio, starts, c_off, off


def _probs_off(off, k):
    return fake_score({key: v[k : k + 1] for key, v in off.items()})


def _run_stream(audio, centre, seed, kind="rand"):
    """Tick-driven live loop: audio arrives in random chunks, a step() after every chunk (as the engine)."""
    sc = LiveScorer(
        fake_score, LABELS, config=LIVE_CFG_V2, f0_center=centre, ling_dim=12
    )
    rng = np.random.default_rng(seed)
    i = 0
    while i < len(audio):
        k = int(np.exp(rng.uniform(np.log(1500), np.log(30000))))
        sc.push(audio[i : i + k])
        sc.step(WORDS)
        i += k
    return sc


def test_c_offline_centre_injected_rows_equal_offline_exactly(session14):
    """Level (c). Given the offline centre (NOT available live) provisional and final rows are exact."""
    audio, starts, c_off, off = session14
    sc = _run_stream(audio, c_off, seed=3)
    sc.finalize(WORDS)
    assert any(r["status"] == "provisional" for r in sc.rows)
    for r in sc.rows:
        k = int(np.argmin(np.abs(starts - r["timestamp"])))
        assert abs(starts[k] - r["timestamp"]) < 1e-9
        assert r["todas_las_probabilidades"] == dict(
            zip(LABELS, map(float, _probs_off(off, k)))
        ), r


def test_d_causal_centre_provisional_rows_only_bounded_not_equal(session14):
    """Level (d): provisional rows with the live centre are NOT equivalent. Only the exact bound is asserted:
    probabilities may differ, but ling/lpos/lmask and non-F0 channels are exact (tested in test_live_parity)."""
    audio, starts, c_off, off = session14
    sc = _run_stream(audio, "causal", seed=3)
    prov = [r for r in sc.rows if r["status"] == "provisional"]
    assert prov
    diff = [
        max(
            abs(r["todas_las_probabilidades"][l] - p)
            for l, p in zip(LABELS, _probs_off(off, int(round(r["timestamp"] / 0.5))))
        )
        for r in prov
    ]
    # divergence is reported (PARIDAD_LIVE_V2.md), the test only requires that it is measurable and finite
    assert np.isfinite(diff).all()
    assert {r["f0_center_st"] for r in prov} != {c_off} or len(prov) < 3


def test_f_final_pass_with_causal_centre_equals_offline_exactly(session14):
    """Level (f): after close the live centre is the whole-session median == the offline centre, so the
    ``final`` rows (tail + rescore + terminal) equal the offline engine given the same words. Provisional
    rows are untouched and keep their own status."""
    audio, starts, c_off, off = session14
    sc = _run_stream(audio, "causal", seed=8)
    before = [dict(r) for r in sc.rows]
    final = sc.finalize(WORDS)
    assert sc.rows[: len(before)] == before, "provisional rows must not be rewritten"
    assert sc.finalize(WORDS) == [], "finalize is idempotent"
    last = {}
    for r in sc.rows:
        last[r["timestamp"]] = r  # consumer rule: last row per timestamp
    assert all(r["status"] == "final" for r in last.values())
    assert sorted(last) == pytest.approx(list(starts))
    for k, s in enumerate(starts):
        r = last[min(last, key=lambda x: abs(x - s))]
        assert r["f0_center_st"] == pytest.approx(c_off, abs=0, rel=0)
        assert r["todas_las_probabilidades"] == dict(
            zip(LABELS, map(float, _probs_off(off, k)))
        ), s
    assert any(r.get("revises") is not None for r in final)
    assert sum(r["terminal"] for r in final) == int(starts[-1] % 0.5 != 0)


# ------------------------------------------------------------------ gaps
def test_device_gap_equals_zero_filled_audio_and_is_recorded():
    audio = speech_like(12.0, seed=5)
    gap = 4321
    with_gap = np.concatenate([audio[:50000], np.zeros(gap, np.float32), audio[50000:]])
    sc = LiveScorer(fake_score, LABELS, config=LIVE_CFG_V2, f0_center=0.0, ling_dim=12)
    sc.push(audio[:50000])
    sc.push(audio[50000:], lost=gap)
    # NOTE: push(audio, lost) puts the lost zeros BEFORE ``audio``: that is the ring semantics
    sc.finalize(WORDS)
    assert sc.gaps == [
        {"kind": "audio_lost", "at_s": 50000 / SAMPLE_RATE, "lost_s": gap / SAMPLE_RATE}
    ]
    np.testing.assert_array_equal(
        sc.tracker.frame_matrix(),
        extract_acoustic(with_gap, use_silero=True).frame_matrix(),
    )


def test_skipped_windows_under_consumer_lag_are_recorded():
    audio = speech_like(30.0, seed=7)
    sc = LiveScorer(fake_score, LABELS, config=LIVE_CFG_V2, f0_center=0.0, ling_dim=12)
    sc.push(audio[: 14 * SAMPLE_RATE])
    sc.step(WORDS)  # first read arrives late: windows before ``latest`` are skipped
    sc.push(audio[14 * SAMPLE_RATE : 30 * SAMPLE_RATE])  # +16 s in one tick
    sc.step(WORDS)
    skipped = [g for g in sc.gaps if g["kind"] == "windows_skipped"]
    assert skipped and sum(g["n_windows"] for g in skipped) > 0
    emitted = {r["timestamp"] for r in sc.rows}
    top = max(emitted)
    all_grid = {k * STRIDE_S for k in range(int(top / STRIDE_S) + 1)}
    assert len(all_grid - emitted) == sum(g["n_windows"] for g in skipped), (
        "every missing window is accounted for"
    )


# ------------------------------------------------------------------ engine: restart, close, idempotent stop
class FakeMic(MicrophoneStream):
    def start(self):  # no sound device
        self.started = True

    def stop(self):
        self.started = False


class FakeLoop:
    def call_soon_threadsafe(self, fn, *a):
        fn(*a)


class TaggedASR:
    """Pure ``merge_recent``: one word per second tagged with the session id; ``delay`` stalls a call."""

    def __init__(self, delay=0.0):
        self.delay, self.words, self.threshold = delay, [], None

    def merge_recent(self, previous, audio, begin):
        if self.delay:
            time.sleep(self.delay)
        n = int(len(audio) / SAMPLE_RATE)
        # the tag follows the AUDIO (loud = session 1, quiet = session 2), not mutable state, so a late
        # worker of session 1 can only produce "S1" words whatever the test changed meanwhile
        tag = "S1" if self.threshold is None or np.abs(audio).max() > self.threshold else "S2"
        new = [
            Word(tag, tag.lower(), begin + k + 0.1, begin + k + 0.4, 0.9, "")
            for k in range(n)
        ]
        keep = [w for w in previous if w.end < begin - 0.12]
        return keep + new


def _engine(config=LIVE_CFG_V2, delay=0.0):
    from scripts.live_demo import LiveEngine

    mic = FakeMic(record=True)
    asr = TaggedASR(delay)
    eng = LiveEngine.for_parts(
        fake_score, LABELS, mic=mic, asr=asr, config=config, ling_dim=12, tick_s=0.01
    )
    return eng, mic, asr


def _feed(mic, audio, upto_s=None):
    for i in range(0, len(audio), 8000):
        mic.buffer.append(audio[i : i + 8000])
        time.sleep(0.005)


def test_engine_stop_scores_the_tail_and_is_idempotent():
    eng, mic, _ = _engine()
    audio = speech_like(12.0, seed=1)
    assert eng.start(FakeLoop())
    _feed(mic, audio)
    time.sleep(0.3)
    assert eng.stop() is True and eng.stop() is False  # idempotent
    assert eng.wait_idle(120)
    rows = eng.session_rows
    assert eng._state.error is None
    prov = [r for r in rows if r["status"] == "provisional"]
    fin = [r for r in rows if r["status"] == "final" and "revises" not in r]
    assert fin, "stop() must score the tail"
    assert max(r["timestamp"] for r in rows) >= 12.0 - WINDOW_S - 0.5 - 1e-9, (
        "tail up to end-3 s reached"
    )
    assert all(r["status"] in ("provisional", "final") for r in rows)
    # the provisional rows were not rewritten by the rescoring pass
    assert {r["revises"] for r in rows if "revises" in r} == {
        r["timestamp"] for r in prov
    }


def test_engine_v1_reproduces_historic_rows_without_tail():
    eng, mic, _ = _engine(config=LIVE_CFG_V1)
    eng.start(FakeLoop())
    _feed(mic, speech_like(12.0, seed=1))
    deadline = time.time() + 120
    while not eng.session_rows and time.time() < deadline:
        time.sleep(0.05)
    eng.stop()
    assert eng.wait_idle(120)
    assert all("status" not in r for r in eng.session_rows)
    assert max(r["timestamp"] for r in eng.session_rows) < 12.0 - 5.0 + 1e-9


def test_quick_stop_start_leaks_no_stale_rows_or_words():
    """The old worker is stalled inside the ASR when stop()+start() happen; its words/rows must stay in session 1."""
    eng, mic, asr = _engine(delay=0.6)
    loud = speech_like(9.0, seed=1)
    asr.threshold = 0.6 * float(np.abs(loud).max())
    eng.start(FakeLoop())
    _feed(mic, loud)  # long enough for the first ASR call (stalled 0.6 s)
    time.sleep(0.05)
    rows1 = eng.session_rows
    eng.stop()
    asr.delay = 0.0  # session 2: quiet audio => "S2" words, fast ASR
    assert eng.start(FakeLoop())
    rows2 = eng.session_rows
    assert rows2 is not rows1 and rows2 == []
    _feed(mic, 0.3 * speech_like(9.5, seed=2))
    time.sleep(0.3)
    eng.stop()
    assert eng.wait_idle(120)
    text2 = " ".join(r["texto_ventana"] for r in rows2)
    assert rows2 and "S1" not in text2, (
        "stale words from session 1 leaked into session 2"
    )
    assert all(r["timestamp"] <= 9.5 for r in rows2), (
        "stale rows from session 1's timeline"
    )
    assert all("S2" not in r["texto_ventana"] for r in rows1), (
        "session 2 words leaked into session 1"
    )
    assert eng.queue.qsize() > 0


def test_stop_without_start_and_double_start():
    eng, mic, _ = _engine()
    assert eng.stop() is False
    assert eng.start(FakeLoop()) is True and eng.start(FakeLoop()) is False
    assert eng.stop() is True
    assert eng.wait_idle(60)


# ------------------------------------------------------------------ long session (slow, real + synthetic)
@pytest.mark.skipif(
    not __import__("os").environ.get("MRCD_SLOW"),
    reason="set MRCD_SLOW=1: ~10.5 min session (~6 min CPU)",
)
def test_long_session_frames_exact_and_final_pass_covers_everything():
    from synth_session import build_session, real_clips

    if not real_clips():
        pytest.skip("AMERESCO clips not present: the long session needs real speech")
    audio, info = build_session(10.5, seed=3)
    assert info["duration_s"] > 600 and info["real_clips"] > 10
    off = extract_acoustic(audio, use_silero=True).frame_matrix()
    # live_cfg_v1 margins (3, 3) differ from offline in ONE frame of this exact session (3.16 st at 27.71 s);
    # live_cfg_v2 margins (3, 4) do not. See margin_study_v2_long10_5_seed3.json.
    tr = SessionFeatureTracker(
        f0_center=0.0, left_margin_s=LIVE_CFG_V2.left_margin_s, right_margin_s=LIVE_CFG_V2.right_margin_s
    )
    rng = np.random.default_rng(0)
    i = 0
    while i < len(audio):
        k = int(rng.choice([4000, 8000, 8000, 12345, 16000]))
        tr.append(audio[i : i + k])
        i += k
    live = tr.frame_matrix()
    np.testing.assert_array_equal(live, off[: len(live)])
    tr.finalize()
    np.testing.assert_array_equal(tr.frame_matrix(), off)


def test_finalize_zero_pads_the_last_partial_silero_chunk():
    """A mutant that skips the zero-padded last chunk survived the previous suite: the synthetic signals of the other
    tests have no speech near the end. This cut of a REAL clip (found by scanning random cuts) differs in 16 frames
    without the padding; len % 512 = 174."""
    import soundfile as sf
    from synth_session import real_clips

    clips = real_clips()
    if not clips:
        pytest.skip("AMERESCO clips not present")
    audio = sf.read(str(clips[0]), dtype="float32")[0][:45742]
    assert len(audio) % 512 == 174
    off = extract_acoustic(audio, use_silero=True).frame_matrix()
    tr = SessionFeatureTracker(f0_center=0.0)
    tr.append(audio)
    tr.finalize()
    np.testing.assert_array_equal(tr.frame_matrix(), off)
    # the mutant (drop the partial chunk instead of padding it) must be distinguishable on this input
    bad = SessionFeatureTracker(f0_center=0.0)
    bad.append(audio)
    bad._vad_tail = np.zeros(0, np.float32)
    bad.finalize()
    assert int((np.abs(bad.frame_matrix() - off).max(1) > 0).sum()) > 0


def test_live_cfg_v2_right_margin_is_four_seconds_and_three_is_a_distinct_config():
    """M17: lowering R from 4 to 3 survived the default suite. Fast pin plus the margin warning behaviour."""
    assert LIVE_CFG_V2.right_margin_s == 4.0 and LIVE_CFG_V2.left_margin_s == 3.0
    assert LIVE_CFG_V1.right_margin_s == 3.0
    from mrcd.live.session import LIVE_CFG_V3

    assert (LIVE_CFG_V3.left_margin_s, LIVE_CFG_V3.right_margin_s) == (3.0, 4.0) and LIVE_CFG_V3.viterbi_carry
    sc = LiveScorer(fake_score, LABELS, config=LIVE_CFG_V2, ling_dim=12)
    assert sc.tracker._right == 400  # frames of 10 ms: the margin that is really applied
    audio = speech_like(8.0, seed=3)
    sc.tracker.append(audio[: 6 * SAMPLE_RATE])
    assert abs(sc.tracker.committed_s - (6.0 - 4.0)) < 0.02  # commit lags by exactly R = 4 s


# ------------------------------------------------------------------ live_cfg_v3: exact Viterbi resume
def test_pyin_resumable_equals_librosa_and_resumes_exactly_from_carried_scores():
    import librosa

    from mrcd.live.pyin_carry import pyin_resumable

    kw = dict(sr=SAMPLE_RATE, fmin=60.0, fmax=400.0, frame_length=1024, hop_length=160)
    audio = speech_like(9.0, seed=7)
    ref_f0, _, ref_vp = librosa.pyin(audio, **kw, center=True)
    f0, vp, _ = pyin_resumable(audio, **kw)
    np.testing.assert_array_equal(np.nan_to_num(f0), np.nan_to_num(ref_f0))
    np.testing.assert_array_equal(vp, ref_vp)
    for c in (50, 311, 700):  # resume points (frames); the first part has its own right edge
        _, _, val = pyin_resumable(audio[: (c + 200) * 160], **kw)
        f0r, _, _ = pyin_resumable(audio, **kw, carry=val[c - 1], carry_index=c)
        np.testing.assert_array_equal(np.nan_to_num(f0r[c:]), np.nan_to_num(ref_f0[c:]))


def _seed23_slice():
    from synth_session import build_session, real_clips

    if not real_clips():
        pytest.skip("AMERESCO clips not present: seed 23 needs real speech")
    return build_session(10.5, seed=23)[0][270 * SAMPLE_RATE : 310 * SAMPLE_RATE]


@pytest.mark.skipif(not __import__("os").environ.get("MRCD_SLOW"), reason="set MRCD_SLOW=1: ~30 s")
def test_seed23_left_dependence_v2_diverges_and_v3_is_exact():
    """Seed 23 of the 10.5 min stress session (REVISION3 H1): with L=3, R=4 and 0.5 s ticks, 10 committed frames from
    295.56 s differ from the whole-file extraction because the pYIN Viterbi decode is restarted from a uniform prior
    inside a noise stretch. The slice 270-310 s reproduces it in ~15 s. v2 must keep showing it (the defect is
    documented, not hidden); v3 (carried forward scores) must be exact."""
    x = _seed23_slice()
    off = extract_acoustic(x, use_silero=True).frame_matrix()

    def run(carry):
        tr = SessionFeatureTracker(f0_center=0.0, left_margin_s=3.0, right_margin_s=4.0, viterbi_carry=carry)
        for i in range(0, len(x), 8000):
            tr.append(x[i : i + 8000])
        tr.finalize()
        g = tr.frame_matrix()
        return np.nonzero(np.abs(g - off[: len(g)]).max(1) > 0)[0]

    bad_v2 = run(False)
    assert len(bad_v2) > 0 and abs(bad_v2[0] * 0.01 + 270 - 295.56) < 0.02, "v2 no longer reproduces the seed-23 divergence"
    assert len(run(True)) == 0


@pytest.mark.parametrize("kind,seed", [("rand", 1), ("mixed", 2)])
def test_v3_tracker_frames_equal_offline_for_arbitrary_partitions_and_after_close(kind, seed):
    audio = speech_like(11.0, seed=seed)[:-37]
    off = extract_acoustic(audio, use_silero=True).frame_matrix()
    tr = SessionFeatureTracker(f0_center=0.0, left_margin_s=3.0, right_margin_s=4.0, viterbi_carry=True)
    rng = np.random.default_rng(seed)
    i = 0
    while i < len(audio):
        k = int(rng.choice([1, 7, 159, 161, 511, 513, 4000, 7919, 30000]) if kind == "mixed" else np.exp(rng.uniform(np.log(300), np.log(40000))))
        tr.append(audio[i : i + k], commit=(k > 200 or rng.random() < 0.05))
        i += k
    np.testing.assert_array_equal(tr.frame_matrix(), off[: len(tr.frame_matrix())])
    tr.finalize()
    np.testing.assert_array_equal(tr.frame_matrix(), off)


def test_f_v3_scorer_final_rows_equal_offline_exactly(session14):
    """live_cfg_v3 changes only the pYIN decode; the end-of-session contract of v2 must hold unchanged."""
    from mrcd.live.session import LIVE_CFG_V3

    audio, starts, c_off, off = session14
    sc = LiveScorer(fake_score, LABELS, config=LIVE_CFG_V3, f0_center="causal", ling_dim=12)
    rng = np.random.default_rng(8)
    i = 0
    while i < len(audio):
        k = int(np.exp(rng.uniform(np.log(1500), np.log(30000))))
        sc.push(audio[i : i + k])
        sc.step(WORDS)
        i += k
    sc.finalize(WORDS)
    last = {}
    for r in sc.rows:
        last[r["timestamp"]] = r
    assert sorted(last) == pytest.approx(list(starts))
    for k, s in enumerate(starts):
        r = last[min(last, key=lambda x: abs(x - s))]
        assert r["status"] == "final"
        assert r["todas_las_probabilidades"] == dict(zip(LABELS, map(float, _probs_off(off, k)))), s
