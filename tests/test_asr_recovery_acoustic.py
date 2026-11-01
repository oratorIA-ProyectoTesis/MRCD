"""O3 acoustic detector on synthetic signals: repeated vs non-repeated syllable sequences, DTW properties."""

from __future__ import annotations

import numpy as np

from mrcd.asr_recovery import acoustic as AC

SR = 16000


def syl(f, dur=0.2):
    t = np.arange(int(SR * dur)) / SR
    return np.hanning(len(t)) * sum(
        np.sin(2 * np.pi * f * h * t) / h for h in (1, 2, 3.3)
    )


def turn(freqs, gap=0.05):
    out = []
    for f in freqs:
        out += [syl(f), np.zeros(int(SR * gap))]
    return np.concatenate(out)


def words_for(n, dur=0.2, step=0.25):
    return [
        {
            "text": "x",
            "norm": "x",
            "start": i * step,
            "end": i * step + dur,
            "prob": 0.9,
        }
        for i in range(n)
    ]


def feat(audio, n):
    va, vv = AC.acoustic_features(audio, words_for(n))
    return dict(zip(AC.NAMES_ASR, va)), dict(zip(AC.NAMES_VAD, vv))


def test_dtw_identity_symmetry_and_time_warp():
    rng = np.random.default_rng(0)
    A = rng.normal(size=(20, 12))
    assert AC.dtw_cost(A, A) == 0.0
    B = rng.normal(size=(24, 12))
    assert abs(AC.dtw_cost(A, B) - AC.dtw_cost(B, A)) < 1e-9
    warped = np.repeat(A, 2, axis=0)  # same content played at half speed
    assert AC.dtw_cost(A, warped) < 0.25 * AC.dtw_cost(A, B)


def test_repeated_trigram_has_much_lower_cost_than_a_non_repeated_one():
    rep = turn([300, 500, 800, 300, 500, 800, 420, 650, 900])
    non = turn([300, 500, 800, 350, 620, 950, 420, 650, 900])
    a_r, v_r = feat(rep, 9)
    a_n, v_n = feat(non, 9)
    assert a_r["asr_minraw_k3"] < 0.3 * a_n["asr_minraw_k3"]
    assert v_r["vad_minraw_k3"] < 0.3 * v_n["vad_minraw_k3"]


def test_vad_detector_does_not_need_the_asr_words():
    rep = turn([300, 500, 800, 300, 500, 800, 420, 650, 900])
    _, v1 = AC.acoustic_features(rep, words_for(9))
    _, v2 = AC.acoustic_features(rep, [])  # ASR returned nothing
    assert np.allclose(v1, v2, equal_nan=True)
    segs = AC.vad_segments(rep)
    assert 7 <= len(segs) <= 11  # ~ one segment per synthetic syllable


def test_single_word_repetition_inside_one_asr_word_is_found_by_the_half_split():
    audio = turn([300, 300])  # two identical syllables
    one_word = [{"text": "x", "norm": "x", "start": 0.0, "end": 0.51, "prob": 0.9}] + [
        {
            "text": "y",
            "norm": "y",
            "start": 0.6 + 0.25 * i,
            "end": 0.8 + 0.25 * i,
            "prob": 0.9,
        }
        for i in range(3)
    ]
    audio = np.concatenate([audio, turn([500, 700, 900])])
    va, _ = AC.acoustic_features(audio, one_word)
    f = dict(zip(AC.NAMES_ASR, va))
    assert f["asr_minraw_k1"] < 0.8


def test_too_short_audio_returns_nan_vectors_without_raising():
    va, vv = AC.acoustic_features(np.zeros(800, np.float32), [])
    assert len(va) == len(AC.NAMES_ASR) and len(vv) == len(AC.NAMES_VAD)
    assert np.isnan(va).all() and np.isnan(vv).all()


def test_features_do_not_depend_on_other_turns_or_state():
    a = turn([300, 500, 800, 300, 500, 800])
    f1 = AC.acoustic_features(a, words_for(6))
    AC.acoustic_features(turn([400, 450, 900]), words_for(3))
    f2 = AC.acoustic_features(a, words_for(6))
    assert np.allclose(f1[0], f2[0], equal_nan=True) and np.allclose(
        f1[1], f2[1], equal_nan=True
    )
