"""CTC Viterbi aligner of protocol G: synthetic emission matrices only (no model, no audio, no download)."""

from __future__ import annotations

import numpy as np
import pytest

from scripts.protocol_g import ctc_align as A

VOCAB = {"<pad>": 0, "|": 1, "a": 2, "b": 3, "c": 4}
BLANK = 0
STRIDE = 0.02


def onehot_path(states: list[int], V: int = 5, eps: float = 1e-3) -> np.ndarray:
    """Log-probs with ~1 on the chosen symbol per frame."""
    p = np.full((len(states), V), eps / (V - 1))
    for t, s in enumerate(states):
        p[t, s] = 1 - eps
    return np.log(p)


def test_viterbi_recovers_the_planted_path_and_token_times():
    # frames: blank, a a, blank, b b b, | , c, blank
    syms = [0, 2, 2, 0, 3, 3, 3, 1, 4, 0]
    out = A.align_tokens(onehot_path(syms), ["ab", "c"], VOCAB, BLANK, STRIDE)
    t_ab, t_c = out.tokens
    assert t_ab.timed and t_c.timed
    assert t_ab.start == pytest.approx(1 * STRIDE)  # first frame of 'a'
    assert t_ab.end == pytest.approx(7 * STRIDE)  # last frame of 'b' (6) + 1
    assert t_c.start == pytest.approx(8 * STRIDE) and t_c.end == pytest.approx(
        9 * STRIDE
    )
    assert t_ab.conf > 0.99 and t_c.conf > 0.99
    assert out.dropped_chars == 0 and out.interpolated_tokens == 0


def test_repeated_characters_need_a_blank_between_them():
    # 'aa' needs a, blank, a: three frames at least
    with pytest.raises(A.AlignmentError):
        A.ctc_viterbi(onehot_path([2, 2]), [2, 2], BLANK)
    path = A.ctc_viterbi(onehot_path([2, 0, 2]), [2, 2], BLANK)
    assert list(path) == [1, 2, 3]


def test_infeasible_and_empty_targets_raise():
    with pytest.raises(A.AlignmentError):
        A.ctc_viterbi(onehot_path([2]), [2, 3], BLANK)
    with pytest.raises(A.AlignmentError):
        A.ctc_viterbi(onehot_path([2]), [], BLANK)


def test_characters_outside_the_vocabulary_are_dropped_and_empty_tokens_interpolated():
    syms = [2, 2, 0, 0, 0, 0, 4, 4]
    # '7' is not in the vocabulary: the token '7' has no character left -> untimed, interpolated between neighbours
    out = A.align_tokens(onehot_path(syms), ["a", "7", "c"], VOCAB, BLANK, STRIDE)
    assert out.dropped_chars == 1 and out.interpolated_tokens == 1
    mid = out.tokens[1]
    assert not mid.timed and np.isnan(mid.conf)
    assert out.tokens[0].end <= mid.start <= mid.end <= out.tokens[2].start + 1e-9


def test_confidence_gate_counts_untimed_tokens_as_not_confident():
    g = A.confidence_gate([0.9] * 7 + [0.2] * 3)
    assert g["passed"] and g["share"] == pytest.approx(0.7)  # exactly 70 % passes (>=)
    g = A.confidence_gate([0.9] * 6 + [float("nan")] * 4)
    assert not g["passed"]
    assert not A.confidence_gate([])["passed"]


def test_low_posterior_gives_low_token_confidence():
    p = np.log(np.full((6, 5), 0.2))  # flat emissions: path exists, confidence 0.2
    out = A.align_tokens(p, ["ab"], VOCAB, BLANK, STRIDE)
    assert out.tokens[0].timed and out.tokens[0].conf == pytest.approx(0.2)
    assert not A.confidence_gate([out.tokens[0].conf])["passed"]


def test_windowed_emissions_equal_the_whole_audio_emission():
    n = 45 * 16000
    audio = np.arange(n, dtype=np.float32)

    def emit(
        chunk,
    ):  # frame k of a chunk = sample 320*k; value encodes the sample index of the chunk's own frame
        T = max(0, (len(chunk) - 400) // 320 + 1)
        return np.stack([chunk[np.arange(T) * 320] / 320.0, np.zeros(T)], axis=1)

    out = A.windowed_emissions(emit, audio)
    assert out.shape[0] == (n - 400) // 320 + 1
    # chunk-relative index + chunk offset reconstruct the GLOBAL frame index: no frame dropped or duplicated
    assert np.allclose(out[:, 0], np.arange(out.shape[0]))


def test_windowed_emissions_detect_a_gap():
    audio = np.zeros(30 * 16000, np.float32)
    calls = []

    def emit(chunk):
        calls.append(len(chunk))
        T = (len(chunk) - 400) // 320 + 1
        return np.zeros(
            (T - (150 if len(calls) == 1 else 0), 2)
        )  # first window loses its tail frames

    with pytest.raises(A.AlignmentError):
        A.windowed_emissions(emit, audio)


def test_anchor_to_words_levenshtein_marks_only_equal_words_as_matched():
    toks = ["la", "casa", "es", "grande"]
    words = [
        ("la", 0.0, 0.2),
        ("caza", 0.2, 0.5),
        ("es", 0.5, 0.6),
        ("grande", 0.6, 1.0),
    ]
    out = A.anchor_to_words(toks, words, 1.0)
    assert (
        [t.conf for t in out.tokens][0] == 1.0
        and out.tokens[2].timed
        and out.tokens[3].timed
    )
    assert (
        not out.tokens[1].timed and out.interpolated_tokens == 1
    )  # substituted word is not an exact match
    assert (
        out.tokens[0].end
        <= out.tokens[1].start
        <= out.tokens[1].end
        <= out.tokens[2].start
    )


def test_forced_aligner_with_a_fake_provider_is_deterministic():
    syms = [0] + [2] * 4 + [0] + [3] * 4 + [1] + [4] * 3 + [0]

    class Fake:
        vocab, blank_id, stride_s = VOCAB, BLANK, STRIDE

        def emissions(self, audio):
            return onehot_path(syms)[: max(0, (len(audio) - 400) // 320 + 1)]

    n = 400 + 320 * (len(syms) - 1)
    r1 = A.ForcedAligner(Fake()).align(np.zeros(n, np.float32), ["ab", "c"])
    r2 = A.ForcedAligner(Fake()).align(np.zeros(n, np.float32), ["ab", "c"])
    assert [(t.start, t.end, t.conf) for t in r1.tokens] == [
        (t.start, t.end, t.conf) for t in r2.tokens
    ]
