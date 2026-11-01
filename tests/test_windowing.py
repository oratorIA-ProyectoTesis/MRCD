import numpy as np
import pytest

from mrcd import windowing as W


def test_plan_short_unit_is_padded_to_one_window():
    starts, b, extra, padded = W.plan_unit(16000, 0.0, 1.0)
    assert padded and starts.tolist() == [0.0] and b == 3.0 and extra == 2 * 16000


def test_plan_long_unit_has_terminal_window():
    starts, b, extra, padded = W.plan_unit(int(5.2 * 16000), 0.0, 5.2)
    assert not padded and extra == 0
    assert starts[-1] == pytest.approx(2.2) and starts[-2] == pytest.approx(2.0)


def test_plan_exact_grid_has_no_extra_terminal():
    starts, *_ = W.plan_unit(5 * 16000, 0.0, 5.0)
    assert starts.tolist() == [0.0, 0.5, 1.0, 1.5, 2.0]


def test_zero_pad_and_noise_pad_lengths_and_determinism():
    rng = np.random.default_rng(0)
    a = (rng.standard_normal(8000) * 0.05).astype(np.float32)
    z = W.pad_audio(a, 4000, "zero")
    n1 = W.pad_audio(a, 4000, "noise", "u1")
    n2 = W.pad_audio(a, 4000, "noise", "u1")
    n3 = W.pad_audio(a, 4000, "noise", "u2")
    assert len(z) == len(n1) == 12000 and not z[8000:].any()
    assert np.array_equal(n1, n2) and not np.array_equal(n1, n3)
    assert np.array_equal(n1[:8000], a) and n1[8000:].std() > 0
    assert W.pad_audio(a, 0, "noise") is a


def test_noise_pad_stays_at_or_below_speech_level():
    a = np.concatenate(
        [np.zeros(4000, np.float32) + 1e-3, np.ones(4000, np.float32) * 0.2]
    )
    pad = W.pad_audio(a, 8000, "noise", "x")[8000:]
    assert pad.std() < 0.05


def test_bad_pad_mode():
    with pytest.raises(ValueError):
        W.pad_audio(np.zeros(10, np.float32), 5, "mirror")


def test_abstain():
    assert (
        W.should_abstain(0.4, 0.5)
        and not W.should_abstain(0.5, 0.5)
        and not W.should_abstain(0.1, None)
    )


def test_pause_stats():
    vad = np.array([0] * 20 + [1] * 30 + [0] * 70 + [1] * 10 + [0] * 5)
    s = W.internal_pause_stats(vad)
    assert s["max_internal_pause_s"] == pytest.approx(0.70)
    assert s["edge_silence_s"] == pytest.approx(0.25)
    assert W.internal_pause_stats(np.zeros(10))["speech_frac"] == 0.0
    # padded tail excluded through n_valid
    assert (
        W.internal_pause_stats(np.r_[np.ones(10), np.zeros(300)], n_valid=10)[
            "edge_silence_s"
        ]
        == 0.0
    )
