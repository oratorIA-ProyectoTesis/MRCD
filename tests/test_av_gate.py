import numpy as np

from mrcd.av.config import GATE
from mrcd.av.gate import (
    FaceSeries,
    analyze_video,
    identity_drift,
    lag_correlation,
    peak_offset_frames,
    speech_envelope,
)

FPS = 25.0


def _speech_env(n, rng):
    """Bursty envelope (syllable-like) with unit variance."""
    x = rng.normal(size=n)
    env = np.convolve(x, np.ones(5) / 5, mode="same")
    return (env - env.mean()) / env.std()


def _series(n, shift, rng, two_faces=False, no_face=False, noise=0.1):
    env = _speech_env(n, rng)
    lip = np.roll(env, shift) + noise * rng.normal(
        size=n
    )  # lip[t] = env[t - shift]: video late by `shift`
    present = np.ones((1, n), bool)
    shape = np.tile(
        np.array([0.4, 0.5, 0.3, 0.7, 0.2]), (1, n, 1)
    ) + 0.002 * rng.normal(size=(1, n, 5))
    lip_a = lip[None].copy()
    if two_faces:
        present = np.vstack([present, np.ones((1, n), bool)])
        shape = np.concatenate([shape, shape * 1.3], 0)
        lip_a = np.vstack([lip_a, rng.normal(size=(1, n))])
    if no_face:
        present[:] = False
        lip_a[:] = np.nan
    return FaceSeries(FPS, lip_a, present, present.copy(), shape, env)


def _units(n, dur=6.0):
    out, t = [], 4.0
    while t + dur < n / FPS - 4:
        out.append((f"u{len(out)}", t, t + dur))
        t += dur + 1.0
    return out


def test_lag_recovers_known_offset_sign_and_value():
    rng = np.random.default_rng(0)
    env = _speech_env(400, rng)
    off, r = peak_offset_frames(lag_correlation(np.roll(env, 3), env, 10, 20))
    assert abs(off - 3) < 0.3 and r > 0.95
    off2, _ = peak_offset_frames(lag_correlation(np.roll(env, -2), env, 10, 20))
    assert abs(off2 + 2) < 0.3


def test_envelope_follows_energy():
    sr = 16000
    a = np.concatenate([np.zeros(sr), 0.5 * np.sin(np.arange(sr) * 0.2), np.zeros(sr)])
    e = speech_envelope(a, sr, FPS, 75)
    assert e[37] > e[5] + 2 and e[37] > e[70] + 2


def test_synchronised_video_is_admitted_and_offset_estimated():
    rng = np.random.default_rng(1)
    n = 25 * 120
    rows, v = analyze_video(_series(n, 2, rng), _units(n), seed=1)
    assert v["reason"] == "ok" and abs(v["offset_ms"] - 80) < 40
    assert sum(r["admitted"] for r in rows) >= 0.9 * len(rows)


def test_large_desync_is_excluded_with_reason():
    rng = np.random.default_rng(2)
    n = 25 * 120
    rows, v = analyze_video(_series(n, 8, rng), _units(n), seed=2)  # 320 ms
    assert v["reason"] != "ok"
    assert not any(r["admitted"] for r in rows)
    assert all(r["primary_reason"].startswith("video_sync_unverified") for r in rows)


def test_no_face_is_excluded_not_zero():
    rng = np.random.default_rng(3)
    n = 25 * 60
    rows, _ = analyze_video(_series(n, 0, rng, no_face=True), _units(n), seed=3)
    assert rows and not any(r["admitted"] for r in rows)
    assert all("face_not_visible_enough" in r["reasons"] for r in rows)
    assert all(r["r_at_video_offset"] is None for r in rows)


def test_two_faces_makes_active_speaker_ambiguous():
    rng = np.random.default_rng(4)
    n = 25 * 90
    rows, _ = analyze_video(_series(n, 1, rng, two_faces=True), _units(n), seed=4)
    assert rows and not any(r["admitted"] for r in rows)
    assert all(
        "second_face_present_active_speaker_ambiguous" in r["reasons"] for r in rows
    )


def test_unrelated_audio_fails_correlation_gate():
    rng = np.random.default_rng(5)
    n = 25 * 120
    fs = _series(n, 0, rng)
    fs.env = _speech_env(n, np.random.default_rng(99))  # audio unrelated to the lips
    rows, _ = analyze_video(fs, _units(n), seed=5)
    assert sum(r["admitted"] for r in rows) == 0


def test_short_unit_excluded():
    rng = np.random.default_rng(6)
    rows, _ = analyze_video(_series(25 * 60, 0, rng), [("s", 10.0, 10.8)], seed=6)
    assert rows[0]["primary_reason"] == "unit_too_short"


def test_identity_drift_detects_face_swap():
    rng = np.random.default_rng(7)
    same = 0.5 + 0.01 * rng.normal(size=(80, 5))
    assert identity_drift(same) < 3.0
    swap = np.vstack([same[:40], 0.7 + 0.01 * rng.normal(size=(40, 5))])
    assert identity_drift(swap) > 3.0
    assert np.isnan(identity_drift(same[:10]))


def test_short_video_few_units_cannot_estimate_offset():
    rng = np.random.default_rng(8)
    n = 25 * 30
    fs = _series(n, 1, rng)
    rows, v = analyze_video(fs, [("a", 3.0, 8.0), ("b", 10.0, 14.0)], seed=8)  # 2 units < 3
    assert v["reason"] == "offset_unestimable_few_units"
    assert not any(r["admitted"] for r in rows)


def test_pooled_offset_works_with_three_short_units():
    rng = np.random.default_rng(9)
    n = 25 * 60
    fs = _series(n, 2, rng)
    rows, v = analyze_video(fs, [("a", 3.0, 9.0), ("b", 12.0, 18.0), ("c", 21.0, 27.0)], seed=9)
    assert v["reason"] == "ok" and abs(v["offset_ms"] - 80) < 40


def test_a2_merged_tracks_admit_a_person_reacquired_under_new_ids():
    rng = np.random.default_rng(10)
    n = 25 * 120
    fs = _series(n, 2, rng)
    # same person, tracker alternates between two ids every 40 frames (fragmentation)
    present = np.zeros((2, n), bool)
    for k in range(n):
        present[(k // 40) % 2, k] = True
    lip = np.where(present, np.vstack([fs.lip[0], fs.lip[0]]), np.nan)
    shape = np.where(present[..., None], np.concatenate([fs.shape, fs.shape], 0), np.nan)
    frag = FaceSeries(FPS, lip, present, present.copy(), shape, fs.env)
    a1, _ = analyze_video(frag, _units(n), seed=10)
    a2, v2 = analyze_video(frag, _units(n), seed=10, cfg=dict(GATE, merge_tracks=True))
    assert sum(r["admitted"] for r in a1) < 0.3 * len(a1)
    assert v2["reason"] == "ok" and sum(r["admitted"] for r in a2) >= 0.9 * len(a2)


def test_a2_still_rejects_simultaneous_faces():
    rng = np.random.default_rng(11)
    n = 25 * 90
    rows, _ = analyze_video(_series(n, 1, rng, two_faces=True), _units(n), seed=11, cfg=dict(GATE, merge_tracks=True))
    assert rows and not any(r["admitted"] for r in rows)
