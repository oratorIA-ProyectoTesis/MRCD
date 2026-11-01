"""Observable visual features per unit, plus the audio / text blocks of the A+T baseline.

Visual features are geometric descriptions of lip aperture, jaw opening and head movement. They are never mapped to
emotions, doubt, anxiety, intention or any diagnosis. Speaker normalisation uses label-free per-speaker robust
statistics of the speaker's own admitted-unit frames; the speaker identity is never a feature.
"""

from __future__ import annotations

import numpy as np

V_NAMES = [
    "ap_mean",
    "ap_std",
    "ap_p90",
    "ap_p10",
    "ap_vel_rms",
    "steady_frac",
    "longest_steady_s",
    "jaw_mean",
    "jaw_std",
    "pucker_mean",
    "head_speed",
    "head_dev",
]
SIGNALS = ("ap", "jaw", "pucker", "pitch", "yaw", "roll")


def _robust(x: np.ndarray) -> tuple[float, float]:
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan"), float("nan")
    med = float(np.median(x))
    mad = float(1.4826 * np.median(np.abs(x - med)))
    return med, max(mad, 1e-6)


def speaker_baseline(frames: dict[str, np.ndarray], min_frames: int) -> dict | None:
    """Robust (median, scale) per signal from a speaker's own frames. None if there are too few frames.

    `frames[name]` is a 1-D array of per-frame values pooled over the speaker's admitted units (label-free).
    The head-speed scale is the MAD of absolute frame-to-frame angular change (pooled, no cross-unit differences).
    """
    n = len(frames["ap"]) if "ap" in frames else 0
    if n < min_frames:
        return None
    base = {k: _robust(np.asarray(frames[k], float)) for k in SIGNALS}
    base["speed"] = (
        _robust(np.asarray(frames["speed"], float)) if "speed" in frames else (0.0, 1.0)
    )
    return base


def _runs(mask: np.ndarray) -> int:
    best = cur = 0
    for m in mask:
        cur = cur + 1 if m else 0
        best = max(best, cur)
    return best


def visual_features(
    sig: dict[str, np.ndarray], fps: float, base: dict | None
) -> np.ndarray | None:
    """12-D observable features for one unit. `sig` has per-frame arrays for SIGNALS (NaN when missing).

    base=None gives the RAW (un-normalised) variant: used only for the identity-leakage comparison.
    Returns None when the unit has fewer than 10 valid frames (caller must record 'not available', never zeros).
    """
    ap = np.asarray(sig["ap"], float)
    ok = np.isfinite(ap)
    if ok.sum() < 10:
        return None

    def norm(name):
        x = np.asarray(sig[name], float)
        if base is None:
            return x
        m, s = base[name]
        return (x - m) / s

    a, jw, pk = norm("ap")[ok], norm("jaw")[ok], norm("pucker")[ok]
    d_ap = np.diff(a) * fps
    steady = (
        np.abs(np.diff(a)) < 0.10
        if base is not None
        else np.abs(np.diff(a)) < 0.10 * np.nanstd(a)
    )
    ang = np.stack([np.asarray(sig[k], float)[ok] for k in ("pitch", "yaw", "roll")], 1)
    speed = np.sqrt(np.nansum(np.diff(ang, axis=0) ** 2, 1)) * fps
    if base is None:
        spd = speed
        dev = np.nanmean(np.abs(ang - np.nanmedian(ang, 0)), 0).mean()
    else:
        m, s = base["speed"]
        spd = (speed - m) / s
        dev = np.nanmean(
            [
                np.nanmean(np.abs((ang[:, i] - base[k][0]) / base[k][1]))
                for i, k in enumerate(("pitch", "yaw", "roll"))
            ]
        )
    v = np.array(
        [
            np.nanmean(a),
            np.nanstd(a),
            np.nanpercentile(a, 90),
            np.nanpercentile(a, 10),
            float(np.sqrt(np.nanmean(d_ap**2))) if d_ap.size else np.nan,
            float(np.mean(steady)) if steady.size else np.nan,
            _runs(steady) / fps,
            np.nanmean(jw),
            np.nanstd(jw),
            np.nanmean(pk),
            np.nanmean(spd) if spd.size else np.nan,
            dev,
        ],
        np.float32,
    )
    return v if np.isfinite(v).all() else None


# ------------------------------------------------------------------ audio block (A)
A_NAMES = [
    "dur_s",
    "speech_frac",
    "voiced_frac",
    "rms_mean",
    "rms_std",
    "f0_med_rel_st",
    "f0_std_st",
    "f0_absdiff_st",
    "longest_voiced_s",
    "f0_std_longest_voiced",
    "longest_silence_s",
    "silences_per_s",
]


def _longest_run(mask: np.ndarray, hop: float) -> tuple[int, int]:
    best, bs, cur, cs = 0, 0, 0, 0
    for i, m in enumerate(mask):
        if m:
            if cur == 0:
                cs = i
            cur += 1
            if cur > best:
                best, bs = cur, cs
        else:
            cur = 0
    return bs, best


def audio_features(
    frame_matrix: np.ndarray, f0_center_st: float, hop_s: float = 0.01
) -> np.ndarray:
    """From AcousticTrack.frame_matrix() = [log_rms, f0_st, voiced_prob, vad, df0] of the unit clip."""
    T = len(frame_matrix)
    log_rms, f0, vp, vad = (
        frame_matrix[:, 0],
        frame_matrix[:, 1],
        frame_matrix[:, 2],
        frame_matrix[:, 3],
    )
    voiced = f0 != 0.0
    speech = vad > 0.5
    bs, bl = _longest_run(voiced & speech, hop_s)
    f0_run = f0[bs : bs + bl] if bl >= 5 else np.array([np.nan])
    sil = ~speech
    ss, sl = _longest_run(sil, hop_s)
    n_sil = int(np.sum(np.diff(np.concatenate([[0], sil.astype(int)])) == 1))
    fv = f0[voiced]
    v = np.array(
        [
            T * hop_s,
            speech.mean(),
            voiced.mean(),
            log_rms[speech].mean() if speech.any() else np.nan,
            log_rms[speech].std() if speech.any() else np.nan,
            (np.median(fv) - f0_center_st) if fv.size else np.nan,
            fv.std() if fv.size > 3 else np.nan,
            np.mean(np.abs(np.diff(fv))) if fv.size > 3 else np.nan,
            bl * hop_s,
            np.nanstd(f0_run) if np.isfinite(f0_run).any() else np.nan,
            sl * hop_s,
            n_sil / max(T * hop_s, 1e-6),
        ],
        np.float32,
    )
    return v


# ------------------------------------------------------------------ text block (T), from ASR words only
T_NAMES = [
    "n_words_per_s",
    "mean_prob",
    "min_prob",
    "low_prob_frac",
    "lexical_filler_per_s",
    "repeat_per_s",
    "max_gap_s",
    "mean_gap_s",
    "max_word_dur_s",
    "mean_log_word_dur",
    "trunc_per_s",
    "sentence_punct_per_s",
]


def text_features(words, dur_s: float, word_feat: np.ndarray) -> np.ndarray:
    """`word_feat` = mrcd.extractors.linguistic.word_features(words) (N,10). Empty ASR -> all-zero counts, NaN stats.

    A unit with zero ASR words is a real observation (no recognised words), so counts are 0 by definition; the
    statistics that need words are NaN and are imputed with the training median in the classifier pipeline.
    """
    n = len(words)
    dur_s = max(dur_s, 1e-6)
    if n == 0:
        return np.array([0, *([np.nan] * 3), 0, 0, *([np.nan] * 4), 0, 0], np.float32)
    prob = np.array([w.prob for w in words], float)
    durs = np.array([max(w.end - w.start, 1e-3) for w in words], float)
    gaps = word_feat[:, 8]
    v = np.array(
        [
            n / dur_s,
            prob.mean(),
            prob.min(),
            float(np.mean(prob < 0.5)),
            word_feat[:, 0].sum() / dur_s,
            word_feat[:, 3].sum() / dur_s,
            float(np.max(gaps)),
            float(np.mean(gaps)),
            float(durs.max()),
            float(np.mean(np.log(durs))),
            word_feat[:, 4].sum() / dur_s,
            word_feat[:, 5].sum() / dur_s,
        ],
        np.float32,
    )
    return v
