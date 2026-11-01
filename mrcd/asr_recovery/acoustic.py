"""O3: acoustic "sounded twice" detector (DTW self-similarity between adjacent segments). Protocol section 5.

Segmentations: ``asr`` (cached word timestamps) and ``vad`` (energy valleys, no ASR). Nothing here is fitted:
each score is internal to the turn; the learning (fills, standardisation, thresholds) lives in the CV layer.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks
from scipy.spatial.distance import cdist

try:  # numba is optional: the fallback is the same recursion in pure Python
    from numba import njit
except Exception:  # pragma: no cover

    def njit(*a, **k):
        def deco(f):
            return f

        return deco if not (len(a) == 1 and callable(a[0])) else a[0]


SR = 16000
HOP = 320  # 20 ms
FRAME_S = HOP / SR
MIN_FRAMES = 6  # 0.12 s
MAX_DUR_RATIO = 1.6
N_BG = 6
KS_ASR = (1, 2, 3)
KS_VAD = (1, 2, 3, 4)
HALF_MIN_S = 0.5
RATIO_THR = 0.6


@njit
def _dtw(D):
    n, m = D.shape
    acc = np.full((n + 1, m + 1), np.inf)
    acc[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            a = acc[i - 1, j]
            b = acc[i, j - 1]
            c = acc[i - 1, j - 1]
            best = a if a < b else b
            best = best if best < c else c
            acc[i, j] = D[i - 1, j - 1] + best
    return acc[n, m]


def dtw_cost(A: np.ndarray, B: np.ndarray) -> float:
    """Length-normalised DTW cost between two (frames, dims) sequences (Euclidean frame distance)."""
    D = cdist(A, B).astype(np.float64)
    return float(_dtw(D) / (len(A) + len(B)))


def mfcc_turn(audio: np.ndarray) -> np.ndarray:
    """(frames, 12) MFCC without c0, per-turn mean / variance normalised (telephone band: fmax 4 kHz)."""
    import librosa

    m = librosa.feature.mfcc(
        y=audio.astype(np.float32),
        sr=SR,
        n_mfcc=13,
        n_fft=400,
        hop_length=HOP,
        n_mels=40,
        fmin=0,
        fmax=4000,
    )[1:].T
    mu, sd = m.mean(0), m.std(0)
    sd[sd < 1e-6] = 1.0
    return (m - mu) / sd


def vad_segments(audio: np.ndarray) -> list[tuple[int, int]]:
    """Syllable-like segments from the log-energy envelope; no ASR involved."""
    import librosa

    rms = librosa.feature.rms(
        y=audio.astype(np.float32), frame_length=400, hop_length=HOP
    )[0]
    db = 20 * np.log10(rms + 1e-5)
    sm = np.convolve(db, np.ones(3) / 3, mode="same")
    if len(sm) < MIN_FRAMES:
        return []
    speech = sm > (np.percentile(sm, 95) - 35.0)
    peaks, _ = find_peaks(sm, prominence=3.0, distance=4)
    peaks = [p for p in peaks if speech[p]]
    segs: list[tuple[int, int]] = []
    i, n = 0, len(sm)
    while i < n:
        if not speech[i]:
            i += 1
            continue
        j = i
        while j < n and speech[j]:
            j += 1
        pk = [p for p in peaks if i <= p < j]
        cuts = [i]
        for a, b in zip(pk[:-1], pk[1:]):
            cuts.append(a + int(np.argmin(sm[a : b + 1])))
        cuts.append(j)
        segs.extend((a, b) for a, b in zip(cuts[:-1], cuts[1:]) if b > a)
        i = j
    merged: list[list[int]] = []
    for a, b in segs:
        if (
            merged
            and (b - a < 4 or merged[-1][1] - merged[-1][0] < 4)
            and a - merged[-1][1] <= 1
        ):
            merged[-1][1] = b
        else:
            merged.append([a, b])
    return [(a, b) for a, b in merged if b - a >= 2]


def asr_segments(words: list[dict], n_frames: int) -> list[tuple[int, int]]:
    segs = []
    for w in sorted(words, key=lambda w: (w["start"], w["end"])):
        a = max(int(round(w["start"] / FRAME_S)), 0)
        b = min(int(round(w["end"] / FRAME_S)), n_frames)
        if b - a >= 2:
            segs.append((a, b))
    return segs


def _units(segs, k):
    return [(segs[i][0], segs[i + k - 1][1]) for i in range(len(segs) - k + 1)]


def _ok(a, b) -> bool:
    la, lb = a[1] - a[0], b[1] - b[0]
    return (
        min(la, lb) >= MIN_FRAMES and max(la, lb) / max(min(la, lb), 1) <= MAX_DUR_RATIO
    )


def _ratio(feats, A, B, others):
    """(raw cost, ratio) for the pair A-B with a within-turn background of ``others`` (units)."""
    d_ab = dtw_cost(feats[A[0] : A[1]], feats[B[0] : B[1]])
    bg = [
        dtw_cost(feats[A[0] : A[1]], feats[o[0] : o[1]]) for o in others if _ok(A, o)
    ][:N_BG]
    if len(bg) < 2:
        return d_ab, float("nan")
    return d_ab, d_ab / max(float(np.median(bg)), 1e-9)


def pair_scores(feats: np.ndarray, segs, ks) -> dict:
    """Per k: list of (raw, ratio) over every adjacent pair of k-grams."""
    out = {k: [] for k in ks}
    for k in ks:
        U = _units(segs, k)
        for i in range(len(U) - k):
            A, B = U[i], U[i + k]
            if not _ok(A, B):
                continue
            cand = sorted(
                (j for j in range(len(U)) if j <= i - k - 1 or j >= i + 2 * k),
                key=lambda j: abs(j - i),
            )
            out[k].append(_ratio(feats, A, B, [U[j] for j in cand]))
    return out


def _halves(feats, words, segs_all):
    out = []
    word_units = _units(segs_all, 1)
    for w in words:
        if w["end"] - w["start"] <= HALF_MIN_S:
            continue
        a, b = (
            int(round(w["start"] / FRAME_S)),
            min(int(round(w["end"] / FRAME_S)), len(feats)),
        )
        mid = (a + b) // 2
        A, B = (a, mid), (mid, b)
        if not _ok(A, B):
            continue
        cand = sorted(
            (u for u in word_units if u[1] <= a or u[0] >= b),
            key=lambda u: abs(u[0] - a),
        )
        out.append(_ratio(feats, A, B, cand))
    return out


def summarise(
    scores: dict, ks, extra=None, n_units: int = 0, speech_s: float = 0.0
) -> list[float]:
    nan = float("nan")
    ratios, v = [], []
    for k in ks:
        sc = list(scores[k]) + (list(extra) if (extra and k == 1) else [])
        r = [x[1] for x in sc if np.isfinite(x[1])]
        raw = [x[0] for x in sc]
        v.append(min(r) if r else nan)
        v.append(min(raw) if raw else nan)
        ratios.extend(r)
    v.append(min(ratios) if ratios else nan)
    v.append(float(sum(r < RATIO_THR for r in ratios)))
    v.append(float(n_units))
    v.append(float(speech_s))
    return v


def names(prefix: str, ks) -> list[str]:
    return [f"{prefix}_{t}_k{k}" for k in ks for t in ("minratio", "minraw")] + [
        f"{prefix}_minratio_all",
        f"{prefix}_n_ratio_lt06",
        f"{prefix}_n_units",
        f"{prefix}_speech_s",
    ]


NAMES_ASR = names("asr", KS_ASR)
NAMES_VAD = names("vad", KS_VAD)


def acoustic_features(
    audio: np.ndarray, words: list[dict]
) -> tuple[list[float], list[float]]:
    """(ASR-segmented features, VAD-segmented features) for one turn. ``words`` only feeds the first vector."""
    feats = mfcc_turn(audio)
    if len(feats) < MIN_FRAMES:
        nan = float("nan")
        return [nan] * len(NAMES_ASR), [nan] * len(NAMES_VAD)
    sa = asr_segments(words, len(feats))
    va = summarise(
        pair_scores(feats, sa, KS_ASR),
        KS_ASR,
        _halves(feats, words, sa),
        len(sa),
        float(sum(b - a for a, b in sa) * FRAME_S),
    )
    sv = vad_segments(audio)
    vv = summarise(
        pair_scores(feats, sv, KS_VAD),
        KS_VAD,
        None,
        len(sv),
        float(sum(b - a for a, b in sv) * FRAME_S),
    )
    return va, vv
