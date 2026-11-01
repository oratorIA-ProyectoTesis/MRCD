"""Technical admission gate for audio-visual units: active speaker, identity consistency and synchrony.

The gate is label-free by construction: it receives only face/lip series, the speech envelope and unit times.
Absence of a face or of a sync estimate is reported as an exclusion with a reason, never as a zero feature.
Nothing here infers psychological state; the lip aperture is a geometric measure.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mrcd.av.config import GATE


@dataclass
class FaceSeries:
    """Per-video, per-track series on a common frame grid (fps)."""

    fps: float
    lip: (
        np.ndarray
    )  # (Tr, n) inner-lip aperture / face height, NaN when the track is absent
    present: np.ndarray  # (Tr, n) bool
    frontal: np.ndarray  # (Tr, n) bool
    shape: np.ndarray  # (Tr, n, K) landmark-ratio descriptor, NaN when absent
    env: np.ndarray  # (n,) speech log-energy envelope on the same grid

    @property
    def n(self) -> int:
        return self.present.shape[1] if self.present.ndim == 2 else 0


def speech_envelope(
    audio: np.ndarray, sr: int, fps: float, n_frames: int
) -> np.ndarray:
    """Log-RMS envelope sampled at video frame times (40 ms window centred on each frame time)."""
    audio = np.asarray(audio, np.float64)
    half = int(0.020 * sr)
    out = np.full(n_frames, np.nan)
    c2 = np.concatenate([[0.0], np.cumsum(audio**2)])
    for k in range(n_frames):
        c = int(round(k / fps * sr))
        a, b = max(c - half, 0), min(c + half, len(audio))
        if b - a >= 8:
            out[k] = np.log10(np.sqrt((c2[b] - c2[a]) / (b - a)) + 1e-6)
    return out


def lag_correlation(
    lip: np.ndarray, env: np.ndarray, max_lag: int, min_overlap: int
) -> np.ndarray:
    """Pearson r between lip[t + s] and env[t] for s in [-max_lag, max_lag].

    Positive s means the lip movement happens LATER than the sound (video late). NaN where the overlap is short.
    """
    n = len(lip)
    r = np.full(2 * max_lag + 1, np.nan)
    for j, s in enumerate(range(-max_lag, max_lag + 1)):
        a0, a1 = max(0, -s), min(n, n - s)  # t range so that t+s in [0, n)
        if a1 - a0 < min_overlap:
            continue
        x, y = lip[a0 + s : a1 + s], env[a0:a1]
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() < min_overlap:
            continue
        x, y = x[ok], y[ok]
        sx, sy = x.std(), y.std()
        if sx < 1e-9 or sy < 1e-9:
            continue
        r[j] = float(np.mean((x - x.mean()) * (y - y.mean())) / (sx * sy))
    return r


def peak_offset_frames(r: np.ndarray) -> tuple[float, float]:
    """(offset in frames with parabolic refinement, peak r). (nan, nan) if no valid lag."""
    if not np.isfinite(r).any():
        return float("nan"), float("nan")
    L = (len(r) - 1) // 2
    j = int(np.nanargmax(r))
    off = float(j - L)
    if 0 < j < len(r) - 1 and np.isfinite(r[j - 1]) and np.isfinite(r[j + 1]):
        den = r[j - 1] - 2 * r[j] + r[j + 1]
        if den < -1e-12:
            off += float(0.5 * (r[j - 1] - r[j + 1]) / den)
    return off, float(r[j])


def r_near(r: np.ndarray, center_frames: float, tol: int = 1) -> float:
    """Best r within +-tol frames of a given lag."""
    L = (len(r) - 1) // 2
    c = int(round(center_frames))
    idx = [i for i in range(c - tol, c + tol + 1) if -L <= i <= L]
    vals = [r[i + L] for i in idx if np.isfinite(r[i + L])]
    return float(max(vals)) if vals else float("nan")


def surrogate_pvalue(lip_seg, env, i0, i1, center, rng, cfg=GATE) -> float:
    """P(null >= observed) with the audio envelope taken from other times of the same video.

    Null statistic = best r within +-1 frame of `center`, exactly as for the observed one.
    """
    n, m = len(env), i1 - i0
    r_obs = r_near(
        lag_correlation(
            lip_seg, env[i0:i1], cfg["lag_search_frames"], cfg["min_overlap_frames"]
        ),
        center,
    )
    if not np.isfinite(r_obs):
        return float("nan")
    guard = int(3 * 25)
    cands = [s for s in range(0, n - m + 1) if s + m <= i0 - guard or s >= i1 + guard]
    if len(cands) < 5:
        return float("nan")
    starts = rng.choice(
        cands,
        size=min(cfg["n_surrogates"], len(cands)),
        replace=len(cands) < cfg["n_surrogates"],
    )
    ge = 0
    for s in starts:
        rr = r_near(
            lag_correlation(
                lip_seg,
                env[s : s + m],
                cfg["lag_search_frames"],
                cfg["min_overlap_frames"],
            ),
            center,
        )
        if np.isfinite(rr) and rr >= r_obs:
            ge += 1
    return float((1 + ge) / (1 + len(starts)))


def identity_drift(shape_valid: np.ndarray, min_half: int = 10) -> float:
    """Descriptor drift between the two halves of a track segment, in robust SDs (mean over dims). NaN if too short.

    This checks that the SAME face is followed through the unit; it is not biometric identification.
    """
    m = len(shape_valid)
    if m < 2 * min_half:
        return float("nan")
    a, b = shape_valid[: m // 2], shape_valid[m // 2 :]
    ma, mb = np.median(a, 0), np.median(b, 0)
    pooled = np.concatenate([a - ma, b - mb])
    mad = 1.4826 * np.median(np.abs(pooled), 0)
    floor = 0.02 * np.abs((ma + mb) / 2) + 1e-6
    return float(np.mean(np.abs(ma - mb) / np.maximum(mad, floor)))


def active_track(fs: FaceSeries, i0: int, i1: int) -> int | None:
    ok = fs.present[:, i0:i1] & fs.frontal[:, i0:i1]
    if ok.shape[0] == 0 or ok.sum() == 0:
        return None
    return int(np.argmax(ok.sum(1)))


def merged_unit_series(fs: FaceSeries, i0: int, i1: int):
    """Amendment A2: one face per frame regardless of tracker id.

    Returns (lip (m,), shape (m,K), single_front (m,) bool, multi (m,) bool). A frame counts when exactly one face
    is present and it is semi-frontal; a frame with two or more faces is 'multi' (active speaker ambiguous).
    Re-acquisition of the same person under a new track id therefore no longer looks like absence.
    """
    pres, fr = fs.present[:, i0:i1], fs.frontal[:, i0:i1]
    cnt = pres.sum(0)
    single_front = (cnt == 1) & (pres & fr).any(0)
    idx = np.argmax(pres, axis=0)
    cols = np.arange(pres.shape[1])
    lip = np.where(single_front, fs.lip[:, i0:i1][idx, cols], np.nan)
    shape = np.where(single_front[:, None], fs.shape[:, i0:i1][idx, cols], np.nan)
    return lip, shape, single_front, cnt >= 2


def local_unit_stats(fs: FaceSeries, t0: float, t1: float, cfg=GATE) -> dict:
    """Per-unit quantities that do not need the video offset."""
    i0, i1 = max(int(np.floor(t0 * fs.fps)), 0), min(int(np.ceil(t1 * fs.fps)), fs.n)
    out = {"i0": i0, "i1": i1, "n_frames": max(i1 - i0, 0), "track": None}
    if out["n_frames"] < 2 or fs.present.shape[0] == 0:
        out.update(face_coverage=0.0, second_face_share=0.0, drift=float("nan"))
        return out
    if cfg.get("merge_tracks", False):
        lip, shape, sf, multi = merged_unit_series(fs, i0, i1)
        if not sf.any():
            out.update(face_coverage=0.0, second_face_share=float(multi.mean()), drift=float("nan"))
            return out
        r = lag_correlation(lip, fs.env[i0:i1], cfg["lag_search_frames"], cfg["min_overlap_frames"])
        off, rpk = peak_offset_frames(r)
        out.update(
            track=-1,
            face_coverage=float(sf.mean()),
            second_face_share=float(multi.mean()),
            drift=identity_drift(shape[sf]),
            local_offset_frames=off,
            local_r_peak=rpk,
            _lip=lip,
            _rcurve=r,
        )
        return out
    tr = active_track(fs, i0, i1)
    if tr is None:
        out.update(face_coverage=0.0, second_face_share=0.0, drift=float("nan"))
        return out
    cover = float((fs.present[tr, i0:i1] & fs.frontal[tr, i0:i1]).mean())
    others = [k for k in range(fs.present.shape[0]) if k != tr]
    second = 0.0
    if others:
        both = fs.present[tr, i0:i1][None] & fs.present[others, i0:i1]
        second = float(both.any(0).mean())
    sv = fs.shape[tr, i0:i1][fs.present[tr, i0:i1] & fs.frontal[tr, i0:i1]]
    lip = np.where(fs.present[tr, i0:i1], fs.lip[tr, i0:i1], np.nan)
    r = lag_correlation(lip, fs.env[i0:i1], cfg["lag_search_frames"], cfg["min_overlap_frames"])
    off, rpk = peak_offset_frames(r)
    out.update(
        track=tr,
        face_coverage=cover,
        second_face_share=second,
        drift=identity_drift(sv),
        local_offset_frames=off,
        local_r_peak=rpk,
        _lip=lip,
        _rcurve=r,
    )
    return out


def video_offset(local: list[dict], fps: float, rng, cfg=GATE) -> dict:
    """Video-level offset from the frame-weighted mean of unit lag-correlation curves (amendment A1).

    Contributing units: face coverage and second-face share pass C1/C2 (no selection on correlation strength).
    CI: percentile bootstrap over contributing units.
    """
    cand = [
        u
        for u in local
        if u.get("track") is not None
        and u.get("face_coverage", 0) >= cfg["min_face_coverage"]
        and u.get("second_face_share", 1) <= cfg["max_second_face_share"]
        and u.get("_rcurve") is not None
        and np.isfinite(u["_rcurve"]).any()
    ]
    out = {"n_contrib": len(cand), "offset_ms": float("nan"), "ci_lo_ms": float("nan"), "ci_hi_ms": float("nan")}
    if len(cand) < cfg["video_min_units_for_offset"] or sum(u["n_frames"] for u in cand) < cfg["video_min_pooled_frames"]:
        out["reason"] = "offset_unestimable_few_units"
        return out
    curves = np.array([np.nan_to_num(u["_rcurve"], nan=0.0) for u in cand])
    w = np.array([u["n_frames"] for u in cand], float)

    def pooled_offset(idx):
        c = (curves[idx] * w[idx, None]).sum(0) / w[idx].sum()
        return peak_offset_frames(c)[0] * 1000.0 / fps

    all_idx = np.arange(len(cand))
    out["offset_ms"] = float(pooled_offset(all_idx))
    bs = np.array(
        [pooled_offset(rng.choice(all_idx, size=len(all_idx), replace=True)) for _ in range(cfg["n_offset_bootstrap"])]
    )
    lo, hi = np.percentile(bs, [50 * (1 - cfg["offset_ci"]), 100 - 50 * (1 - cfg["offset_ci"])])
    out["ci_lo_ms"], out["ci_hi_ms"] = float(lo), float(hi)
    ok = abs(out["offset_ms"]) <= cfg["video_max_abs_offset_ms"] and (hi - lo) <= cfg["video_max_ci_width_ms"]
    out["reason"] = "ok" if ok else "video_offset_out_of_range_or_imprecise"
    return out


def _nz(x):
    return None if x is None or not np.isfinite(x) else float(x)


def analyze_video(
    fs: FaceSeries, units: list[tuple[str, float, float]], seed: int, cfg=GATE
) -> tuple[list[dict], dict]:
    """Gate every (unit_id, t0, t1) of one video. Returns (per-unit rows, video row). Label-free."""
    rng = np.random.default_rng(seed)
    local = []
    for uid, t0, t1 in units:
        s = local_unit_stats(fs, t0, t1, cfg)
        s["unit_id"], s["t0"], s["t1"] = uid, t0, t1
        local.append(s)
    vrow = video_offset(local, fs.fps, rng, cfg)
    vrow_ok = vrow.get("reason") == "ok"
    center = (
        vrow["offset_ms"] * fs.fps / 1000.0
        if np.isfinite(vrow["offset_ms"])
        else float("nan")
    )
    rows = []
    for s in local:
        reasons = []
        dur = s["t1"] - s["t0"]
        if dur < cfg["min_unit_s"]:
            reasons.append("unit_too_short")
        if s.get("face_coverage", 0.0) < cfg["min_face_coverage"]:
            reasons.append("face_not_visible_enough")
        if s.get("second_face_share", 0.0) > cfg["max_second_face_share"]:
            reasons.append("second_face_present_active_speaker_ambiguous")
        d = s.get("drift", float("nan"))
        if "face_not_visible_enough" not in reasons and not (
            np.isfinite(d) and d <= cfg["max_identity_drift"]
        ):
            reasons.append("identity_inconsistent_or_unverifiable")
        r_at, p = float("nan"), float("nan")
        if not vrow_ok:
            reasons.append("video_sync_unverified:" + vrow.get("reason", "unknown"))
        elif s.get("track") is not None and "face_not_visible_enough" not in reasons:
            lip = s["_lip"]
            i0, i1 = s["i0"], s["i1"]
            r = lag_correlation(
                lip, fs.env[i0:i1], cfg["lag_search_frames"], cfg["min_overlap_frames"]
            )
            r_at = r_near(r, center)
            p = surrogate_pvalue(lip, fs.env, i0, i1, center, rng, cfg)
            if not (np.isfinite(r_at) and r_at >= cfg["unit_min_r"]):
                reasons.append("lip_audio_correlation_below_threshold")
            elif not (np.isfinite(p) and p <= cfg["unit_surrogate_alpha"]):
                reasons.append("correlation_not_above_surrogate_null")
        lo = _nz(s.get("local_offset_frames"))
        rows.append(
            {
                "unit_id": s["unit_id"],
                "admitted": not reasons,
                "reasons": reasons,
                "primary_reason": reasons[0] if reasons else "admitted",
                "face_coverage": round(float(s.get("face_coverage", 0.0)), 4),
                "second_face_share": round(float(s.get("second_face_share", 0.0)), 4),
                "identity_drift": None if _nz(d) is None else round(float(d), 4),
                "local_offset_ms": None if lo is None else round(lo * 1000 / fs.fps, 1),
                "local_r_peak": None
                if _nz(s.get("local_r_peak")) is None
                else round(float(s["local_r_peak"]), 4),
                "r_at_video_offset": None
                if _nz(r_at) is None
                else round(float(r_at), 4),
                "surrogate_p": None if _nz(p) is None else round(float(p), 4),
                "track": s.get("track"),
            }
        )
    return rows, vrow
