"""Technical verification of one Via B recording (audio + optional video). Stores NO content.

Reports only numbers: duration, sample rate, channels, fps, resolution, level statistics, clap (claqueta) times,
audio-video offset estimated from the clap, optional face visibility and S6-gate admission. Nothing here reads, keeps
or prints speech content, frames or landmarks, and nothing infers a psychological state: the face check is the
technical admission gate of ``mrcd.av`` (face present, frontal, single, lip/audio synchrony).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

REQUIREMENTS = {
    "min_duration_s": 120.0,
    "min_audio_sr": 16000,
    "min_video_height": 720,
    "target_fps": 25.0,
    "fps_range": (24.0, 30.0),
    "max_av_duration_diff_s": 0.5,
    "peak_dbfs_range": (-20.0, -1.0),
    "max_clipped_fraction": 0.001,
    "max_noise_floor_dbfs": -50.0,
    "max_abs_clap_offset_ms": 120.0,
    "max_clap_drift_ms": 60.0,
    "clap_search_s": 45.0,
    "min_face_coverage": 0.80,
    "min_admitted_fraction": 0.80,
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ------------------------------------------------------------------------------------------- pure helpers
def level_stats(x: np.ndarray, sr: int) -> dict:
    """Peak, RMS, clipping, noise floor (10th percentile of 50 ms RMS) in dBFS for a float signal in [-1, 1]."""
    x = np.asarray(x, np.float64)
    if x.size == 0:
        return {"available": False, "reason": "no_audio_samples"}
    eps = 1e-9
    peak = float(np.max(np.abs(x)))
    win = max(int(0.05 * sr), 1)
    n = x.size // win
    rms_w = (
        np.sqrt(np.mean(x[: n * win].reshape(n, win) ** 2, axis=1))
        if n
        else np.array([np.sqrt(np.mean(x**2))])
    )
    db = lambda v: float(20 * np.log10(max(v, eps)))  # noqa: E731
    return {
        "available": True,
        "peak_dbfs": db(peak),
        "rms_dbfs": db(float(np.sqrt(np.mean(x**2)))),
        "clipped_fraction": float(np.mean(np.abs(x) >= 0.999)),
        "noise_floor_dbfs": db(float(np.percentile(rms_w, 10))),
        "speech_level_dbfs": db(float(np.percentile(rms_w, 90))),
        "dc_offset": float(np.mean(x)),
    }


def find_claps(
    x: np.ndarray, sr: int, *, search_s: float, min_ratio: float = 12.0
) -> dict:
    """Locate the strongest transient in the first and in the last ``search_s`` seconds.

    A clap is a short event whose 5 ms envelope exceeds ``min_ratio`` times the median envelope of that window.
    Returns times (s) or None with a reason; never a default value.
    """
    x = np.asarray(x, np.float64)
    hop = max(int(0.005 * sr), 1)
    n = x.size // hop
    if n < 10:
        return {"start": None, "end": None, "reason": "too_short"}
    env = np.sqrt(np.mean(x[: n * hop].reshape(n, hop) ** 2, axis=1))
    out: dict = {"reason": None}
    for name, sl in (
        ("start", slice(0, int(search_s / 0.005))),
        ("end", slice(max(n - int(search_s / 0.005), 0), n)),
    ):
        seg = env[sl]
        if seg.size < 10:
            out[name] = None
            continue
        base = float(np.median(seg)) + 1e-9
        k = int(np.argmax(seg))
        out[name] = (
            float((sl.start + k) * 0.005) if seg[k] / base >= min_ratio else None
        )
    if out["start"] is None or out["end"] is None:
        out["reason"] = "clap_not_found:" + ",".join(
            k for k in ("start", "end") if out[k] is None
        )
    return out


def clap_offset_ms(
    clap_t: float | None, motion: np.ndarray, fps: float, *, half_window_s: float = 0.6
) -> dict:
    """Video offset relative to a clap time: peak of the frame-difference series within +-half_window_s.

    Positive = the visual event is LATER than the sound (video late). Resolution is one frame (1000 / fps ms).
    """
    if clap_t is None:
        return {"offset_ms": None, "reason": "no_audio_clap"}
    k0 = int(round(clap_t * fps))
    w = int(round(half_window_s * fps))
    lo, hi = max(k0 - w, 0), min(k0 + w + 1, len(motion))
    if hi - lo < 5:
        return {"offset_ms": None, "reason": "clap_outside_video"}
    seg = motion[lo:hi]
    base = float(np.median(motion)) + 1e-9
    k = int(np.argmax(seg))
    if seg[k] / base < 4.0:
        return {"offset_ms": None, "reason": "no_visual_transient"}
    return {
        "offset_ms": float((lo + k - k0) * 1000.0 / fps),
        "resolution_ms": 1000.0 / fps,
        "reason": None,
    }


def gate_summary(fs, duration_s: float, seed: int = 0) -> dict:
    """Admission of the 3 s evaluation segments under the FROZEN S6 gate; counts only."""
    from mrcd.recording_check.stats import segments
    from mrcd.av.gate import analyze_video

    units = [(f"u{i}", a, b) for i, (a, b) in enumerate(segments(duration_s))]
    rows, v = analyze_video(fs, units, seed=seed)
    n = len(rows)
    adm = sum(r["admitted"] for r in rows)
    reasons: dict[str, int] = {}
    for r in rows:
        reasons[r["primary_reason"]] = reasons.get(r["primary_reason"], 0) + 1
    return {
        "segments": n,
        "admitted": adm,
        "admitted_fraction": (adm / n) if n else None,
        "primary_reasons": reasons,
        "video_offset_ms": v.get("offset_ms")
        if np.isfinite(v.get("offset_ms", np.nan))
        else None,
        "video_offset_ci_ms": [v.get("ci_lo_ms"), v.get("ci_hi_ms")]
        if np.isfinite(v.get("ci_lo_ms", np.nan))
        else None,
        "video_sync_reason": v.get("reason"),
    }


def verdicts(rep: dict, req: dict = REQUIREMENTS) -> dict:
    """Pass/fail/unknown per check; unknown never counts as pass."""
    v: dict[str, bool | None] = {}
    v["duration"] = (
        None
        if rep.get("duration_s") is None
        else rep["duration_s"] >= req["min_duration_s"]
    )
    a = rep.get("audio") or {}
    v["audio_sample_rate"] = None if not a else a["sample_rate"] >= req["min_audio_sr"]
    lv = rep.get("levels") or {}
    if lv.get("available"):
        lo, hi = req["peak_dbfs_range"]
        v["audio_peak_level"] = lo <= lv["peak_dbfs"] <= hi
        v["audio_clipping"] = lv["clipped_fraction"] <= req["max_clipped_fraction"]
        v["audio_noise_floor"] = lv["noise_floor_dbfs"] <= req["max_noise_floor_dbfs"]
    vid = rep.get("video")
    if vid:
        v["video_resolution"] = vid["height"] >= req["min_video_height"]
        lo, hi = req["fps_range"]
        v["video_fps"] = lo <= vid["fps"] <= hi
        v["av_duration"] = (
            None
            if rep.get("av_duration_diff_s") is None
            else rep["av_duration_diff_s"] <= req["max_av_duration_diff_s"]
        )
        cl = rep.get("clap_sync") or {}
        s, e = cl.get("start_offset_ms"), cl.get("end_offset_ms")
        v["clap_offset"] = (
            None if s is None else abs(s) <= req["max_abs_clap_offset_ms"]
        )
        v["clap_drift"] = (
            None if (s is None or e is None) else abs(e - s) <= req["max_clap_drift_ms"]
        )
        g = rep.get("face_gate")
        if g and g.get("available", True) and g.get("admitted_fraction") is not None:
            v["admitted_fraction"] = g["admitted_fraction"] >= req["min_admitted_fraction"]
            v["video_sync_gate"] = g["video_sync_reason"] == "ok"
        else:  # face gate skipped or unavailable: unknown, never a pass
            v["admitted_fraction"] = None
            v["video_sync_gate"] = None
    return v


# --------------------------------------------------------------------------------------------- file work
def _decode(path: Path):
    """One pass over the file: native-rate mono audio, 16 kHz mono audio, per-frame motion series (no frames kept)."""
    import av

    out: dict = {
        "audio_native": [],
        "audio_16k": [],
        "motion": [],
        "video": None,
        "audio": None,
        "duration_s": None,
    }
    with av.open(str(path)) as c:
        a = c.streams.audio[0] if c.streams.audio else None
        v = c.streams.video[0] if c.streams.video else None
        if c.duration:
            out["duration_s"] = c.duration / av.time_base
        rs_n = rs_16 = None
        if a is not None:
            sr = a.codec_context.sample_rate
            out["audio"] = {
                "sample_rate": int(sr),
                "channels": int(a.codec_context.channels),
                "codec": a.codec_context.name,
            }
            rs_n = av.AudioResampler(format="flt", layout="mono", rate=sr)
            rs_16 = av.AudioResampler(format="flt", layout="mono", rate=16000)
        if v is not None:
            fps = float(v.average_rate) if v.average_rate else None
            out["video"] = {
                "width": int(v.codec_context.width),
                "height": int(v.codec_context.height),
                "fps": fps,
                "codec": v.codec_context.name,
            }
        prev = None
        streams = [s for s in (a, v) if s is not None]
        for pkt in c.demux(*streams):
            for fr in pkt.decode():
                if isinstance(fr, av.AudioFrame) and rs_n is not None:
                    for f in rs_n.resample(fr):
                        out["audio_native"].append(f.to_ndarray().reshape(-1))
                    for f in rs_16.resample(fr):
                        out["audio_16k"].append(f.to_ndarray().reshape(-1))
                elif isinstance(fr, av.VideoFrame):
                    g = (
                        fr.reformat(width=64, height=36, format="gray")
                        .to_ndarray()
                        .astype(np.float32)
                    )
                    out["motion"].append(
                        0.0 if prev is None else float(np.mean(np.abs(g - prev)))
                    )
                    prev = g
    out["audio_native"] = (
        np.concatenate(out["audio_native"])
        if out["audio_native"]
        else np.zeros(0, np.float32)
    )
    out["audio_16k"] = (
        np.concatenate(out["audio_16k"])
        if out["audio_16k"]
        else np.zeros(0, np.float32)
    )
    out["motion"] = np.asarray(out["motion"], np.float64)
    return out


def verify_file(
    path: str | Path,
    *,
    face_model: str | None = None,
    run_face: bool = True,
    req: dict = REQUIREMENTS,
) -> dict:
    path = Path(path)
    d = _decode(path)
    rep: dict = {
        "file": path.name,
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "duration_s": d["duration_s"],
        "audio": d["audio"],
        "video": d["video"],
        "stores_content": False,
    }
    if d["audio"] is not None:
        rep["levels"] = level_stats(d["audio_native"], d["audio"]["sample_rate"])
        if d["duration_s"] is None:
            rep["duration_s"] = len(d["audio_native"]) / d["audio"]["sample_rate"]
        rep["audio_duration_s"] = len(d["audio_native"]) / d["audio"]["sample_rate"]
        claps = find_claps(d["audio_16k"], 16000, search_s=req["clap_search_s"])
        rep["claps_audio_s"] = claps
    else:
        rep["levels"] = {"available": False, "reason": "no_audio_stream"}
    if d["video"] is not None and d["video"]["fps"]:
        fps = d["video"]["fps"]
        rep["video_duration_s"] = len(d["motion"]) / fps
        if d["audio"] is not None:
            rep["av_duration_diff_s"] = abs(
                rep["video_duration_s"] - rep["audio_duration_s"]
            )
            cl = rep["claps_audio_s"]
            so = clap_offset_ms(cl.get("start"), d["motion"], fps)
            eo = clap_offset_ms(cl.get("end"), d["motion"], fps)
            rep["clap_sync"] = {
                "start_offset_ms": so["offset_ms"],
                "end_offset_ms": eo["offset_ms"],
                "start_reason": so["reason"],
                "end_reason": eo["reason"],
                "positive_means": "video later than audio",
            }
        if run_face:
            rep["face_gate"] = _face_gate(path, d, face_model)
    rep["verdicts"] = verdicts(rep, req)
    rep["all_checks_passed"] = bool(rep["verdicts"]) and all(
        x is True for x in rep["verdicts"].values()
    )
    rep["checks_unknown"] = sorted(k for k, x in rep["verdicts"].items() if x is None)
    return rep


def _face_gate(path: Path, d: dict, face_model: str | None) -> dict:
    from mrcd.av.faces import extract_video, to_face_series
    from mrcd.av.gate import speech_envelope

    model = face_model or str(
        Path(__file__).resolve().parents[2] / "models" / "face_landmarker.task"
    )
    if not Path(model).is_file():
        return {"available": False, "reason": f"face model not found: {model}"}
    t = extract_video(str(path), model)  # per-frame scalars live only in memory
    env = speech_envelope(d["audio_16k"], 16000, t["fps"], t["n"])
    fs = to_face_series(t, env)
    out = gate_summary(fs, t["n"] / t["fps"])
    pres = t["present"].any(0) if t["present"].size else np.zeros(t["n"], bool)
    front = t["frontal"].any(0) if t["frontal"].size else np.zeros(t["n"], bool)
    out["face_present_fraction"] = float(pres.mean()) if t["n"] else None
    out["face_frontal_fraction"] = float(front.mean()) if t["n"] else None
    out["max_faces_in_frame"] = int(t["n_faces"].max()) if len(t["n_faces"]) else 0
    out["note"] = (
        "technical admission gate only; no emotion, intent or state is inferred"
    )
    return out


def main(argv: list[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("files", nargs="+")
    p.add_argument(
        "--no-face",
        action="store_true",
        help="skip the S6 face gate (slow: about 11 ms per frame)",
    )
    p.add_argument("--face-model", default=None)
    p.add_argument("--json", action="store_true", help="print JSON only")
    a = p.parse_args(argv)
    bad = 0
    for f in a.files:
        rep = verify_file(f, face_model=a.face_model, run_face=not a.no_face)
        if a.json:
            print(json.dumps(rep, indent=2, default=str))
        else:
            print(f"{rep['file']}: {'OK' if rep['all_checks_passed'] else 'REVISAR'}")
            for k, v in rep["verdicts"].items():
                print(f"  {k:22s} {'ok' if v else ('?' if v is None else 'FALLA')}")
        bad += 0 if rep["all_checks_passed"] else 1
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
