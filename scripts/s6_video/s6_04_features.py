"""S6 step 4: build the evaluation dataset on gate-admitted units (A, T, V raw, V speaker-normalised, gate confidence).

Order: V features -> speaker baselines -> deterministic negative subsample -> A and T (ASR cost only for kept units).
The gate decision is read from work/gate_units_main.json; labels are used only for the sha1-negative cap bookkeeping
(the cap itself depends on unit ids, not on features).
"""

import argparse
import csv
import json
import subprocess
import time
from collections import defaultdict

import numpy as np
from s6_common import OUT, WHISPER, WORK, video_paths

from mrcd.av.ablation import subsample_negatives
from mrcd.av.config import ANALYSIS
from mrcd.av.faces import load_tracks
from mrcd.av.features import (
    A_NAMES,
    SIGNALS,
    T_NAMES,
    V_NAMES,
    audio_features,
    speaker_baseline,
    text_features,
    visual_features,
)
from mrcd.extractors.acoustic import extract_acoustic
from mrcd.extractors.linguistic import VerbatimASR, Word, word_features
from mrcd.extractors.prosody import speaker_f0_center

ap = argparse.ArgumentParser()
ap.add_argument("--tag", default="a2")
ap.add_argument("--subset", choices=["gated", "face_only"], default="gated")
args = ap.parse_args()
SYNC_REASONS = ("video_sync_unverified", "lip_audio_correlation_below_threshold", "correlation_not_above_surrogate_null")
gate = json.loads((WORK / f"gate_units_{args.tag}.json").read_text())
G = {r["unit_id"]: r for r in gate["rows"]}
units = {
    u["unit_id"]: u for u in csv.DictReader((WORK / "units.csv").open(encoding="utf-8"))
}
if args.subset == "gated":
    adm = [uid for uid, r in G.items() if r["admitted"]]
else:  # exploratory: C0-C3 only (face visible, single face, consistent identity); synchrony NOT verified
    adm = [uid for uid, r in G.items() if r["reasons"] == "" or all(x.split(":")[0] in SYNC_REASONS for x in r["reasons"].split(";") if x)]
tracks_cache = {}


def unit_signals(uid):
    u, r = units[uid], G[uid]
    d = tracks_cache.get(u["video"])
    if d is None:
        tracks_cache.clear()
        d = tracks_cache[u["video"]] = load_tracks(
            WORK / "tracks" / f"{u['video']}.npz"
        )
    fps = float(d["fps"])
    i0 = max(int(np.floor(float(u["t0"]) * fps)), 0)
    i1 = min(int(np.ceil(float(u["t1"]) * fps)), int(d["n"]))
    k = r["track"]
    if k == -1:  # amendment A2: merged single-face series
        pres = d["present"][:, i0:i1]
        sf = (pres.sum(0) == 1) & (pres & d["frontal"][:, i0:i1]).any(0)
        idx = np.argmax(pres, axis=0)
        v, ok = d["vec"][:, i0:i1][idx, np.arange(pres.shape[1])], sf
    else:
        v, ok = d["vec"][k, i0:i1], d["present"][k, i0:i1]
    # vec columns: 0 lip aperture, 1 jaw open, 2 pucker, 3 pitch, 4 yaw, 5 roll
    return {n: np.where(ok, v[:, j], np.nan) for j, n in enumerate(SIGNALS)}, fps


# 1) per-speaker frame pools from admitted units (label-free)
pool = defaultdict(lambda: defaultdict(list))
sigs = {}
for uid in sorted(adm, key=lambda x: units[x]["video"]):
    sig, fps = unit_signals(uid)
    sigs[uid] = (sig, fps)
    sp = units[uid]["speaker"]
    ok = np.isfinite(sig["ap"])
    for k in SIGNALS:
        pool[sp][k].append(sig[k][ok])
    ang = np.stack([sig[k][ok] for k in ("pitch", "yaw", "roll")], 1)
    pool[sp]["speed"].append(np.sqrt(np.nansum(np.diff(ang, axis=0) ** 2, 1)) * fps)
base = {}
for sp, d in pool.items():
    b = speaker_baseline(
        {k: np.concatenate(v) for k, v in d.items()},
        ANALYSIS["speaker_baseline_min_frames"],
    )
    if b is not None:
        base[sp] = b
# 2) V features
elig, Vn, Vr, reasons = [], {}, {}, defaultdict(int)
for uid in adm:
    sp = units[uid]["speaker"]
    if sp not in base:
        reasons["speaker_baseline_insufficient"] += 1
        continue
    sig, fps = sigs[uid]
    vn, vr = visual_features(sig, fps, base[sp]), visual_features(sig, fps, None)
    if vn is None or vr is None:
        reasons["visual_features_unavailable"] += 1
        continue
    elig.append(uid)
    Vn[uid], Vr[uid] = vn, vr
ids = np.array(elig)
y = np.array([int(units[u]["fpause"]) for u in elig])
keep = subsample_negatives(ids, y, ANALYSIS["negatives_per_positive_cap"])
ids, y = ids[keep], y[keep]
print(
    "eligible",
    len(elig),
    "kept",
    len(ids),
    "pos",
    int(y.sum()),
    dict(reasons),
    flush=True,
)

# 3) A and T (ASR cost only here)
asr = VerbatimASR(str(WHISPER), device="cpu", compute_type="int8")
(WORK / "asr").mkdir(exist_ok=True)
audio_by_video = {}


def video_audio(stem):
    if stem not in audio_by_video:
        audio_by_video.clear()
        cmd = [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(video_paths()[stem]),
            "-f",
            "f32le",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-",
        ]
        audio_by_video[stem] = np.frombuffer(
            subprocess.run(cmd, capture_output=True, check=True).stdout, np.float32
        )
    return audio_by_video[stem]


frames, A, T = {}, {}, {}
t_start = time.perf_counter()
SR = 16000
for n_done, uid in enumerate(sorted(ids, key=lambda x: units[x]["video"])):
    u = units[uid]
    a_all = video_audio(u["video"])
    t0, t1 = float(u["t0"]), float(u["t1"])
    frames[uid] = extract_acoustic(
        a_all[int(t0 * SR) : int(t1 * SR)], SR
    ).frame_matrix()
    cache = WORK / "asr" / f"{uid}.json"
    if cache.exists():
        words = json.loads(cache.read_text())
    else:
        p0, p1 = max(t0 - 1.0, 0.0), min(t1 + 1.0, len(a_all) / SR)
        ws = asr.transcribe_chunked(a_all[int(p0 * SR) : int(p1 * SR)], 8.0, 2.0)
        words = [
            dict(w.to_dict(), start=w.start + p0 - t0, end=w.end + p0 - t0) for w in ws
        ]
        words = [w for w in words if 0.0 <= 0.5 * (w["start"] + w["end"]) <= (t1 - t0)]
        cache.write_text(json.dumps(words))
    W = [Word(**w) for w in words]
    T[uid] = text_features(
        W, t1 - t0, word_features(W) if W else np.zeros((0, 10), np.float32)
    )
    if n_done % 20 == 0:
        print(n_done, len(ids), round(time.perf_counter() - t_start), "s", flush=True)
by_sp = defaultdict(list)
for uid in ids:
    by_sp[units[uid]["speaker"]].append(frames[uid])
center = {sp: speaker_f0_center(np.concatenate(v)) for sp, v in by_sp.items()}
for uid in ids:
    A[uid] = audio_features(frames[uid], center[units[uid]["speaker"]])
conf = np.array(
    [float(np.clip((G[u]["r_at_video_offset"] or 0.0) / 0.6, 0, 1)) for u in ids]
)
np.savez(
    WORK / f"dataset_{args.tag}_{args.subset}.npz",
    ids=ids,
    y=y,
    g=np.array([units[u]["speaker"] for u in ids]),
    A=np.stack([A[u] for u in ids]),
    T=np.stack([T[u] for u in ids]),
    V=np.stack([Vn[u] for u in ids]),
    Vraw=np.stack([Vr[u] for u in ids]),
    conf=conf,
    video=np.array([units[u]["video"] for u in ids]),
    names=np.array([A_NAMES, T_NAMES, V_NAMES], dtype=object),
    elig_g=np.array([units[u]["speaker"] for u in elig]),
    elig_V=np.stack([Vn[u] for u in elig]),
    elig_Vraw=np.stack([Vr[u] for u in elig]),
)
summary = {
    "eligible_after_gate": len(elig),
    "evaluated": len(ids),
    "pos": int(y.sum()),
    "speakers": len({units[u]["speaker"] for u in ids}),
    "pos_speakers": len({units[u]["speaker"] for u, yy in zip(ids, y) if yy}),
    "excluded_after_gate": dict(reasons),
    "gate_config_hash": gate["config_hash"],
    "subset": args.subset,
    "tag": args.tag,
}
(OUT / f"dataset_summary_{args.tag}_{args.subset}.json").write_text(json.dumps(summary, indent=1))
print("done", round(time.perf_counter() - t_start), "s", summary)
