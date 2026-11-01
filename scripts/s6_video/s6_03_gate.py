"""S6 step 3: audio-visual admission gate over all units (label-free), with coverage by video and speaker.

Reads work/units.csv (the label column is NOT passed to the gate; it is joined only for reporting totals).
usage: python s6_03_gate.py [--unit-min-r 0.30] [--tag main]
"""

import argparse
import csv
import json
import subprocess
from collections import defaultdict

import numpy as np
from s6_common import OUT, WORK, video_paths

from mrcd.av.config import GATE, config_hash
from mrcd.av.faces import load_tracks, to_face_series
from mrcd.av.gate import analyze_video, speech_envelope

ap = argparse.ArgumentParser()
ap.add_argument("--unit-min-r", type=float, default=GATE["unit_min_r"])
ap.add_argument("--tag", default="main")
ap.add_argument("--merge", action="store_true", help="amendment A2: merge tracks")
a = ap.parse_args()
cfg = dict(GATE, unit_min_r=a.unit_min_r)
if a.merge:
    cfg["merge_tracks"] = True
import hashlib
CH = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()

units = list(csv.DictReader((WORK / "units.csv").open(encoding="utf-8")))
by_video = defaultdict(list)
for u in units:
    by_video[u["video"]].append(u)
vids = video_paths()
rows, vrows = [], []
for k, (stem, us) in enumerate(sorted(by_video.items())):
    z = WORK / "tracks" / f"{stem}.npz"
    if not z.exists():
        for u in us:
            rows.append({"unit_id": u["unit_id"], "admitted": False, "primary_reason": "video_tracks_missing", "reasons": ["video_tracks_missing"]})
        continue
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(vids[stem]), "-f", "f32le", "-ac", "1", "-ar", "16000", "-"], capture_output=True, check=True).stdout
    audio = np.frombuffer(raw, np.float32)
    d = load_tracks(z)
    env = speech_envelope(audio, 16000, float(d["fps"]), int(d["n"]))
    fs = to_face_series(d, env)
    r, v = analyze_video(fs, [(u["unit_id"], float(u["t0"]), float(u["t1"])) for u in us], seed=20261101 + k, cfg=cfg)
    rows += r
    vrows.append({"video": stem, "speaker": us[0]["speaker"], "n_units": len(us), "n_admitted": sum(x["admitted"] for x in r), **{kk: vv for kk, vv in v.items()}})
    print(stem, vrows[-1]["n_admitted"], "/", len(us), v.get("reason"), flush=True)

lab = {u["unit_id"]: (int(u["fpause"]), u["speaker"], u["video"]) for u in units}
for r in rows:
    r["fpause"], r["speaker"], r["video"] = lab[r["unit_id"]]
    r["reasons"] = ";".join(r["reasons"])
out = WORK / f"gate_units_{a.tag}.json"
out.write_text(json.dumps({"config_hash": CH, "unit_min_r": a.unit_min_r, "rows": rows, "videos": vrows}, default=float))
# small coverage tables (counts only; safe to commit)
def cov(key):
    t = defaultdict(lambda: [0, 0, 0, 0])
    for r in rows:
        c = t[r[key]]
        c[0] += 1; c[1] += int(r["admitted"]); c[2] += r["fpause"]; c[3] += int(r["admitted"]) * r["fpause"]
    return t
for key in ("video", "speaker"):
    with (OUT / f"coverage_by_{key}_{a.tag}.csv").open("w", newline="") as f:
        w = csv.writer(f); w.writerow([key, "units", "admitted", "pos_units", "pos_admitted"])
        for k2, c in sorted(cov(key).items()):
            w.writerow([k2, *c])
reasons = defaultdict(int)
for r in rows:
    reasons[r["primary_reason"].split(":")[0]] += 1
summ = {"tag": a.tag, "units": len(rows), "admitted": sum(r["admitted"] for r in rows),
        "pos_total": sum(r["fpause"] for r in rows), "pos_admitted": sum(r["fpause"] for r in rows if r["admitted"]),
        "speakers_admitted": len({r["speaker"] for r in rows if r["admitted"]}),
        "videos_with_admitted": len({r["video"] for r in rows if r["admitted"]}),
        "video_sync_ok": sum(1 for v in vrows if v.get("reason") == "ok"), "videos": len(vrows),
        "primary_reasons": dict(reasons), "config_hash": CH, "merge_tracks": bool(a.merge)}
(OUT / f"gate_summary_{a.tag}.json").write_text(json.dumps(summ, indent=1))
print(json.dumps(summ, indent=1))
