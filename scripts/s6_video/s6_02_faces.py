"""S6 step 2: face tracks for every local EPIC video at native fps (serial, resumable). Heavy: run only after S1.

Outputs (gitignored, never committed): work/tracks/<video>.npz with per-track scalars; work/faces_timing.json.
"""

import json
import time

from s6_common import FACE_MODEL, WORK, video_paths

from mrcd.av.faces import extract_video, save_tracks

D = WORK / "tracks"
D.mkdir(exist_ok=True)
timing = json.loads((WORK / "faces_timing.json").read_text()) if (WORK / "faces_timing.json").exists() else {}
for stem, p in video_paths().items():
    out = D / f"{stem}.npz"
    if out.exists():
        continue
    t0 = time.perf_counter()
    d = extract_video(str(p), FACE_MODEL)
    save_tracks(out, d)
    timing[stem] = {"frames": int(d["n"]), "wall_s": round(time.perf_counter() - t0, 1), "tracks": int(d["vec"].shape[0])}
    (WORK / "faces_timing.json").write_text(json.dumps(timing))
    print(stem, timing[stem], flush=True)
