"""2-D (left x right) margin study of SessionFeatureTracker over ALL 98 AMERESCO clips.

Criterion: number of committed frames (any of the 5 columns) that differ from whole-file extraction.
A margin pair is acceptable only when it is 0 on every clip. Right margins >= 4 s can hide failures
(failing frames are simply not committed), so the committed frame count is reported too.

usage: python margin_study.py --cfgs 1:3,3:3 --tag A [--concat]
"""

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import _pin_omp  # noqa: E402,F401

REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

from mrcd.extractors.acoustic import extract_acoustic  # noqa: E402
from mrcd.live.session_features import SessionFeatureTracker  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfgs", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument(
        "--concat",
        action="store_true",
        help="also an 8-clip concatenation (long session)",
    )
    ap.add_argument("--out", default=str(REPO / "results" / "paridad_live_20261030"))
    a = ap.parse_args()
    cfgs = [tuple(float(x) for x in c.split(":")) for c in a.cfgs.split(",")]
    man = json.loads(
        (REPO / "results/baselines_20261030/ameresco/manifest.json").read_text("utf8")
    )
    items = [(m["unit_id"], m["wav"]) for m in man]
    res = {f"{L:g}:{R:g}": {"clips": {}} for L, R in cfgs}
    audios = {}

    def run(name, audio, off):
        for L, R in cfgs:
            tr = SessionFeatureTracker(f0_center=0.0, left_margin_s=L, right_margin_s=R)
            for i in range(0, len(audio), 8000):
                tr.append(audio[i : i + 8000])
            F = tr.frame_matrix()
            bad = int((np.abs(F - off[: len(F)]).max(1) > 0).sum())
            maxf0 = float(np.abs(F[:, 1] - off[: len(F), 1]).max()) if len(F) else 0.0
            res[f"{L:g}:{R:g}"]["clips"][name] = {
                "committed": len(F),
                "bad_frames": bad,
                "max_abs_f0_st": maxf0,
            }
        print(
            name,
            {k: v["clips"][name]["bad_frames"] for k, v in res.items()},
            flush=True,
        )

    t0 = time.time()
    for name, wav in items:
        audio, _ = sf.read(wav, dtype="float32")
        audios[name] = audio
        run(name, audio, extract_acoustic(audio, use_silero=True).frame_matrix())
    if a.concat:
        rng = np.random.default_rng(0)
        pick = rng.choice(len(items), 8, replace=False)
        audio = np.concatenate([audios[items[i][0]] for i in pick])
        run("concat8", audio, extract_acoustic(audio, use_silero=True).frame_matrix())
    for k, v in res.items():
        c = v["clips"]
        v["summary"] = {
            "clips": len(c),
            "clips_with_bad_frames": sum(x["bad_frames"] > 0 for x in c.values()),
            "bad_frames": sum(x["bad_frames"] for x in c.values()),
            "committed_frames": sum(x["committed"] for x in c.values()),
            "max_abs_f0_st": max(x["max_abs_f0_st"] for x in c.values()),
        }
    res["elapsed_s"] = time.time() - t0
    Path(a.out, f"margin_study_{a.tag}.json").write_text(
        json.dumps(res, indent=1), "utf8"
    )
    for k, v in res.items():
        if k != "elapsed_s":
            print(k, v["summary"])


if __name__ == "__main__":
    main()
