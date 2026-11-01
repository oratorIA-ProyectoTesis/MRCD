"""Does the v3 residual depend on the length of the degenerate stretch? R=4 s (default) vs R=30 s (>= any silence).

REVISION4 section 1.2 refuted "the residual cannot be bounded by a finite margin": seed 111 truncated at 360 s is exact with R=30 s.
This script re-tests that with NEW sessions. Seeds are FIXED BEFORE RUNNING (this file is committed before the first run):
301-312 (never used in any earlier study; earlier: 1-6, 21-26, 101-112, 201-208). Partition: tick (0.5 s), L = 3 s,
viterbi_carry=True, exact 5-column criterion against ``extract_acoustic`` of the whole session, counted after ``finalize``.
Jobs: every seed at R=4 and R=30 (the generator draws silences uniformly in 0.5-30 s, so R=30 s is >= every silence);
R=12 only for sessions whose R=4 run diverges (declared post-hoc diagnostic, dose-response).
For each diverging job the stretch of the generator layout that contains the first bad frame is stored (kind, start, length).
Also stored: wall time of the job (latency/compute cost proxy; the commit lag itself equals R).

usage: python residual_vs_right_margin.py --out X.jsonl --workers 4
"""

import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import _pin_omp  # noqa: E402,F401

REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))
import numpy as np  # noqa: E402

SR = 16_000
SEEDS = tuple(range(301, 313))
R_DEFAULT, R_LARGE, R_MID = 4.0, 30.0, 12.0


def stretch_of(layout, t):
    start = 0.0
    for kind, name, secs in layout:
        if start <= t < start + secs:
            return {"kind": kind, "start_s": round(start, 2), "len_s": round(secs, 2), "offset_in_stretch_s": round(t - start, 2)}
        start += secs
    return None


def job(args):
    seed, R = args
    import warnings

    warnings.simplefilter("ignore")
    from margin_study_v2 import feed
    from synth_session import build_session

    from mrcd.extractors.acoustic import extract_acoustic
    from mrcd.live.session_features import SessionFeatureTracker

    audio, info = build_session(10.5, seed)
    off = extract_acoustic(audio, use_silero=True).frame_matrix()
    tr = SessionFeatureTracker(f0_center=0.0, left_margin_s=3.0, right_margin_s=R, viterbi_carry=True)
    t0 = time.time()
    feed(tr, audio, "tick", seed)
    F = tr.frame_matrix()
    dc = np.abs(F - off[: len(F)]).max(1) > 0
    tr.finalize()
    G = tr.frame_matrix()
    d = np.abs(G - off[: len(G)]).max(1) > 0
    bad = np.nonzero(d)[0] * 0.01
    longest = max((s for k, _, s in info["layout"] if k == "silence"), default=0.0)
    return {
        "seed": seed,
        "R": R,
        "committed_bad": int(dc.sum()),
        "final_bad": int(d.sum()),
        "first_bad_s": bad.round(2).tolist()[:6],
        "stretch_of_first_bad": stretch_of(info["layout"], float(bad[0])) if len(bad) else None,
        "longest_silence_s": round(float(longest), 2),
        "len_equal": len(G) == len(off),
        "sec": round(time.time() - t0, 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--resume", action="store_true", help="keep the rows already in --out and run only the missing jobs")
    a = ap.parse_args()
    out = Path(a.out)
    rows = [json.loads(x) for x in out.read_text("utf8").splitlines() if x.strip()] if a.resume and out.exists() else []
    done = {(r["seed"], r["R"]) for r in rows}
    with Pool(a.workers) as pool, out.open("a" if a.resume else "w", encoding="utf8") as fh:
        jobs = [(s, r) for s in SEEDS for r in (R_LARGE, R_DEFAULT) if (s, r) not in done]  # long jobs first
        for r in pool.imap_unordered(job, jobs):
            rows.append(r)
            fh.write(json.dumps(r) + "\n")
            fh.flush()
            print(json.dumps(r), flush=True)
        div = sorted({r["seed"] for r in rows if r["R"] == R_DEFAULT and r["final_bad"] > 0})
        for r in pool.imap_unordered(job, [(s, R_MID) for s in div if (s, R_MID) not in done]):
            rows.append(r)
            fh.write(json.dumps(r) + "\n")
            fh.flush()
            print(json.dumps(r), flush=True)


if __name__ == "__main__":
    main()
