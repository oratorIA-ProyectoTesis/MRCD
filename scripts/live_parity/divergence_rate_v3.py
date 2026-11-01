"""Rate of long sessions whose committed frames differ from whole-file extraction: live_cfg_v2 vs live_cfg_v3.

Sessions: ``synth_session.build_session(10.5, seed)`` for seeds FIXED BEFORE RUNNING (below), none of them used in
any earlier study (earlier: 1-6, 21-26). Seeds 23 and 26 are the two known failing sessions and are reported
separately (``--known``), never mixed into the fresh-seed rate.

Per (seed, partition): offline extraction once, then the tracker with viterbi_carry=False (v2 numerics) and
True (v3). Criterion: committed frames (all 5 columns) not equal to the offline frame (exact), counted before
``finalize``; the count after ``finalize`` is also stored. Output: one JSON line per job + a summary with
Clopper-Pearson 95% intervals for "session diverges in at least one partition" and for "(session, partition) diverges".

usage: python divergence_rate_v3.py --out X.jsonl --workers 18 [--known]
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
# Fixed on 2026-10-30 BEFORE the first run of this script (see the commit that adds it).
FRESH_SEEDS = tuple(range(101, 113))  # 12 sessions
KNOWN_SEEDS = (23, 26)
PARTS = ("tick", "rand", "mix")


def clopper_pearson(k, n, alpha=0.05):
    from scipy.stats import beta

    lo = 0.0 if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - alpha / 2, k + 1, n - k))
    return lo, hi


def job(args):
    seed, part = args
    import warnings

    warnings.simplefilter("ignore")
    from margin_study_v2 import feed
    from synth_session import build_session

    from mrcd.extractors.acoustic import extract_acoustic
    from mrcd.live.session_features import SessionFeatureTracker

    audio = build_session(10.5, seed)[0]
    off = extract_acoustic(audio, use_silero=True).frame_matrix()
    out = {"seed": seed, "part": part, "duration_s": len(audio) / SR}
    for name, carry in (("v2", False), ("v3", True)):
        tr = SessionFeatureTracker(f0_center=0.0, left_margin_s=3.0, right_margin_s=4.0, viterbi_carry=carry)
        t0 = time.time()
        feed(tr, audio, part, seed)
        F = tr.frame_matrix()
        d = np.abs(F - off[: len(F)]).max(1) > 0
        tr.finalize()
        G = tr.frame_matrix()
        dg = np.abs(G - off[: len(G)]).max(1) > 0
        out[name] = {
            "committed_bad": int(d.sum()),
            "final_bad": int(dg.sum()),
            "n_frames": len(G),
            "max_abs_f0_st": float(np.abs(F[:, 1] - off[: len(F), 1]).max()),
            "first_bad_s": (np.nonzero(dg)[0] * 0.01).round(2).tolist()[:5],
            "elapsed_s": round(time.time() - t0, 1),
        }
    return out


def summarize(rows, label):
    res = {"label": label, "sessions": sorted({r["seed"] for r in rows}), "jobs": len(rows)}
    for v in ("v2", "v3"):
        by_seed = {}
        for r in rows:
            by_seed.setdefault(r["seed"], []).append(r[v]["final_bad"] > 0)
        ns = len(by_seed)
        ks = sum(any(b) for b in by_seed.values())
        kj = sum(r[v]["final_bad"] > 0 for r in rows)
        res[v] = {
            "sessions_diverging": f"{ks}/{ns}",
            "session_rate": ks / ns if ns else None,
            "session_ci95_clopper_pearson": clopper_pearson(ks, ns) if ns else None,
            "session_partition_diverging": f"{kj}/{len(rows)}",
            "job_rate_ci95": clopper_pearson(kj, len(rows)) if rows else None,
            "max_frames_in_a_job": max((r[v]["final_bad"] for r in rows), default=0),
            "max_abs_f0_st": max((r[v]["max_abs_f0_st"] for r in rows), default=0.0),
        }
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--known", action="store_true", help="run seeds 23 and 26 instead of the fresh seeds")
    ap.add_argument("--summary-only", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    seeds = KNOWN_SEEDS if a.known else FRESH_SEEDS
    if not a.summary_only:
        jobs = [(s, p) for s in seeds for p in PARTS]
        with Pool(a.workers) as pool, out.open("w", encoding="utf8") as fh:
            for r in pool.imap_unordered(job, jobs):
                fh.write(json.dumps(r) + "\n")
                fh.flush()
                print(json.dumps(r), flush=True)
    rows = [json.loads(line) for line in out.read_text("utf8").splitlines() if line.strip()]
    summ = summarize(rows, "known" if a.known else "fresh")
    out.with_suffix(".summary.json").write_text(json.dumps(summ, indent=1), "utf8")
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
