"""Cost / memory of SessionFeatureTracker over a long synthetic+real session (0.5 s ticks).

Reports per-minute mean tick time and tracemalloc / RSS growth, then (optionally) checks committed
frames against the whole-file extraction. Run BEFORE and AFTER a change with the same --seed/--minutes.

usage: OMP_NUM_THREADS=4 python scripts/live_parity/long_session_cost.py --minutes 12 --label before
"""

import argparse
import json
import sys
import time
import tracemalloc
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import _pin_omp  # noqa: E402,F401

REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))
import numpy as np  # noqa: E402

from synth_session import SR, build_session  # noqa: E402
from mrcd.live.session_features import SessionFeatureTracker  # noqa: E402


def rss_mb():
    try:
        import psutil

        return psutil.Process().memory_info().rss / 1e6
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--label", default="run")
    ap.add_argument("--no-offline", action="store_true")
    ap.add_argument("--out", default=str(REPO / "results/ejecucion_material4_20261030/paridad"))
    a = ap.parse_args()
    audio, info = build_session(a.minutes, a.seed)
    print("session", {k: v for k, v in info.items() if k != "layout"}, flush=True)
    tr = SessionFeatureTracker(f0_center="causal")
    tracemalloc.start()
    tick, per_min, t_all = SR // 2, {}, time.time()
    for i in range(0, len(audio), tick):
        t0 = time.perf_counter()
        tr.append(audio[i : i + tick])
        tr.latest_start()
        tr.f0_center_st()
        dt = time.perf_counter() - t0
        m = int(i / SR // 60)
        per_min.setdefault(m, []).append(dt)
        if (i // tick) % 240 == 239:
            cur, peak = tracemalloc.get_traced_memory()
            print(f"t={i/SR/60:.1f} min tick_mean_last_min={np.mean(per_min[m]):.4f}s traced={cur/1e6:.1f}MB rss={rss_mb()}", flush=True)
    cur, peak = tracemalloc.get_traced_memory()
    res = {
        "label": a.label,
        "minutes": info["duration_s"] / 60,
        "real_clips": info["real_clips"],
        "tick_mean_s_per_minute": {str(k): float(np.mean(v)) for k, v in per_min.items()},
        "tick_max_s": float(max(max(v) for v in per_min.values())),
        "traced_mb_final": cur / 1e6,
        "traced_mb_peak": peak / 1e6,
        "rss_mb": rss_mb(),
        "elapsed_s": time.time() - t_all,
    }
    tracemalloc.stop()
    if not a.no_offline:
        from mrcd.extractors.acoustic import extract_acoustic

        off = extract_acoustic(audio, use_silero=True).frame_matrix()
        F = tr.frame_matrix()
        res["committed_frames"] = len(F)
        res["bad_frames_vs_offline"] = int((np.abs(F - off[: len(F)]).max(1) > 0).sum())
        res["max_abs_f0_diff_st"] = float(np.abs(F[:, 1] - off[: len(F), 1]).max())
    Path(a.out, f"long_session_cost_{a.label}.json").write_text(json.dumps(res, indent=1), "utf8")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
