"""S1 task 3: does the EXISTING streaming ASR route keep up with a live tick? One config per process, serial.

Route = ``StreamingASR.merge_recent`` on the last 13 s ring every ``tick`` s (warm-up 6 s, same as scripts/cierre3b/asr_compare.py).
Per tick: decode latency, RTF_tick = latency / tick, and a simulated backlog (a tick can start only when the previous one has
finished; lag = how far the transcript is behind the audio). Real time is sustained when RTF_tick <= 1 (p95 <= 1 for a margin).
usage: OMP_NUM_THREADS=4 python tick_bench.py --config p0_cpu|p0_cuda --tick 0.5 --out DIR [--limit N]
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _pin_omp  # noqa: F401,E402

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cierre3b"))
import _sys  # noqa: E402

from mrcd.constants import CONTEXT_S, SAMPLE_RATE, WINDOW_S  # noqa: E402
from mrcd.extractors.linguistic import VerbatimASR  # noqa: E402
from mrcd.live.streaming_asr import StreamingASR  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
INPUTS = REPO / "results/cierre_oe1_oe3_20261031/3b/cost/inputs"
CONFIGS = {"p0_cpu": ("p0", "cpu", "int8"), "p0_cuda": ("p0_cuda", "cuda", "float16")}
WARMUP_S = 6.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, choices=sorted(CONFIGS))
    ap.add_argument("--tick", type=float, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0, help="use only the N shortest eligible (>=13 s) units; 0 = all")
    a = ap.parse_args()
    prof, dev, ct = CONFIGS[a.config]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    bg = _sys.system_cpu_load(3)
    asr = VerbatimASR("small", device=dev, compute_type=ct, profile=prof)
    # warm the model on a throw-away pass (first CUDA call pays kernel JIT; reported in gpu bench as cold start)
    sasr = StreamingASR(asr=asr)
    files = [p for p in sorted(INPUTS.glob("*.wav"), key=lambda p: sf.info(p).duration) if sf.info(p).duration >= 13.0]
    if a.limit:
        files = files[: a.limit]
    x0, _ = sf.read(files[0], dtype="float32")
    sasr.merge_recent([], x0[: 13 * SAMPLE_RATE], 0.0)
    keep = int(round((CONTEXT_S + WINDOW_S) * SAMPLE_RATE))
    units, lat_all = [], []
    for p in files:
        audio, sr = sf.read(p, dtype="float32")
        dur = len(audio) / sr
        ticks = list(np.arange(WARMUP_S, dur, a.tick)) + [dur]
        words, lat, lag, free_at = [], [], [], 0.0
        sasr = StreamingASR(asr=asr)
        for T in ticks:
            n = int(round(T * SAMPLE_RATE))
            seg = audio[max(n - keep, 0): n]
            begin = (n - len(seg)) / SAMPLE_RATE
            t1 = time.perf_counter()
            words = sasr.merge_recent(words, seg, begin)
            dt = time.perf_counter() - t1
            start = max(T, free_at)  # the tick starts when the audio is there AND the previous tick finished
            free_at = start + dt
            lat.append(dt)
            lag.append(free_at - T)
        lat_all += lat
        units.append({"unit": p.stem, "dur_s": dur, "n_ticks": len(ticks), "latency_median_s": float(np.median(lat)), "latency_p95_s": float(np.percentile(lat, 95)),
                      "latency_max_s": float(max(lat)), "rtf_tick_median": float(np.median(lat)) / a.tick, "rtf_tick_pooled": sum(lat) / (len(lat) * a.tick),
                      "frac_ticks_over_budget": float(np.mean(np.array(lat) > a.tick)), "final_lag_s": lag[-1], "max_lag_s": float(max(lag))})
    lat_all = np.array(lat_all)
    res = {"config": a.config, "profile": prof, "device": asr.device, "compute_type": asr.compute_type, "tick_s": a.tick, "date": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "background_cpu_pct_start": bg, "background_cpu_pct_end": _sys.system_cpu_load(2), "route": "StreamingASR.merge_recent, 13 s ring",
           "n_units": len(units), "n_ticks": int(lat_all.size), "latency_median_s": float(np.median(lat_all)), "latency_p95_s": float(np.percentile(lat_all, 95)),
           "latency_max_s": float(lat_all.max()), "rtf_tick_pooled": float(lat_all.mean() / a.tick), "frac_ticks_over_budget": float(np.mean(lat_all > a.tick)),
           "max_lag_s": max(u["max_lag_s"] for u in units), "real_time_sustained_all_units": all(u["max_lag_s"] <= a.tick for u in units), "units": units}
    (out / f"tick_{a.config}_t{a.tick}.json").write_text(json.dumps(res, indent=1), "utf8")
    print(a.config, a.tick, "rtf_tick", round(res["rtf_tick_pooled"], 3), "p95", round(res["latency_p95_s"], 3), "max_lag", round(res["max_lag_s"], 2))


if __name__ == "__main__":
    main()
