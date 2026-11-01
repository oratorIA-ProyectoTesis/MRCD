"""live_cfg v1/v2/v3 comparison on NEW seeds (rule: mrcd/docs/LIVE_CFG_V3_DECISION_RULE.md), strictly serial.

usage: python live_cfg_compare.py --out DIR [--seeds 501 502 ...] [--minutes 10.5] [--silence-s 120]
Per (seed, config): LiveScorer driven with 0.5 s ticks (8000 samples), stub score, empty words.
Metrics: per-tick wall time, process CPU time, peak working set, confirmation latency of provisional rows
(stream time at emission - window end), windows without a row (not in gaps), duplicate rows,
frame parity vs whole-session offline extraction (committed before finalize and after finalize).
"""
import sys as _s, pathlib as _p; _s.path.insert(0, str(_p.Path(__file__).resolve().parents[1])); import _pin_omp  # noqa: F401
import argparse, json, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path[:0] = [str(REPO), str(REPO / "scripts"), str(REPO / "scripts" / "live_parity"), str(HERE)]
import numpy as np  # noqa: E402
import _sys  # noqa: E402
from synth_session import build_session  # noqa: E402
from mrcd.constants import SAMPLE_RATE, STRIDE_S, WINDOW_S  # noqa: E402
from mrcd.extractors.acoustic import extract_acoustic  # noqa: E402
from mrcd.live.session import CONFIGS, LiveScorer  # noqa: E402

TICK = 8000
LABELS = ["a", "b"]
CFGS = ["live_cfg_v1", "live_cfg_v2", "live_cfg_v3"]


def stub(_t):
    return np.array([0.6, 0.4])


def drive(audio, cfg, offline=None):
    sc = LiveScorer(stub, LABELS, config=cfg, f0_center="causal", ling_dim=12)
    tick_s, lat = [], []
    cpu0, wall0 = time.process_time(), time.perf_counter()
    for i in range(0, len(audio), TICK):
        t0 = time.perf_counter()
        sc.push(audio[i:i + TICK])
        rows = sc.step([])
        tick_s.append(time.perf_counter() - t0)
        for r in rows or []:
            lat.append(sc.duration_s - (r["timestamp"] + WINDOW_S))
    committed = sc.tracker.frame_matrix().copy()
    t0 = time.perf_counter()
    fin = sc.finalize([])
    fin_s = time.perf_counter() - t0
    out = {"config": cfg, "duration_s": len(audio) / SAMPLE_RATE, "n_ticks": len(tick_s),
           "tick_s": {k: float(v) for k, v in zip(("median", "p95", "max", "mean"), (np.median(tick_s), np.percentile(tick_s, 95), np.max(tick_s), np.mean(tick_s)))},
           "tick_over_budget_pct": float(100 * np.mean(np.array(tick_s) > 0.5)),
           "total_wall_s": time.perf_counter() - wall0, "total_cpu_s": time.process_time() - cpu0,
           "finalize_s": fin_s, "finalize_rows": len(fin),
           "latency_s": {"median": float(np.median(lat)), "p95": float(np.percentile(lat, 95)), "max": float(np.max(lat)), "n": len(lat)} if lat else None,
           "mem": _sys.mem_mb(), "gaps": sc.gaps, "n_rows": len(sc.rows)}
    # windows without any row / duplicates
    end = sc.duration_s
    grid = int(np.floor((end - WINDOW_S) / STRIDE_S + 1e-9)) + 1
    have = {int(round(r["timestamp"] / STRIDE_S)) for r in sc.rows}
    skipped = {int(round(t / STRIDE_S)) for g in sc.gaps if g["kind"] == "windows_skipped"
               for t in np.arange(g["from_s"], g["to_s"], STRIDE_S)}
    lost = [k for k in range(grid) if k not in have and k not in skipped]
    keys = [(r["timestamp"], r.get("status"), r.get("revises")) for r in sc.rows]
    out.update(grid_windows=grid, windows_without_row=len(lost), windows_without_row_first=lost[:5],
               windows_recorded_skipped=len(skipped), duplicate_rows=len(keys) - len(set(keys)))
    if offline is not None:
        F, G = committed, sc.tracker.frame_matrix()
        bad_c = np.abs(F - offline[:len(F)]).max(1) > 0
        bad_f = np.abs(G - offline[:len(G)]).max(1) > 0
        out.update(frames_total=int(len(G)), frames_offline=int(len(offline)),
                   committed_bad=int(bad_c.sum()), final_bad=int(bad_f.sum()),
                   max_abs_f0_st=float(np.abs(F[:, 1] - offline[:len(F), 1]).max()) if len(F) else None,
                   first_bad_s=(np.nonzero(bad_f)[0] * 0.01).round(2).tolist()[:5])
    return out


def silence_audio(kind, secs, seed=7):
    n = int(secs * SAMPLE_RATE)
    if kind == "digital_zeros":
        return np.zeros(n, np.float32)
    return (np.random.default_rng(seed).normal(0, 10 ** (-70 / 20), n)).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[501, 502, 503, 504, 505, 506])
    ap.add_argument("--minutes", type=float, default=10.5)
    ap.add_argument("--silence-s", type=float, default=120)
    ap.add_argument("--max-wait-min", type=float, default=0, help="wait this long for background CPU < 15 %% before each run")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    jl = out / "runs.jsonl"
    done = {(r["kind"], r.get("seed"), r["config"]) for r in map(json.loads, jl.read_text("utf8").splitlines())} if jl.exists() else set()
    cfgsha = {c: __import__("hashlib").sha256(repr(CONFIGS[c]).encode()).hexdigest()[:16] for c in CFGS}
    (out / "config.json").write_text(json.dumps({
        "seeds": a.seeds, "minutes": a.minutes, "partition": "tick 8000 samples (0.5 s)", "configs": {c: repr(CONFIGS[c]) for c in CFGS},
        "hardware": _sys.hardware(), "code_sha256_mrcd": _sys.code_sha(REPO),
        "rule_sha256": _sys.sha256_file(REPO / "docs/LIVE_CFG_V3_DECISION_RULE.md"), "omp": __import__("os").environ.get("OMP_NUM_THREADS")}, indent=1), "utf8")

    def run(kind, seed, audio, offline):
        for cfg in CFGS:
            if (kind, seed, cfg) in done:
                continue
            waited = 0.0
            bg = _sys.background_report(3)
            while bg["system_cpu_pct"] >= 15 and waited < a.max_wait_min * 60:
                time.sleep(30); waited += 33; bg = _sys.background_report(3)
            res = drive(audio, cfg, offline)
            res.update(kind=kind, seed=seed, background_before=bg, load_valid=bg["system_cpu_pct"] < 15, waited_s=waited)
            res["mem_after"] = _sys.mem_mb()
            with jl.open("a", encoding="utf8") as f:
                f.write(json.dumps(res) + "\n")
            print(kind, seed, cfg, "tick_med", round(res["tick_s"]["median"], 3), "bad", res.get("final_bad"), "load", bg["system_cpu_pct"], flush=True)

    for kind in ("digital_zeros", "noise_-70dBFS"):
        run(kind, None, silence_audio(kind, a.silence_s), None)
    for seed in a.seeds:
        audio = build_session(a.minutes, seed)[0].astype(np.float32)
        off = extract_acoustic(audio, use_silero=True).frame_matrix()
        run("session", seed, audio, off)
    print("DONE")


if __name__ == "__main__":
    main()
