"""Plan F latency gate G4 for decision profile o2026_13 (protocol section 5): 1 cold round discarded + 5 measured rounds.

    python scripts/plan_f/g4_o2026_13.py [--profiles o2026_13,o2026_11] [--measured 5]

Same procedure and metric as ``scripts/plan_a/g4_o2026_12.py`` (it reuses ``scripts/s8_incremental/benchmark.py``): a REAL service
process (``mrcd.service serve`` on test port 8791) with ``--incremental --device cpu --preload p0,ft_small_v1 --warmup``,
OMP_NUM_THREADS=4, HF_HUB_OFFLINE=1, the 120 s recording of the S8 benchmark (the same seed clip as Plan A's G4), real-time pace
(2 s batches). Metric: wait after close of the incremental session; limit 10 s. G4 PASSES only if the median AND the maximum of the
measured rounds are <= 10 s (and every document completed and equals the offline document). Background CPU load is recorded
before every round (flagged if >= 20 %, never excluded). Output: results/plan_f_20261108/g4/ (summary + runs are force-added).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

WT = Path(__file__).resolve().parents[2]
#: Main checkout: MRCD_REPO_ROOT, else the layout <root>/_wt/<worktree> next to <root>/mrcd.
MAIN = Path(os.environ.get("MRCD_REPO_ROOT") or WT.parents[1] / "mrcd")
sys.path.insert(0, str(WT))

from scripts.s8_incremental import benchmark as B  # noqa: E402

OUT = WT / "results" / "plan_f_20261108" / "g4"
B.INPUTS = {k: MAIN / v.relative_to(B.REPO) for k, v in B.INPUTS.items()}
_http = B.http
PROFILE = {"name": "o2026_13"}
LIMIT_S = 10.0
BG_FLAG_PCT = 20.0


def http_with_profile(method, path, body=None, headers=None):
    if method == "POST" and (
        path.startswith("/v1/jobs?") or path.startswith("/v1/sessions?")
    ):
        path = f"{path}&decision_profile={PROFILE['name']}"
    return (
        _http(method, path, body, headers)
        if headers is not None or body is not None
        else _http(method, path)
    )


def spawn(flags: list[str], log: Path) -> subprocess.Popen:
    env = {
        **os.environ,
        "OMP_NUM_THREADS": "4",
        "HF_HUB_OFFLINE": "1",
        "PYTHONUNBUFFERED": "1",
        "PYTHONPATH": str(WT),
    }
    return subprocess.Popen(
        [
            str(MAIN / ".venv/Scripts/python.exe"),
            "-m",
            "mrcd.service",
            "serve",
            "--port",
            str(B.PORT),
            *flags,
        ],
        cwd=WT,
        env=env,
        stdout=open(log, "ab"),
        stderr=subprocess.STDOUT,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profiles", default="o2026_13,o2026_11")
    ap.add_argument("--seconds", type=float, default=120.0)
    ap.add_argument("--measured", type=int, default=5)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    B.http = http_with_profile
    p = spawn(
        [
            "--incremental",
            "--device",
            "cpu",
            "--preload",
            "p0,ft_small_v1",
            "--warmup",
        ],
        OUT / "service_g4.log",
    )
    rows = []
    try:
        B.wait_health()
        spid = B.service_pid(p.pid)
        pcm = B.recording(args.seconds)
        for prof in args.profiles.split(","):
            PROFILE["name"] = prof
            for rnd in range(1 + args.measured):
                ref = None
                for mode in ("offline", "incremental"):
                    load = B.system_cpu_percent(1.0)
                    res = (B.run_offline if mode == "offline" else B.run_incremental)(
                        spid, pcm, 1.0
                    )
                    doc = res["doc"]
                    if mode == "offline":
                        ref = B.strip(doc)
                    dp = (doc.get("provenance") or {}).get("decision_profile") or {}
                    rows.append(
                        {
                            "profile": prof,
                            "round": rnd,
                            "cold_discarded": rnd == 0,
                            "mode": mode,
                            "seconds": args.seconds,
                            "latency_s": round(res["latency_s"], 2),
                            "rtf_after_close": round(
                                res["latency_s"] / args.seconds, 4
                            ),
                            "cpu_s": round(res["cpu_s"], 2),
                            "peak_ws_mb": round(res["peak_ws_mb"], 1),
                            "equal_to_offline": B.strip(doc) == ref,
                            "status": doc["status"],
                            "doc_profile": dp.get("name"),
                            "asr_profile": (doc.get("request") or {}).get(
                                "asr_profile"
                            ),
                            "background_cpu_pct_before": load,
                            "background_flagged": bool(load >= BG_FLAG_PCT),
                        }
                    )
                    print(rows[-1], flush=True)
                    (OUT / "g4_runs.json").write_text(
                        json.dumps(rows, indent=1), "utf8"
                    )
    finally:
        B.stop(p)
    summary = {}
    for prof in args.profiles.split(","):
        for mode in ("offline", "incremental"):
            m = [
                r
                for r in rows
                if r["profile"] == prof
                and r["mode"] == mode
                and not r["cold_discarded"]
            ]
            cold = [
                r["latency_s"]
                for r in rows
                if r["profile"] == prof and r["mode"] == mode and r["cold_discarded"]
            ]
            summary[f"{prof}|{mode}"] = {
                "cold_discarded_s": cold,
                "latency_measured_s": [r["latency_s"] for r in m],
                "latency_median_s": statistics.median(r["latency_s"] for r in m),
                "latency_max_s": max(r["latency_s"] for r in m),
                "all_equal_to_offline": all(r["equal_to_offline"] for r in m),
                "all_completed": all(r["status"] == "completed" for r in m),
                "profile_in_document": sorted({str(r["doc_profile"]) for r in m}),
                "asr_profile_in_document": sorted({str(r["asr_profile"]) for r in m}),
                "background_cpu_pct_before": [
                    r["background_cpu_pct_before"] for r in m
                ],
                "background_flagged_rounds": [
                    r["round"] for r in m if r["background_flagged"]
                ],
            }
    first = args.profiles.split(",")[0]
    inc, off = summary[f"{first}|incremental"], summary[f"{first}|offline"]
    summary["G4"] = {
        "profile": first,
        "measured_rounds": args.measured,
        "wait_after_close_median_s": inc["latency_median_s"],
        "wait_after_close_max_s": inc["latency_max_s"],
        "limit_s": LIMIT_S,
        "pass": bool(
            inc["latency_median_s"] <= LIMIT_S
            and inc["latency_max_s"] <= LIMIT_S
            and inc["all_equal_to_offline"]
            and inc["all_completed"]
        ),
        "rule": "median AND max of the measured rounds <= 10 s, every document completed and equal to the offline document",
        "rtf_offline_median": round(off["latency_median_s"] / args.seconds, 4),
        "measured_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "procedure": "scripts/plan_a/g4_o2026_12.py procedure with 1 cold discarded + 5 measured rounds, 120 s, real time",
    }
    summary["protocol_sha256"] = hashlib.sha256(
        (WT / "results/plan_f_20261108/PROTOCOL_EXPLORATORY_F.md")
        .read_bytes()
        .replace(b"\r\n", b"\n")
    ).hexdigest()
    (OUT / "g4_summary.json").write_text(json.dumps(summary, indent=1), "utf8")
    print(json.dumps(summary["G4"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
