"""G4: wait after closing a 2 min session (incremental, CPU) with decision profile o2026_11b; RTF of the offline path.

    .venv\\Scripts\\python.exe -m scripts.mejora_servicio.g4_benchmark [--profiles o2026_11b,o2026_11]

Reuses ``scripts/s8_incremental/benchmark.py``: a REAL service process (``mrcd.service serve`` on port 8791, never
8765) with ``--incremental --device cpu --preload p0,p1_disfluent_prompt --warmup``, OMP_NUM_THREADS=4, the 120 s
recording of the S8 benchmark, real-time pace (2 s batches sent at their wall-clock moment), 1 discarded cold round +
2 measured rounds per profile. The profile goes in the query of the offline job and of the session.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scripts.s8_incremental import benchmark as B  # noqa: E402

OUT = REPO / "results" / "mejora_servicio_20261102" / "g4"
_http = B.http
PROFILE = {"name": "o2026_11b"}


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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profiles", default="o2026_11b,o2026_11")
    ap.add_argument("--seconds", type=float, default=120.0)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    B.http = http_with_profile
    log = OUT / "service_g4.log"
    p = B.spawn(
        [
            "--incremental",
            "--device",
            "cpu",
            "--preload",
            "p0,p1_disfluent_prompt",
            "--warmup",
        ],
        log,
    )
    rows = []
    try:
        B.wait_health()
        spid = B.service_pid(p.pid)
        pcm = B.recording(args.seconds)
        for prof in args.profiles.split(","):
            PROFILE["name"] = prof
            for rnd in range(3):
                ref = None
                for mode in ("offline", "incremental"):
                    load = B.system_cpu_percent(1.0)
                    res = (B.run_offline if mode == "offline" else B.run_incremental)(
                        spid, pcm, 1.0
                    )
                    doc = res["doc"]
                    if mode == "offline":
                        ref = B.strip(doc)
                    row = {
                        "profile": prof,
                        "round": rnd,
                        "cold_discarded": rnd == 0,
                        "mode": mode,
                        "seconds": args.seconds,
                        "latency_s": round(res["latency_s"], 2),
                        "rtf_after_close": round(res["latency_s"] / args.seconds, 4),
                        "cpu_s": round(res["cpu_s"], 2),
                        "peak_ws_mb": round(res["peak_ws_mb"], 1),
                        "equal_to_offline": B.strip(doc) == ref,
                        "status": doc["status"],
                        "doc_profile": (
                            (doc.get("provenance") or {}).get("decision_profile") or {}
                        ).get("name"),
                        "background_cpu_pct_before": load,
                    }
                    rows.append(row)
                    print(row, flush=True)
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
            summary[f"{prof}|{mode}"] = {
                "latency_median_s": statistics.median(r["latency_s"] for r in m),
                "latency_range_s": [
                    min(r["latency_s"] for r in m),
                    max(r["latency_s"] for r in m),
                ],
                "all_equal_to_offline": all(r["equal_to_offline"] for r in m),
                "all_completed": all(r["status"] == "completed" for r in m),
                "profile_in_document": sorted({r["doc_profile"] for r in m}),
            }
    first = args.profiles.split(",")[0]
    inc = summary[f"{first}|incremental"]
    off = summary[f"{first}|offline"]
    summary["G4"] = {
        "profile": first,
        "wait_after_close_median_s": inc["latency_median_s"],
        "pass": bool(
            inc["latency_median_s"] <= 10.0
            and inc["all_equal_to_offline"]
            and inc["all_completed"]
        ),
        "rtf_offline_median": round(off["latency_median_s"] / args.seconds, 4),
        "measured_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (OUT / "g4_summary.json").write_text(json.dumps(summary, indent=1), "utf8")
    print(json.dumps(summary["G4"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
