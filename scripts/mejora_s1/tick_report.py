"""S1 task 3: table from tick_*.json; GPU is also restricted to the units the CPU baseline used (paired view)."""
import json
import sys
from pathlib import Path

import numpy as np

D = Path(sys.argv[1])
R = {f.stem: json.loads(f.read_text("utf8")) for f in sorted(D.glob("tick_*.json"))}
cpu_units = {t: {u["unit"] for u in r["units"]} for k, r in R.items() if "_cpu_" in k for t in [r["tick_s"]]}
L = ["| config | tick s | units | ticks | latency median s | p95 s | max s | RTF_tick pooled | ticks over budget | max lag s | all units keep up (lag <= tick) | bg CPU % start/end |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
for k, r in R.items():
    L.append(f"| {r['config']} ({r['device']}/{r['compute_type']}) | {r['tick_s']} | {r['n_units']} | {r['n_ticks']} | {r['latency_median_s']:.3f} | {r['latency_p95_s']:.3f} | {r['latency_max_s']:.3f} | "
             f"{r['rtf_tick_pooled']:.3f} | {r['frac_ticks_over_budget']:.1%} | {r['max_lag_s']:.2f} | {r['real_time_sustained_all_units']} | {r['background_cpu_pct_start']:.1f}/{r['background_cpu_pct_end']:.1f} |")
L += ["", "Paired view: GPU restricted to the units used by the CPU baseline at the same tick (per-unit figures; pooled RTF = unit-tick-weighted mean).", "",
      "| config | tick s | units | RTF_tick pooled | worst unit p95 s | max lag s |", "|---|---|---|---|---|---|"]
for k, r in R.items():
    ids = cpu_units.get(r["tick_s"])
    if not ids:
        continue
    us = [u for u in r["units"] if u["unit"] in ids]
    n = sum(u["n_ticks"] for u in us)
    pooled = sum(u["rtf_tick_pooled"] * u["n_ticks"] for u in us) / n
    L.append(f"| {r['config']} | {r['tick_s']} | {len(us)} | {pooled:.3f} | {max(u['latency_p95_s'] for u in us):.3f} | {max(u['max_lag_s'] for u in us):.2f} |")
(D / "TICK_tables.md").write_text("\n".join(L) + "\n", "utf8")
print("\n".join(L))
