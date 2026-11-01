"""S1 task 2: tables from bench_*.json + 'desacuerdo' (text disagreement) CPU int8 vs GPU on the same units.

Disagreement = word-level edit distance between the CPU and the GPU transcript / words of the CPU transcript. It is NOT a
WER against a human reference and says nothing about which text is better. Only numbers are written (no transcripts).
usage: python gpu_report.py DIR   (DIR holds bench_*.json and _texts/)
"""
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "cierre3b"))
from asr_analyze import norm_text  # noqa: E402

from mrcd.asr_profiles import PROFILES, detect_asr_anomalies  # noqa: E402
from mrcd.asr_recovery.stats import edit_distance  # noqa: E402

D = Path(sys.argv[1])
CFG = ["p0_cpu", "p0_cuda", "p1_cpu", "p1_cuda"]
B = {c: json.loads((D / f"bench_{c}.json").read_text("utf8")) for c in CFG if (D / f"bench_{c}.json").exists()}


def med(xs):
    return float(np.median(xs)) if xs else float("nan")


summary = {"configs": {}, "disagreement": {}}
for c, b in B.items():
    rows_all = [r for r in b["rows"] if r["round"] >= 1]
    rows = [r for r in rows_all if r["bg_cpu_pct_before_round"] <= 15.0]  # load gate: rounds above 15 % are discarded
    dropped_rounds = sorted({r["round"] for r in rows_all} - {r["round"] for r in rows})
    r0 = [r for r in b["rows"] if r["round"] == 0]
    dur = sum(r["dur_s"] for r in r0)
    per_round = {}
    for r in rows:
        per_round.setdefault(r["round"], []).append(r)
    pooled = [sum(x["wall_s"] for x in v) / sum(x["dur_s"] for x in v) for v in per_round.values()]
    summary["configs"][c] = {
        "actual": b["actual"], "n_files": len(r0), "audio_s_total": dur,
        "load_s": b["load_s"], "first_pass_rtf_pooled": sum(r["wall_s"] for r in r0) / dur,
        "first_file_wall_s": r0[0]["wall_s"], "first_file_dur_s": r0[0]["dur_s"],
        "warm_rtf_median_per_file": med([r["rtf"] for r in rows]),
        "warm_rtf_pooled_per_round": pooled,
        "warm_rtf_range_per_file": [min(r["rtf"] for r in rows), max(r["rtf"] for r in rows)],
        "process_rss_peak_mb": b["mem_end"]["peak_rss_mb"], "rss_after_load_mb": b["mem_after_load"]["rss_mb"],
        "gpu_used_mb_delta_after_load": (b["gpu_used_mb_after_load"] - b["gpu_used_mb_before_load"]) if b.get("gpu_used_mb_after_load") else None,
        "gpu_peak_delta_mb": (b["gpu_peak_used_mb"] - b["gpu_used_mb_before_load"]) if b.get("gpu_peak_used_mb") else None,
        "background_cpu_pct_start": b["background_start"]["system_cpu_pct"], "background_cpu_pct_end": b["background_end"],
        "bg_before_rounds": sorted({r["bg_cpu_pct_before_round"] for r in b["rows"]}),
        "gate_ok_start": b["gate_ok"], "rounds_discarded_for_load": dropped_rounds, "rounds_kept": sorted(per_round), "total_process_s": b["total_process_s"],
        "profile_experimental": PROFILES[b["profile"]].experimental if b["profile"] in PROFILES else None,
    }


def toks(cfg, unit):
    p = D / "_texts" / f"{cfg}__{unit}.txt"
    return norm_text(p.read_text("utf8")) if p.exists() else None


units = sorted({r["file"] for b in B.values() for r in b["rows"]})
for cpu, gpu in (("p0_cpu", "p0_cuda"), ("p1_cpu", "p1_cuda")):
    if cpu not in B or gpu not in B:
        continue
    per, E, N, ident = [], 0, 0, 0
    for u in units:
        a, g = toks(cpu, u), toks(gpu, u)
        if a is None or g is None:
            continue
        e = edit_distance(a, g)
        E, N, ident = E + e, N + len(a), ident + (a == g)
        per.append({"unit": u, "n_cpu_words": len(a), "n_gpu_words": len(g), "edits": e, "desacuerdo": e / max(1, len(a))})
    loops = {k: sum(bool([x for x in detect_asr_anomalies(" ".join(t), PROFILES[B[k]["profile"]].initial_prompt) if x["kind"] == "repeated_ngram_loop"])
                    for u in units if (t := toks(k, u)) is not None) for k in (cpu, gpu)}
    summary["disagreement"][f"{cpu}_vs_{gpu}"] = {
        "label": "desacuerdo (edit distance CPU int8 vs GPU, normalised by CPU words); NOT WER vs a human reference",
        "units": len(per), "identical_units": ident, "pooled_desacuerdo": E / max(1, N), "per_unit": per,
        "median_per_unit": med([p["desacuerdo"] for p in per]), "max_per_unit": max((p["desacuerdo"] for p in per), default=None),
        "units_with_loop_flag": loops,
    }
(D / "gpu_summary.json").write_text(json.dumps(summary, indent=1), "utf8")

L = ["| config | device/compute | load s | first pass RTF (cold) | warm RTF median/file [min-max] | warm pooled RTF per round | RSS peak MB | GPU MB (load / peak, delta) | bg CPU % (start/end) | rounds discarded (load >15 %) |", "|---|---|---|---|---|---|---|---|---|---|"]
for c, s in summary["configs"].items():
    a = s["actual"]
    L.append(f"| {c} | {a['device']}/{a['compute_type']} | {s['load_s']:.2f} | {s['first_pass_rtf_pooled']:.3f} | {s['warm_rtf_median_per_file']:.3f} [{s['warm_rtf_range_per_file'][0]:.3f}-{s['warm_rtf_range_per_file'][1]:.3f}] | "
             f"{', '.join(f'{x:.3f}' for x in s['warm_rtf_pooled_per_round'])} | {s['process_rss_peak_mb']:.0f} | "
             f"{('%.0f / %.0f' % (s['gpu_used_mb_delta_after_load'], s['gpu_peak_delta_mb'])) if s['actual']['device'] == 'cuda' and s['gpu_used_mb_delta_after_load'] is not None else 'n/a (CPU)'} | {s['background_cpu_pct_start']:.1f}/{s['background_cpu_pct_end']:.1f} | {s['rounds_discarded_for_load'] or '-'} |")
L += ["", "| pair | units | identical | pooled desacuerdo | median/unit | max/unit | loop-flag units (cpu, gpu) |", "|---|---|---|---|---|---|---|"]
for k, d in summary["disagreement"].items():
    L.append(f"| {k} | {d['units']} | {d['identical_units']} | {d['pooled_desacuerdo']:.4f} | {d['median_per_unit']:.4f} | {d['max_per_unit']:.4f} | {tuple(d['units_with_loop_flag'].values())} |")
(D / "GPU_tables.md").write_text("\n".join(L) + "\n", "utf8")
print("\n".join(L))
