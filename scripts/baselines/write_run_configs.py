"""Write config.json (hashes of inputs/checkpoint/code, versions, seed) for the MRCD inference runs and the report."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from bl_common import OUT, write_config, sha256_file
S = str(Path(__file__).with_name("run_mrcd.py"))
notes = {"dihana": "7 shards x 2 torch threads; order: fold 5 (test), fold 4 (tune), then dev folds 1-3 in seeded pseudo-random order; run STOPPED after 1253 dev turns (of 3816) by time budget; "
                   "an earlier 4-shard launch (first ~160 dev turns, dihana_shard0-3 files) used the same code with 4 threads. Units <3 s zero-padded at the end to 3 s.",
         "ameresco": "1 shard x 4 threads, all 98 pool segments; +-5 s context as in the pilot; target-contained 3 s windows, 0.5 s stride.",
         "epic": "2 shards x 2 threads, all 136 Codex-set sentences (clips with real context, automatic sentence times)."}
for k, n in notes.items():
    d = OUT / k / "mrcd_run"
    ins = {"manifest.json": sha256_file(OUT / k / "manifest.json")}
    ins.update({p.name: sha256_file(p) for p in sorted(d.glob("*.jsonl"))})
    write_config(d, f"mrcd_inference_{k}", S, {"notes": n, "window_s": 3.0, "stride_s": 0.5, "pred_rule": "any window argmax == class",
                 "units_completed": sum(1 for p in d.glob("*.jsonl") for l in p.read_text("utf8").splitlines() if l.strip())}, ins, ckpt=True)
write_config(OUT, "baselines_report", __file__, {"files": "BASELINES.md, metrics_audio.json, text/text_metrics.json, error_analysis.json"},
             {"metrics_audio.json": sha256_file(OUT / "metrics_audio.json"), "text_metrics.json": sha256_file(OUT / "text" / "text_metrics.json")})
print("ok")
