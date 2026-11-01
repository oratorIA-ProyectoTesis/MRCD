"""Assemble results/paridad_20261030/PARIDAD.json from the parity outputs + environment/hash provenance."""
import hashlib, json, sys, platform
from importlib import metadata
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import compare as C
OUT = C.OUT; S = Path(C.P) / "engine_snapshot"; R = Path(r"C:\zArnes-tesis\mrcd")
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
par = json.loads((OUT / "parity_snapshot_vs_current.json").read_text("utf8"))
tabs = {(t, m): C.table(t, m) for t in ("snapshot", "current") for m in ("batch", "bw", "engine")}
labels = tabs[("snapshot", "batch")][1]
engkeys = set(tabs[("snapshot", "engine")][0])
grid_metrics = {f"{t}/{m}_grid_only": C.metrics({k: v for k, v in tabs[(t, m)][0].items() if k in engkeys}, labels) for t in ("snapshot", "current") for m in ("batch", "bw")}
pilot_pred = [json.loads(l) for l in (C.P / "results/audio_predictions.jsonl").read_text("utf8").splitlines()]
import numpy as np
ptab = {}
for r in pilot_pred:
    for s, p in zip(r["window_starts_local_s"], r["probabilities"]): ptab[(r["sample_id"], round(s, 3))] = {"probs": np.array(p)}
bt = tabs[("current", "engine")][0]
vs_pilot = C.compare(ptab, tabs[("current", "batch")][0], labels); vs_pilot_snap = C.compare(ptab, tabs[("snapshot", "bw")][0], labels)
pk = lambda d: {"n_windows": d["n_windows"], "argmax_changes": d["argmax_changes"], "prob_max_abs": d["tensors"]["probs"]["max_abs"]}
vers = {m: metadata.version(m) for m in ["torch", "numpy", "scipy", "librosa", "soundfile", "faster-whisper", "ctranslate2", "onnxruntime", "silero-vad"]}
vers["python"] = platform.python_version(); vers["platform"] = platform.platform()
code = {}
for name, root in (("snapshot", S), ("current", R)):
    code[name] = {str(p.relative_to(root)).replace("\\", "/"): sha(p) for p in sorted((root / "mrcd").rglob("*.py"))}
diff_files = [k for k in code["current"] if code["current"][k] != code["snapshot"].get(k)]
out = {"environment": vers,
       "params": {"torch_threads": 4, "OMP_NUM_THREADS": 4, "torch_seed": 0, "device": "cpu", "window_s": 3.0, "stride_s": 0.5, "context": "10 s (5 s before)",
                  "whisper": "faster-whisper-small 536b0662 CPU int8 transcribe_chunked chunk_s=8 overlap 2, beam 5, T=0", "pilot_starts": "a..b-3 step 0.5 + terminal b-3",
                  "engine_starts": "0..dur-3 step 0.5 (grid only); decision uses engine windows with start>=a and start+3<=b",
                  "decision_rule": "any window argmax == prolongation among target-contained windows", "pause_threshold": "n/a (not used by these paths)"},
       "inputs": {j["id"]: j["sha256"] for j in C.jobs.values()}, "extra_source": [j for j in C.jobs.values() if j["kind"] == "extra"][0]["source_sha256"],
       "checkpoint_sha256": {"snapshot": sha(S / "ckpt_audio_text.pt"), "current": sha(R / "models/ckpt_audio_text.pt")},
       "code_sha256": code, "code_files_differing_snapshot_vs_current": diff_files,
       "pairs": {k: {kk: vv for kk, vv in v.items() if kk != "argmax_change_windows"} for k, v in par.items() if k != "metrics"},
       "metrics_all_paths": par["metrics"], "metrics_grid_only": grid_metrics,
       "current_batch_vs_saved_pilot_probs": pk(vs_pilot), "snapshot_bw_vs_saved_pilot_probs": pk(vs_pilot_snap),
       "asr": json.loads((OUT / "asr_variability.json").read_text("utf8"))["summary"],
       "attribution_slice": json.loads((OUT / "attribution_slice_summary.json").read_text("utf8"))}
(OUT / "PARIDAD.json").write_text(json.dumps(out, indent=1), "utf8")
print(json.dumps({k: out[k] for k in ("metrics_grid_only", "current_batch_vs_saved_pilot_probs", "snapshot_bw_vs_saved_pilot_probs", "code_files_differing_snapshot_vs_current", "environment", "checkpoint_sha256")}, indent=1))
