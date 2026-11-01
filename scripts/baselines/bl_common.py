"""Shared helpers for the 2026-10-30 baseline study (new file; touches nothing existing).

Inference path = the one verified with diff 0 on the pilot tensors:
one acoustic extraction per file + windows built per batch with the CURRENT mrcd/ package
(same calls as results/verificacion_20261030/verify_current_vs_pilot.py and the pilot's run_audio_pilot.py).
"""
from __future__ import annotations
import os
import hashlib, json, platform, sys, time
from pathlib import Path
import numpy as np

REPO = Path(r"C:\zArnes-tesis\mrcd")
OUT = REPO / "results" / "baselines_20261030"
CKPT = REPO / "models" / "ckpt_audio_text.pt"
WHISPER = Path.home() / ".cache/huggingface/hub/models--Systran--faster-whisper-small/snapshots/536b0662742c02347bc0e980a01041f333bce120"
CORPORA = Path(os.environ.get("MRCD_CORPORA_ROOT", r"<CORPORA_ROOT>"))
SEED = 13
CLASSES = ["filler_word", "prolongation", "repetition", "revision"]


def sha256_file(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def code_hashes() -> dict:
    files = sorted((REPO / "mrcd").rglob("*.py"))
    d = {str(p.relative_to(REPO)): sha256_file(p) for p in files}
    d["scripts/baselines/bl_common.py"] = sha256_file(Path(__file__))
    return d


def versions() -> dict:
    v = {"python": platform.python_version(), "platform": platform.platform()}
    for m in ["numpy", "scipy", "torch", "soundfile", "faster_whisper", "ctranslate2", "librosa", "silero_vad", "onnxruntime"]:
        try:
            mod = __import__(m)
            v[m] = getattr(mod, "__version__", "?")
        except Exception:
            v[m] = None
    return v


def write_config(outdir: Path, name: str, script: str, extra: dict, inputs: dict | None = None, ckpt: bool = False):
    outdir.mkdir(parents=True, exist_ok=True)
    cfg = {"run": name, "script": script, "script_sha256": sha256_file(script), "seed": SEED,
           "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "versions": versions(), "code_sha256": code_hashes(),
           "input_hashes": inputs or {}, **extra}
    if ckpt:
        cfg["checkpoint"] = {"path": str(CKPT), "sha256": sha256_file(CKPT)}
        cfg["whisper"] = {"snapshot": str(WHISPER), "model_bin_sha256": sha256_file(WHISPER / "model.bin"),
                          "device": "cpu", "compute_type": "int8", "chunk_s": 8.0}
        cfg["inference_path"] = ("single acoustic extraction per file (extract_acoustic use_silero=True) + windows per batch "
                                 "(acoustic_windows/prosody_windows/linguistic_windows, 3 s window, 0.5 s stride) with CURRENT mrcd/; "
                                 "== results/verificacion_20261030/verify_current_vs_pilot.py (diff 0 on pilot tensors). "
                                 "NOT Engine.analyze (parity under study elsewhere).")
    (outdir / "config.json").write_text(json.dumps(cfg, indent=1, ensure_ascii=False), encoding="utf8")
    return cfg


def prf(tp, fp, fn, tn):
    p = tp / (tp + fp) if tp + fp else float("nan")
    r = tp / (tp + fn) if tp + fn else float("nan")
    f = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else float("nan")
    return {"n": tp + fp + fn + tn, "support_pos": tp + fn, "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
            "precision": p, "recall": r, "f1": f}


def confusion(y, p):
    y = np.asarray(y).astype(int); p = np.asarray(p).astype(int)
    return prf(int(((y == 1) & (p == 1)).sum()), int(((y == 0) & (p == 1)).sum()),
               int(((y == 1) & (p == 0)).sum()), int(((y == 0) & (p == 0)).sum()))


def cluster_bootstrap(y, p, groups, n_boot=2000, seed=SEED):
    """95% percentile CI for P/R/F1 resampling GROUPS (speakers/conversations) with replacement."""
    y = np.asarray(y); p = np.asarray(p); g = np.asarray(groups)
    ug = np.unique(g)
    idx = {k: np.where(g == k)[0] for k in ug}
    rng = np.random.default_rng(seed)
    res = {"precision": [], "recall": [], "f1": []}
    for _ in range(n_boot):
        pick = rng.choice(len(ug), len(ug), replace=True)
        ii = np.concatenate([idx[ug[k]] for k in pick])
        m = confusion(y[ii], p[ii])
        for k in res:
            res[k].append(m[k])
    out = {}
    for k, v in res.items():
        v = np.array(v, float)
        out[k] = [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))] if np.isfinite(v).any() else [float("nan")] * 2
    out["n_groups"] = int(len(ug)); out["n_boot"] = n_boot
    return out


def per_group(y, p, groups):
    y = np.asarray(y); p = np.asarray(p); g = np.asarray(groups)
    return {str(k): confusion(y[g == k], p[g == k]) for k in np.unique(g)}
