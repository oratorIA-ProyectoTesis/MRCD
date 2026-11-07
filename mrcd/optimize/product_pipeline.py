"""Versioned, hash-verified packaging of the optimised per-class pipelines (opt-in; the service integrates it later).

A pipeline maps a unit record (same fields as ``optimize.data``) to ``score``, ``decision`` and a three-state
``state`` (present / absent / uncertain). Scores are model outputs, NOT calibrated probabilities unless the
manifest says ``calibration_claim`` otherwise (it never does)."""

from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path

import numpy as np

from mrcd.optimize import space as SP
from mrcd.optimize.class_pipeline import ClassPipeline  # noqa: F401  (re-exported: existing pickles resolve it here)

PIPELINE_VERSION = "o-campaign/1"
DEFAULT_PIPELINE = False  # opt-in until the promotion rule of PROTOCOL.md section 7 is met and the service adopts it


def _sha_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def fit_class_pipeline(d, tr: np.ndarray, sel_best: dict, note: str) -> ClassPipeline:
    """``sel_best``: ``search()['best']`` computed on rows ``tr`` (global indices) of ``d`` (search.Data)."""
    from mrcd.optimize import search as SE

    cfg = sel_best["config"]
    m, _ = SE.fit_config(d, tr, cfg)
    st = d.store.stats(cfg.profile, tr)
    method, cal, band, _, _ = SE.choose_calibration(
        sel_best["inner_oof"], d.y[tr], d.w[tr], d.g[tr], sel_best["thr"]
    )
    return ClassPipeline(
        cfg,
        {"med": st.med, "scale": st.scale},
        m,
        float(sel_best["thr"]),
        sel_best["t_or"],
        cal.to_dict() if cal else None,
        band,
        note,
    )


def save_pipelines(pipes: dict, out_dir: Path, meta: dict) -> dict:
    """Writes ``<out_dir>/o_pipeline_<cls>.pkl`` + ``MANIFEST.json`` (sha256 of every file, config, versions)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = {}
    for c, p in pipes.items():
        blob = pickle.dumps(p, protocol=4)
        fp = out_dir / f"o_pipeline_{c}.pkl"
        fp.write_bytes(blob)
        files[fp.name] = {
            "sha256": _sha_bytes(blob),
            "bytes": len(blob),
            "config": p.cfg.to_dict(),
            "note": p.train_note,
        }
    man = {
        "version": PIPELINE_VERSION,
        "default": DEFAULT_PIPELINE,
        "calibration_claim": "none",
        "files": files,
        "space_sha256": SP.space_hash(),
        **meta,
    }
    (out_dir / "MANIFEST.json").write_text(
        json.dumps(man, indent=1, sort_keys=True, default=str), "utf8"
    )
    man["manifest_sha256"] = _sha_bytes((out_dir / "MANIFEST.json").read_bytes())
    return man


def load_pipelines(out_dir: Path) -> dict:
    """Loads and VERIFIES every pickle against ``MANIFEST.json`` before unpickling (refuses a mismatch)."""
    out_dir = Path(out_dir)
    man = json.loads((out_dir / "MANIFEST.json").read_text("utf8"))
    if man["version"] != PIPELINE_VERSION:
        raise ValueError(f"pipeline version {man['version']} != {PIPELINE_VERSION}")
    out = {}
    for name, info in man["files"].items():
        blob = (out_dir / name).read_bytes()
        if _sha_bytes(blob) != info["sha256"]:
            raise ValueError(f"{name}: sha256 mismatch (artifact altered)")
        out[info["config"]["cls"]] = pickle.loads(blob)
    return out
