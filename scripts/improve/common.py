"""Shared paths/helpers for the 2026-10-30 improvement study (new file). Reuses scripts/baselines/bl_common (imported, not edited)."""
from __future__ import annotations
import csv, json, sys
from pathlib import Path

REPO = Path(r"C:\zArnes-tesis\mrcd")
sys.path.insert(0, str(REPO / "scripts" / "baselines"))
sys.path.insert(0, str(REPO))
import bl_common as B  # noqa: E402
from bl_common import sha256_file, confusion, prf, SEED  # noqa: E402,F401

BASE = B.OUT                                   # results/baselines_20261030 (read-only here)
import os as _os
OUT = Path(_os.environ.get("MRCD_IMPROVE_OUT", str(REPO / "results" / "mejoras_20261030")))  # env override only for debugging in scratch
CACHE = OUT / "asr_cache"
FEAT = OUT / "features"
CKPT = B.CKPT
WHISPER = B.WHISPER
CORPORA = ["dihana", "epic", "ameresco"]
CLASSES3 = ["filler_word", "prolongation", "repetition"]   # trainable classes with compatible human reference
CLASSES4 = CLASSES3 + ["revision"]


def manifest(corpus: str):
    return json.loads((BASE / corpus / "manifest.json").read_text("utf8"))


def load_asr(corpus: str, unit_id: str):
    p = CACHE / corpus / f"{unit_id}.json"
    return json.loads(p.read_text("utf8"))


def dihana_turns():
    return {r["turn_id"]: r for r in csv.DictReader((BASE / "dihana" / "turns.csv").open(encoding="utf-8-sig"))}


def references(corpus: str):
    """unit_id -> dict(group, split, y_<class>=int). Original labels only."""
    if corpus == "dihana":
        T = dihana_turns()
        return {u: {"group": t["speaker"], "fold": int(t["fold"]), "split": t["split"],
                    "filler_word": int(t["filler_present"]), "prolongation": int(t["lengthening_present"]),
                    "repetition": int(t["repetition_present"]), "revision": int(t["reformulation_present"]),
                    "raw_tags": t["raw_tag_counts"], "ref_text": t["ref_text_marks_stripped"]} for u, t in T.items()}
    if corpus == "epic":
        return {r["unit_id"]: {"group": r["group"], "split": "dev(all 136)", "filler_word": int(r["filler_present"])}
                for r in csv.DictReader((BASE / "epic" / "references.csv").open(encoding="utf-8-sig"))}
    if corpus == "ameresco":
        return {r["unit_id"]: {"group": r["group"], "split": "pilot32" if r["in_pilot32"] == "True" else "new66",
                               "prolongation": int(r["prolongation_present"])}
                for r in csv.DictReader((BASE / "ameresco" / "references.csv").open(encoding="utf-8-sig"))}
    raise KeyError(corpus)


def write_cfg(outdir, name, script, extra=None, inputs=None, ckpt=True):
    """config.json = bl_common.write_config + OMP/threads + hashes of the improvement code and ASR cache manifests."""
    import os
    extra = dict(extra or {})
    try:
        import torch
        extra["torch_threads"] = torch.get_num_threads()
    except Exception:
        extra["torch_threads"] = None
    extra["env"] = {"OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"), "cpu_count_logical": os.cpu_count()}
    extra["improve_code_sha256"] = {str(p.relative_to(REPO)): sha256_file(p) for p in sorted((REPO / "scripts" / "improve").glob("*.py"))}
    inp = dict(inputs or {})
    for c in CORPORA:
        m = CACHE / c / "manifest.json"
        if m.exists():
            inp[f"asr_cache_manifest_{c}"] = sha256_file(m)
    return B.write_config(Path(outdir), name, script, extra, inp, ckpt=ckpt)
