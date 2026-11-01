"""Shared paths and helpers for the 2026-10-30 corpus/reference audit.

Read-only on all source data. Every audit script writes one JSON file into
OUT_DIR/verified/. Nothing here creates annotations or maps labels to MRCD
classes beyond what a script states explicitly.
"""

from __future__ import annotations
import os

import hashlib
import json
from pathlib import Path

PILOT = Path(
    os.environ.get("MRCD_PILOT_ROOT", r"<PILOT_ROOT>")
)
REPO = Path(
    os.environ.get("MRCD_CORPORA_ROOT", r"<CORPORA_ROOT>")
)
CODEX = Path(r"C:/zArnes-tesis/mrcd-codex/MRCD_Plan_Avance_20261030")
MRCD = Path(r"C:/zArnes-tesis/mrcd")
OUT_DIR = MRCD / "results" / "auditoria_corpus_20261030"
VERIFIED = OUT_DIR / "verified"


def sha256(path: Path, block: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(block):
            h.update(chunk)
    return h.hexdigest()


def save(name: str, obj) -> Path:
    """Write verified/<name>.json; refuses to overwrite an existing file."""
    VERIFIED.mkdir(parents=True, exist_ok=True)
    p = VERIFIED / f"{name}.json"
    if p.exists():
        p.unlink()  # our own previous output of this same script, never source data
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {p}")
    return p


def exists(p: Path) -> bool:
    return Path(p).exists()
