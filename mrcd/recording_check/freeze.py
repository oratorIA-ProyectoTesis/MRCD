"""Freeze (sha256 + date + git head) of the technical pre-registration and of the code that computes its report.

The freeze file is a local, tamper-evident record: it proves that the text and code hashed here are the ones that
existed when it was written. For an external timestamp, send the hash of the freeze file itself to the advisor.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from mrcd.recording_check.config import config_hash

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FILES = (
    "docs/via-b/PREREGISTRO_TECNICO.md",
    "mrcd/recording_check/config.py",
    "mrcd/recording_check/stats.py",
    "mrcd/recording_check/technical.py",
    "mrcd/recording_check/verify_recording.py",
    "mrcd/av/config.py",
    "mrcd/av/gate.py",
)


class FreezeError(RuntimeError):
    """The frozen files do not match the freeze record (or the record is missing)."""


def _sha(path: Path) -> str:
    """sha256 of the file with line endings normalised to LF, so a CRLF checkout (git autocrlf) hashes the same."""
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _git(root: Path, *args: str) -> str | None:
    try:
        r = subprocess.run(
            ["git", "-C", str(root), *args], capture_output=True, text=True, timeout=20
        )
        return r.stdout.strip() if r.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def freeze(
    out: Path,
    *,
    root: Path = REPO_ROOT,
    files: Iterable[str] = DEFAULT_FILES,
    tag: str = "preregistro",
    note: str = "",
) -> dict:
    entries = {}
    for rel in files:
        p = root / rel
        if not p.is_file():
            raise FreezeError(f"cannot freeze missing file {rel}")
        entries[rel] = _sha(p)
    try:
        from mrcd.av.config import config_hash as av_hash

        av = av_hash()
    except Exception:  # pragma: no cover - av config always importable in the repo
        av = None
    status = _git(root, "status", "--porcelain", "--", *entries)
    rec = {
        "tag": tag,
        "frozen_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "files_sha256": entries,
        "config_hash": config_hash(),
        "av_gate_config_hash": av,
        "git_head": _git(root, "rev-parse", "HEAD"),
        "frozen_files_uncommitted": bool(status) if status is not None else None,
        "note": note,
    }
    out.write_text(json.dumps(rec, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return rec


def verify_freeze(freeze_file: Path, *, root: Path = REPO_ROOT) -> list[str]:
    """Return the list of mismatches (empty = intact)."""
    if not freeze_file.is_file():
        return [f"freeze record missing: {freeze_file}"]
    rec = json.loads(freeze_file.read_text(encoding="utf-8"))
    bad = []
    for rel, digest in rec["files_sha256"].items():
        p = root / rel
        if not p.is_file():
            bad.append(f"{rel}: missing")
        elif _sha(p) != digest:
            bad.append(f"{rel}: sha256 differs from the freeze")
    if rec.get("config_hash") != config_hash():
        bad.append("recording_check config_hash differs from the freeze")
    return bad


def require_frozen(freeze_file: Path, *, root: Path = REPO_ROOT) -> dict:
    bad = verify_freeze(freeze_file, root=root)
    if bad:
        raise FreezeError(
            "technical report refused, freeze not intact: " + "; ".join(bad)
        )
    return json.loads(freeze_file.read_text(encoding="utf-8"))
