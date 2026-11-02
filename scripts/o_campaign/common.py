"""Shared paths, frozen-protocol verification and the S8 compute gate for the O-campaign scripts."""

import hashlib
import json
import subprocess
from pathlib import Path

REPO = Path(r"<repo>")
OUT = REPO / "results" / "mejora_objetivos_20261101" / "O_campaign"
S8_README = REPO / "results" / "mejora_objetivos_20261101" / "S8_incremental" / "README_S8.md"
MODELS = REPO / "models" / "o_campaign_20261101"
CODE = [*sorted((REPO / "mrcd" / "optimize").glob("*.py"))]


def sha_norm(p: Path) -> str:
    """sha256 of a text file with CRLF normalised to LF (stable across checkouts)."""
    return hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def code_hashes() -> dict:
    return {str(p.relative_to(REPO)).replace("\\", "/"): sha_norm(p) for p in CODE}


def git_head() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()


def verify_frozen() -> dict:
    fz = json.loads((OUT / "PROTOCOL.sha256.json").read_text("utf8"))
    if sha_norm(OUT / "PROTOCOL.md") != fz["sha256"]:
        raise SystemExit("PROTOCOL.md changed after freezing: write an annex instead")
    from mrcd.optimize import space

    if space.space_hash() != fz["space_sha256"]:
        raise SystemExit("search space changed after freezing")
    changed = [k for k, v in code_hashes().items() if fz["code_sha256"].get(k) != v]
    if changed:
        print("WARNING code differs from the frozen hashes (declare in an annex):", changed)
    return fz


def gate():
    if not S8_README.exists():
        raise SystemExit(f"heavy compute is gated until {S8_README} exists")
