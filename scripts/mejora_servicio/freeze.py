"""Freeze (sha256 manifest) of the protocol, the code under test, the harness and the prepared inputs.

    .venv\\Scripts\\python.exe scripts\\mejora_servicio\\freeze.py            # write PROTOCOL.sha256.json (once)
    .venv\\Scripts\\python.exe scripts\\mejora_servicio\\freeze.py --verify   # what every run script calls

Text files are hashed after normalising CRLF to LF, so a checkout with converted line endings verifies too.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results" / "mejora_servicio_20261102"
MANIFEST = OUT / "PROTOCOL.sha256.json"

FILES = [
    "results/mejora_servicio_20261102/PROTOCOL.md",
    "scripts/mejora_servicio/common.py",
    "scripts/mejora_servicio/freeze.py",
    "scripts/mejora_servicio/m1_prepare.py",
    "scripts/mejora_servicio/m1_run.py",
    "scripts/mejora_servicio/m1_eval.py",
    "scripts/mejora_servicio/m2_run.py",
    "scripts/mejora_servicio/m2_eval.py",
    "mrcd/service/analyzer.py",
    "mrcd/service/decision_profile.py",
    "mrcd/service/s3_features.py",
    "mrcd/service/contract.py",
    "mrcd/service/incremental/session.py",
    "mrcd/service/incremental/cached_asr.py",
    "mrcd/service/incremental/cache.py",
    "mrcd/engine.py",
    "mrcd/fusion_text.py",
    "mrcd/context_eval/textfeat.py",
    "mrcd/asr_recovery/stats.py",
    "mrcd/service/profiles/o2026_11/filler_word.json",
    "mrcd/service/profiles/o2026_11/prolongation.json",
    "mrcd/service/experimental/rep_ctx_p1_route.json",
    "results/mejora_servicio_20261102/m1/frame.json",
    "results/mejora_servicio_20261102/m1/sessions.json",
    "results/mejora_servicio_20261102/m1/routes/fold_0.json",
    "results/mejora_servicio_20261102/m1/routes/fold_1.json",
    "results/mejora_servicio_20261102/m1/routes/fold_2.json",
    "results/mejora_servicio_20261102/m1/routes/fold_3.json",
    "results/mejora_servicio_20261102/m1/routes/fold_4.json",
]


def file_sha(rel: str) -> str:
    b = (REPO / rel).read_bytes()
    if not rel.endswith((".pkl", ".bin")):
        b = b.replace(b"\r\n", b"\n")
    return hashlib.sha256(b).hexdigest()


def freeze() -> dict:
    if MANIFEST.exists():
        raise SystemExit("already frozen; changes after the freeze go to a dated annex")
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True
    ).stdout.strip()
    m = {
        "frozen_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_head": head,
        "note": "code under test is uncommitted at freeze time; its hashes are below",
        "files": {f: file_sha(f) for f in FILES},
    }
    m["protocol_sha256"] = m["files"]["results/mejora_servicio_20261102/PROTOCOL.md"]
    MANIFEST.write_text(json.dumps(m, indent=1), "utf8")
    return m


def verify(extra_allowed: tuple[str, ...] = ()) -> None:
    m = json.loads(MANIFEST.read_text("utf8"))
    annexes = sorted(OUT.glob("PROTOCOL_ANNEX_*.sha256.json"))
    allowed = dict(m["files"])
    for a in (
        annexes
    ):  # a dated annex may re-pin files changed after the freeze (declared there)
        allowed.update(json.loads(a.read_text("utf8")).get("files", {}))
    bad = [f for f, h in allowed.items() if f not in extra_allowed and file_sha(f) != h]
    if bad:
        raise SystemExit(f"frozen files changed without an annex: {bad}")


if __name__ == "__main__":
    if "--verify" in sys.argv:
        verify()
        print("ok")
    else:
        print(json.dumps(freeze(), indent=1))
