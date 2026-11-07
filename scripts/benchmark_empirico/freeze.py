"""Freeze the empirical-benchmark protocol: writes PROTOCOL.sha256.json ONCE (refuses to overwrite)."""

import datetime
import hashlib
import json
import subprocess
import sys
from pathlib import Path

R = Path(__file__).resolve().parents[2]
D = R / "results" / "benchmark_empirico_20261107"
CODE = [
    "scripts/benchmark_empirico/analyze.py",
    "scripts/benchmark_empirico/run_opaf.py",
    "mrcd/asr_recovery/stats.py",
    "mrcd/context_eval/textfeat.py",
    "mrcd/optimize/blocks.py",
]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


out = D / "PROTOCOL.sha256.json"
if out.exists():
    sys.exit("already frozen")
p = D / "PROTOCOL.md"
head = subprocess.run(
    ["git", "-C", str(R), "rev-parse", "--short", "HEAD"],
    capture_output=True,
    text=True,
).stdout.strip()
out.write_text(
    json.dumps(
        {
            "file": "PROTOCOL.md",
            "sha256": sha(p),
            "bytes": p.stat().st_size,
            "frozen_utc": datetime.datetime.now(datetime.timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            ),
            "git_head_before_commit": head,
            "code_sha256": {c: sha(R / c) for c in CODE},
        },
        indent=1,
    ),
    "utf8",
)
print(out.read_text("utf8"))
