"""Freeze the verbatim-profile protocol: writes PROTOCOL.sha256.json ONCE (refuses to overwrite)."""

import datetime
import hashlib
import json
import subprocess
import sys
from pathlib import Path

R = Path(__file__).resolve().parents[2]
D = R / "results" / "verbatim_profile_20261107"
CODE = [
    "scripts/verbatim_profile/analyze.py",
    "scripts/verbatim_profile/latency.py",
    "mrcd/verbatim.py",
    "mrcd/asr_profiles.py",
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
head = subprocess.run(["git", "-C", str(R), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
out.write_text(
    json.dumps(
        {
            "file": "PROTOCOL.md",
            "sha256": sha(p),
            "bytes": p.stat().st_size,
            "frozen_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "git_head_before_commit": head,
            "code_sha256": {c: sha(R / c) for c in CODE},
        },
        indent=1,
    ),
    "utf8",
)
print(out.read_text("utf8"))
