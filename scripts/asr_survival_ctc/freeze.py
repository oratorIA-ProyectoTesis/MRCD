"""Freeze the ASR-survival-models protocol: writes PROTOCOL.sha256.json ONCE (refuses to overwrite)."""

import datetime
import hashlib
import json
import sys
from pathlib import Path

R = Path(__file__).resolve().parents[2]
D = R / "results" / "asr_survival_ctc_20261107"
CODE = [
    "scripts/asr_survival_ctc/decode_ctc.py",
    "scripts/asr_survival_ctc/analyze_ctc.py",
    "scripts/asr_survival_models/analyze_models.py",
    "scripts/s2_noloop/analyze.py",
]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


out = D / "PROTOCOL.sha256.json"
if out.exists():
    sys.exit("already frozen")
p = D / "PROTOCOL.md"
out.write_text(
    json.dumps(
        {
            "file": "PROTOCOL.md",
            "sha256": sha(p),
            "bytes": p.stat().st_size,
            "frozen_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "code_sha256": {c: sha(R / c) for c in CODE},
        },
        indent=1,
    ),
    "utf8",
)
print(out.read_text("utf8"))
