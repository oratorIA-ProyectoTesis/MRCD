"""Freeze the robustness addendum: writes ADDENDUM.sha256.json ONCE (refuses to overwrite)."""

import datetime
import hashlib
import json
import sys
from pathlib import Path

R = Path(__file__).resolve().parents[2]
D = R / "results" / "asr_survival_robustness_20261104"
CODE = [
    "scripts/asr_survival_robustness/make_chm_frame.py",
    "scripts/asr_survival_robustness/decode_small_gpu.py",
    "scripts/asr_survival_robustness/decode_chm150.py",
    "scripts/asr_survival_robustness/decode_chm150_crisper.py",
    "scripts/asr_survival_robustness/analyze_robustness.py",
    "scripts/asr_survival_models/analyze_models.py",
    "mrcd/asr_recovery/decode.py",
    "scripts/s2_noloop/analyze.py",
]
EXTRA = ["results/asr_survival_robustness_20261104/chm_frame.json"]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


out = D / "ADDENDUM.sha256.json"
if out.exists():
    sys.exit("already frozen")
p = D / "ADDENDUM.md"
out.write_text(
    json.dumps(
        {
            "file": "ADDENDUM.md",
            "sha256": sha(p),
            "bytes": p.stat().st_size,
            "frozen_utc": datetime.datetime.now(datetime.timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            ),
            "code_sha256": {c: sha(R / c) for c in CODE + EXTRA},
        },
        indent=1,
    ),
    "utf8",
)
print(out.read_text("utf8"))
