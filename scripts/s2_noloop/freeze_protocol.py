"""Write PROTOCOL.sha256.json for the S2 protocol (run once, before any alternative is decoded)."""

import hashlib
import json
import sys
import time
from pathlib import Path

p = Path(__file__).resolve().parents[2] / "results" / "mejora_objetivos_20261101" / "S2" / "PROTOCOL.md"
out = p.with_name("PROTOCOL.sha256.json")
if out.exists():
    sys.exit("already frozen; refusing to overwrite")
b = p.read_bytes()
out.write_text(
    json.dumps(
        {
            "file": p.name,
            "sha256": hashlib.sha256(b).hexdigest(),
            "bytes": len(b),
            "frozen_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
        indent=1,
    ),
    "utf8",
)
print(out.read_text())
