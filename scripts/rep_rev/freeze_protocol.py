"""Freeze PROTOCOL.md: write its sha256 and a UTC timestamp. Refuses to overwrite an existing freeze."""

import hashlib
import json
import sys
import time
from pathlib import Path

EXP = Path(r"C:\zArnes-tesis\mrcd\results\ejecucion_material4_20261030\experimentos")


def main() -> None:
    proto = EXP / "PROTOCOL.md"
    out = EXP / "PROTOCOL.sha256.json"
    if out.exists():
        sys.exit("already frozen; record deviations in a dated annex instead")
    h = hashlib.sha256(proto.read_bytes()).hexdigest()
    rec = {
        "file": proto.name,
        "sha256": h,
        "frozen_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "bytes": proto.stat().st_size,
    }
    out.write_text(json.dumps(rec, indent=1), encoding="utf8")
    print(json.dumps(rec))


if __name__ == "__main__":
    main()
