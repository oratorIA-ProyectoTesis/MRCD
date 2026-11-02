"""Write PROTOCOL.sha256.json (run once, before any number of the campaign is computed)."""

import _pin  # noqa: F401
import json
import sys
import time

from common import OUT, code_hashes, git_head, sha_norm

from mrcd.optimize import space

out = OUT / "PROTOCOL.sha256.json"
if out.exists():
    sys.exit("already frozen; refusing to overwrite")
rec = {
    "file": "PROTOCOL.md",
    "sha256": sha_norm(OUT / "PROTOCOL.md"),
    "frozen_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "git_head": git_head(),
    "space_sha256": space.space_hash(),
    "space": space.describe(),
    "budget": space.BUDGET,
    "patience": space.PATIENCE,
    "eps": space.EPS,
    "code_sha256": code_hashes(),
}
out.write_text(json.dumps(rec, indent=1), "utf8")
print(out.read_text())
