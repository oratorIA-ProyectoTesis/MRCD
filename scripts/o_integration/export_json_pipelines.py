"""Export the packaged O-campaign pickles to pickle-free JSON for the service (decision profile ``o2026_11``).

The pickles are read only here, after their sha256 matched ``models/o_campaign_20261101/MANIFEST.json``.
Only filler_word and prolongation are exported (the profile keeps the engine decision for repetition).
Writes ``mrcd/service/profiles/o2026_11/<cls>.json`` and ``SOURCE.json`` (sha256 of every output and its source).
Run from the repo root: ``.venv/Scripts/python.exe scripts/o_integration/export_json_pipelines.py``.
"""

import hashlib
import json
from pathlib import Path

from mrcd.optimize import product_pipeline as PP
from mrcd.optimize import safe_pipeline as SAFE

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "models" / "o_campaign_20261101"
OUT = ROOT / "mrcd" / "service" / "profiles" / "o2026_11"
CLASSES = ("filler_word", "prolongation")


def main() -> None:
    man = json.loads((SRC / "MANIFEST.json").read_text("utf8"))
    pipes = PP.load_pipelines(SRC)  # refuses a sha256 mismatch before unpickling
    OUT.mkdir(parents=True, exist_ok=True)
    files = {}
    for c in CLASSES:
        blob = SAFE.dumps(SAFE.export_pipeline(pipes[c]))
        (OUT / f"{c}.json").write_bytes(blob)
        src_name = f"o_pipeline_{c}.pkl"
        files[f"{c}.json"] = {
            "sha256": hashlib.sha256(blob).hexdigest(),
            "source_pickle": src_name,
            "source_pickle_sha256": man["files"][src_name]["sha256"],
        }
    (OUT / "SOURCE.json").write_text(
        json.dumps(
            {
                "profile": "o2026_11",
                "source_manifest_version": man["version"],
                "calibration_claim": "none",
                "files": files,
            },
            indent=1,
            sort_keys=True,
        )
        + "\n",
        "utf8",
    )
    print(json.dumps(files, indent=1))


if __name__ == "__main__":
    main()
