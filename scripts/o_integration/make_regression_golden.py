"""Golden regression set: decisions of the CAMPAIGN PICKLE pipelines on 40 saved DIHANA units (folds 1-4 only).

Writes ``tests/data/o2026_11_regression.json``. The service profile (JSON artifacts) must reproduce these decisions.
Run from the repo root: ``.venv/Scripts/python.exe scripts/o_integration/make_regression_golden.py``.
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "improve"))
from common import CACHE, manifest  # noqa: E402

from mrcd.optimize import data as OD  # noqa: E402
from mrcd.optimize import product_pipeline as PP  # noqa: E402

N = 40


def main() -> None:
    feats = OD.load_features()
    units = [m["unit_id"] for m in manifest("dihana") if m["fold"] != 5 and m["unit_id"] in feats]
    rng = np.random.default_rng(13)
    pick = sorted(units[i] for i in rng.choice(len(units), size=N, replace=False))
    recs = [OD._rec(u, feats, ("p0",), None) for u in pick]
    pipes = PP.load_pipelines(ROOT / "models" / "o_campaign_20261101")
    out = {"source": "models/o_campaign_20261101 pickles (sha256-verified)", "units": pick, "classes": {}}
    for c in ("filler_word", "prolongation"):
        r = pipes[c].predict(recs)
        out["classes"][c] = {
            "score": [round(float(x), 12) for x in r["score"]],
            "decision": [int(x) for x in r["decision"]],
            "state": [str(x) for x in r["state"]],
        }
    p = ROOT / "tests" / "data" / "o2026_11_regression.json"
    p.write_text(json.dumps(out, indent=1) + "\n", "utf8")
    print({c: sum(v["decision"]) for c, v in out["classes"].items()}, "positives of", N)


if __name__ == "__main__":
    main()
