"""Build models/calibration_20261107/ (opt-in; default false) from the results of run_calibration.py."""

import os

os.environ["OMP_NUM_THREADS"] = "4"
os.environ.setdefault("MRCD_REPO_ROOT", r"/path/to/mrcd")
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import run_calibration as R  # noqa: E402

from mrcd import score_calibration as SC  # noqa: E402
from mrcd.asr_recovery import common as RC  # noqa: E402
from mrcd.calibration import crossfit, fit_calibrator  # noqa: E402
from mrcd.optimize import data as OD  # noqa: E402

res = json.loads((R.OUT / "results.json").read_text("utf8"))
pool = RC.load_pool()
feats = OD.load_features()
turns = {r["turn_id"]: r for r in csv.DictReader(RC.TURNS.open(encoding="utf-8-sig"))}
classes = {}
for cls in R.CLASSES:
    r = res["results"][cls]
    m = r["calibration"]["adopted"]
    if not m:
        continue
    c = R.load_class(cls, feats, pool, turns)
    cal = fit_calibrator(c["s"], c["y"], m, c["w"])
    p_cf = crossfit(c["s"], c["y"], c["spk"], m, k=3, seed=R.SEED, w=c["w"])
    t = R.best_t(p_cf, c["y"], c["w"])
    band = None
    if r["abstention"]["demonstrates_value"]:
        best, _ = R.choose_d(p_cf, c["y"], c["w"], c["spk"], t, np.random.default_rng(R.SEED + 7))
        band = best[0] if best else None
    classes[cls] = {"calibrator": cal, "threshold": t, "band": band}
man = SC.save(
    Path(R.HERE) / "models" / "calibration_20261107",
    classes,
    {
        "protocol_sha256": res["protocol_sha256"],
        "results_json_sha256": SC.sha256_file(R.OUT / "results.json"),
        "score_definition": "max over the 0.5 s windows of a turn of the V0 softmax probability of the class",
        "fitted_on": "DIHANA folds 1-4 (fold 5 never read), all speakers; method per class chosen by the frozen rule",
        "methods": {c: res["results"][c]["calibration"]["adopted"] for c in classes},
        "rf09_enabled_classes": [c for c, v in classes.items() if v["band"] is not None],
    },
)
print(json.dumps(man, indent=1))
