"""HISTORICAL, CONTAMINATED single report on DIHANA fold 5 (filler_word, prolongation). Run once, last, after the packaged
pipelines exist. Nothing here feeds any decision. Refuses to run before README_S8.md exists."""

import _pin  # noqa: F401
import json

import numpy as np

from common import MODELS, OUT, gate, verify_frozen

from mrcd.optimize import data as DT
from mrcd.optimize import evaluate as EV
from mrcd.optimize import product_pipeline as PP


def main():
    gate()
    verify_frozen()
    out_path = OUT / "fold5_historical_contaminated.json"
    if out_path.exists():
        raise SystemExit("the single fold-5 report already exists")
    pipes = PP.load_pipelines(MODELS)
    res = {
        "label": "historical, contaminated: fold 5 was evaluated several times in earlier phases; not used for any decision"
    }
    from mrcd.constants import LABELS
    from mrcd.decision import any_window_argmax

    for cls in ("filler_word", "prolongation"):
        recs, y, g = DT.load_fold5_historical(cls)
        o = pipes[cls].predict(recs)
        v0 = np.array([any_window_argmax(r["probs"], list(LABELS), cls) for r in recs])
        res[cls] = {
            "n": int(len(y)),
            "pos": int(y.sum()),
            "optimised": EV.describe(y, o["decision"], g),
            "V0_argmax": EV.describe(y, v0, g),
            "delta_vs_V0": EV.paired(y, o["decision"], v0, g),
        }
    out_path.write_text(json.dumps(res, indent=1, default=float), "utf8")
    print(out_path)


if __name__ == "__main__":
    main()
