"""Fold 5 is HISTORICAL: this script only re-tabulates, AS IS, the frozen V0 / Vc predictions of the earlier study.

Nothing is fitted, selected or compared against any new model here. Reference marks are those of the earlier
study (full repetition mark, grouped revision mark); the rep_literal sub-type is NOT derived for fold 5.
Usage: python scripts/rep_rev/historical_fold5.py
"""

import csv
import json
import sys
from pathlib import Path

REPO = Path(r"C:\zArnes-tesis\mrcd")
sys.path.insert(0, str(REPO))
from mrcd.context_eval.metrics import boot_f1_ci, confusion  # noqa: E402

SRC = REPO / "results" / "mejoras_20261030" / "final" / "predictions_dihana_fold5.csv"
OUT = (
    REPO
    / "results"
    / "ejecucion_material4_20261030"
    / "experimentos"
    / "historical_fold5.json"
)


def main():
    rows = list(csv.DictReader(SRC.open(encoding="utf-8-sig")))
    g = [r["speaker"] for r in rows]
    res = {
        "status": "HISTORICAL, copied as is from the frozen earlier evaluation; not used to adjust anything",
        "source": str(SRC),
        "n_units": len(rows),
        "n_speakers": len(set(g)),
    }
    for task, ycol, models in (
        ("rep_full", "y_repetition", ("V0_repetition", "Vc_repetition")),
        ("revision", "y_revision", ("V0_revision",)),
    ):
        y = [int(r[ycol]) for r in rows]
        res[task] = {}
        for m in models:
            p = [int(r[m]) for r in rows]
            c = confusion(y, p)
            c["f1_ci95"] = boot_f1_ci(y, p, g, 2000, 13)
            res[task][m] = c
    OUT.write_text(json.dumps(res, indent=1), encoding="utf8")
    print(
        json.dumps(
            {
                k: {m: (round(c["f1"], 3), c["support_pos"]) for m, c in v.items()}
                for k, v in res.items()
                if isinstance(v, dict) and k in ("rep_full", "revision")
            }
        )
    )


if __name__ == "__main__":
    main()
