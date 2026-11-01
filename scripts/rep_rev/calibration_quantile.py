"""Post hoc (NOT pre-declared, exploratory): equal-mass (quantile) calibration bins for ctx_LR.

Why: with 10 equal-width bins ~94% of units fall in the first bin, so the pre-declared ECE is nearly blind to the
mid-score over-confidence the reliability table shows (REVISION3 section 5). Same predictions as run_main
(``predictions.csv``: raw ``score`` and cross-fitted Platt ``calibrated``), same speaker bootstrap; only the binning
changes. Writes experimentos/calibration_quantile.json and prints a markdown table.
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import _pin_omp  # noqa: E402,F401

REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
import csv  # noqa: E402

import numpy as np  # noqa: E402

from mrcd.context_eval import metrics as M  # noqa: E402

RUN = REPO / "results/ejecucion_material4_20261030/experimentos"
BINS = 10


def reliability_q(y, s, bins=BINS):
    """Equal-count bins by rank of the score (stable order for ties): list of (n, mean_score, frac_pos)."""
    y = np.asarray(y, float)
    s = np.asarray(s, float)
    order = np.argsort(s, kind="stable")
    out = []
    for idx in np.array_split(order, bins):
        if len(idx):
            out.append((len(idx), float(s[idx].mean()), float(y[idx].mean())))
    return out


def ece_q(y, s, bins=BINS):
    rel = reliability_q(y, s, bins)
    n = sum(r[0] for r in rel)
    return float(sum(r[0] * abs(r[1] - r[2]) for r in rel) / n) if n else float("nan")


def main():
    rows = [r for r in csv.DictReader((RUN / "run_main/predictions.csv").open(encoding="utf8")) if r["model"] == "ctx_LR"]
    res = {}
    for task in sorted({r["task"] for r in rows}):
        for cond in ("A", "B"):
            sel = [r for r in rows if r["task"] == task and r["condition"] == cond and r["score"] != ""]
            if not sel:
                continue
            y = np.array([int(r["y"]) for r in sel])
            raw = np.array([float(r["score"]) for r in sel])
            cal = np.array([float(r["calibrated"]) for r in sel])
            g = np.array([r["speaker"] for r in sel])
            d = M.boot_diff(ece_q, y, raw, cal, g, 1000, 13)
            res[f"{task}|{cond}"] = {
                "n": len(y),
                "n_pos": int(y.sum()),
                "ece_q_raw": ece_q(y, raw),
                "ece_q_cal": ece_q(y, cal),
                "ece_q_raw_minus_cal": d,
                "ece_equal_width_raw": M.ece(y, raw),
                "reliability_q_raw": reliability_q(y, raw),
                "reliability_q_cal": reliability_q(y, cal),
            }
    (RUN / "calibration_quantile.json").write_text(json.dumps(res, indent=1), "utf8")
    print("| task | cond | n / pos | ECE equal-width raw | ECE quantile raw | ECE quantile cal | dECE_q raw-cal [95% CI] |\n|---|---|---|---|---|---|---|")
    for k, v in res.items():
        t, c = k.split("|")
        dd = v["ece_q_raw_minus_cal"]
        print(f"| {t} | {c} | {v['n']} / {v['n_pos']} | {v['ece_equal_width_raw']:.4f} | {v['ece_q_raw']:.4f} | {v['ece_q_cal']:.4f} | {dd['delta']:+.4f} [{dd['ci'][0]:+.4f}, {dd['ci'][1]:+.4f}] |")


if __name__ == "__main__":
    main()
