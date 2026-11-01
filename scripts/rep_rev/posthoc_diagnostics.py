"""POST-HOC diagnostics (declared in PROTOCOL_ANNEX_*.md; descriptive, outside the pre-declared family).

1. Shortcut check: ctx_LR restricted to turn-length / punctuation / filler-count features only (same nested protocol).
2. Standardised coefficients of ctx_LR fitted on the whole pool (descriptive; NOT an evaluation, in-sample).
Fold 5 is never loaded. Usage: python scripts/rep_rev/posthoc_diagnostics.py
"""

import os

os.environ["OMP_NUM_THREADS"] = "4"

import json  # noqa: E402
import pickle  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

REPO = Path(r"C:\zArnes-tesis\mrcd")
sys.path.insert(0, str(REPO))
import numpy as np  # noqa: E402

from mrcd.context_eval import metrics as M  # noqa: E402
from mrcd.context_eval.model import fit_lr, nested_oof  # noqa: E402
from mrcd.context_eval.partition import group_folds  # noqa: E402
from mrcd.context_eval.textfeat import NAMES, featurize  # noqa: E402

EXP = REPO / "results" / "ejecucion_material4_20261030" / "experimentos"
SHORTCUT = ("log_len", "log_fillers", "log_commas", "log_ellipsis", "log_dash")


def main():
    pool = pickle.loads((EXP / "_cache" / "pool.pkl").read_bytes())
    assert all(u["fold"] != 5 for u in pool)
    outer = group_folds([u["speaker"] for u in pool], 5, 13)
    out = {}
    for task in ("rep_literal", "revision"):
        idx = np.array([i for i, u in enumerate(pool) if u["ref"][task] is not None])
        y = np.array([int(pool[i]["ref"][task]) for i in idx])
        g = np.array([pool[i]["speaker"] for i in idx])
        for cond in ("A", "B"):
            X = featurize([pool[i]["text_" + cond] for i in idx], task)
            names = NAMES[task]
            keep = [names.index(n) for n in SHORTCUT]
            nest = nested_oof(X[:, keep], y, g, outer[idx])
            res = {
                "shortcut_only_confusion": M.confusion(y, nest["decision"]),
                "shortcut_only_f1_ci95": M.boot_f1_ci(y, nest["decision"], g, 2000, 13),
            }
            # full-pool in-sample coefficients, standardised (descriptive)
            m = fit_lr(X, y, 0.3)
            coef = dict(zip(names, m.clf.coef_[0].tolist()))
            res["coef_C0.3_in_sample_standardised"] = dict(
                sorted(coef.items(), key=lambda kv: -abs(kv[1]))
            )
            out[f"{task}|{cond}"] = res
    (EXP / "run_main" / "posthoc_diagnostics.json").write_text(
        json.dumps(out, indent=1), encoding="utf8"
    )
    for k, v in out.items():
        c = v["shortcut_only_confusion"]
        print(
            k,
            "shortcut-only F1",
            round(c["f1"], 3),
            v["shortcut_only_f1_ci95"],
            "tp/fp/fn",
            c["tp"],
            c["fp"],
            c["fn"],
        )
        print("   top coef", list(v["coef_C0.3_in_sample_standardised"].items())[:6])


if __name__ == "__main__":
    main()
