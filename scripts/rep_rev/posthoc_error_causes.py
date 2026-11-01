"""POST-HOC descriptive error causes for repetition (annex). No labels are created; categories are text patterns.

* Rule false positives in B: polarity word repeated (si / no / ya ...), ASR loop (>= 4 identical consecutive tokens),
  other; and whether the human transcript also shows an unmarked adjacent duplicate.
* V0 recall on positives visible vs destroyed in the ASR.
Fold 5 is never loaded. Usage: python scripts/rep_rev/posthoc_error_causes.py
"""

import os

os.environ["OMP_NUM_THREADS"] = "4"

import csv  # noqa: E402
import json  # noqa: E402
import pickle  # noqa: E402
import sys  # noqa: E402
from collections import Counter  # noqa: E402
from pathlib import Path  # noqa: E402

REPO = Path(r"C:\zArnes-tesis\mrcd")
sys.path.insert(0, str(REPO))
from mrcd.context_eval.survival import turn_survives  # noqa: E402
from mrcd.context_eval.textfeat import POLARITY, rule_rep, tokenize  # noqa: E402

EXP = REPO / "results" / "ejecucion_material4_20261030" / "experimentos"


def kind(text_b: str, text_a: str) -> str:
    w, _, _ = tokenize(text_b)
    run = 1
    for a, b in zip(w, w[1:]):
        run = run + 1 if a == b else 1
        if run >= 4:
            return "asr_loop"
    for i in range(len(w) - 1):
        if w[i] == w[i + 1] and w[i] in POLARITY:
            return "polarity_word_repeated"
    return "other"


def main():
    pool = {
        u["unit_id"]: u
        for u in pickle.loads((EXP / "_cache" / "pool.pkl").read_bytes())
    }
    rows = list(
        csv.DictReader((EXP / "run_main" / "predictions.csv").open(encoding="utf8"))
    )
    out = {}
    fp = [
        r
        for r in rows
        if r["task"] == "rep_literal"
        and r["condition"] == "B"
        and r["model"] == "rule"
        and r["y"] == "0"
        and r["decision"] == "1"
    ]
    out["rule_FP_B"] = {
        "total": len(fp),
        "kinds": dict(
            Counter(
                kind(pool[r["unit_id"]]["text_B"], pool[r["unit_id"]]["text_A"])
                for r in fp
            )
        ),
        "human_text_has_adjacent_duplicate_unmarked": sum(
            rule_rep(pool[r["unit_id"]]["text_A"]) for r in fp
        ),
        "human_marked_only_B_C_D_repetition": sum(
            1 for r in fp if pool[r["unit_id"]]["ref"]["rep_full"] == 1
        ),
    }
    fpA = [
        r
        for r in rows
        if r["task"] == "rep_literal"
        and r["condition"] == "A"
        and r["model"] == "rule"
        and r["y"] == "0"
        and r["decision"] == "1"
    ]
    out["rule_FP_A"] = {
        "total": len(fpA),
        "kinds": dict(Counter(kind(pool[r["unit_id"]]["text_A"], "") for r in fpA)),
        "comma_between_copies": sum(
            1 for r in fpA if "," in pool[r["unit_id"]]["text_A"]
        ),
    }
    pos = [
        r
        for r in rows
        if r["task"] == "rep_literal" and r["condition"] == "B" and r["y"] == "1"
    ]
    for m in ("rule", "V0", "ctx_LR"):
        sel = [r for r in pos if r["model"] == m]
        vis = [
            turn_survives(
                "repetition",
                pool[r["unit_id"]]["ref"]["rep_marks"],
                pool[r["unit_id"]]["text_B"],
                {"A_literal_repeat"},
            )
            for r in sel
        ]
        out[f"recall_{m}_B"] = {
            "visible": [
                sum(int(r["decision"]) for r, v in zip(sel, vis) if v),
                sum(vis),
            ],
            "destroyed": [
                sum(int(r["decision"]) for r, v in zip(sel, vis) if not v),
                len(vis) - sum(vis),
            ],
        }
    (EXP / "run_main" / "posthoc_error_causes.json").write_text(
        json.dumps(out, indent=1), encoding="utf8"
    )
    print(json.dumps(out))


if __name__ == "__main__":
    main()
