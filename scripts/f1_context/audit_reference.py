"""STEP 0: how DIHANA marks ambiguous lexical forms (folds 1-4 only). Writes counts, not text."""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

from mrcd.context_ambiguity.pool import load_turns

REPO = Path(__file__).resolve().parents[2]
DIHANA = Path(
    r"<DATASET_ROOT>"
)
TURNS = REPO / "results/baselines_20261030/dihana/turns.csv"
OUT = REPO / "results/fundamentacion_20261101/f1_context"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dihana-root", type=Path, default=DIHANA)
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args()
    turns = load_turns(a.dihana_root, TURNS)
    per_form: dict = collections.defaultdict(lambda: collections.Counter())
    spk: dict = collections.defaultdict(set)
    d_words: dict = collections.defaultdict(collections.Counter)
    n_d_total = collections.Counter()
    for u in turns:
        for tk in u["tokens"]:
            if tk.d_type:
                n_d_total[tk.d_type] += 1
        for o in u["occs"]:
            c = per_form[o.form]
            c["n"] += 1
            c[f"d:{o.d_type or 'NONE'}"] += 1
            if o.f_before or o.f_after:
                c["adjacent_f"] += 1
            spk[o.form].add(u["speaker"])
            d_words[o.d_type or "NONE"][o.form] += 1
    res = {
        "n_turns": len(turns),
        "n_speakers": len({u["speaker"] for u in turns}),
        "folds": sorted({u["fold"] for u in turns}),
        "d_token_totals": dict(n_d_total),
        "per_form": {
            k: {**dict(v), "speakers": len(spk[k])} for k, v in sorted(per_form.items())
        },
    }
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "step0_audit.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), "utf8")
    print(json.dumps(res, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
