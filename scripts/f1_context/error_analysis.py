"""Per-form table and error index list (turn id + word offsets only; no transcript text is written)."""

from __future__ import annotations

import csv
import json
import pickle
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parents[2] / "results/fundamentacion_20261101/f1_context"


def main() -> None:
    res = json.loads((OUT / "results.json").read_text("utf8"))
    P = pickle.load(open(OUT / "_preds.pkl", "rb"))
    T = pickle.load(open(OUT / "_table.pkl", "rb"))
    rows = [r for r in T["rows"] if r["form"] != "eh"]
    ctx = P["systems"]["A"]["CTX"]
    rf = P["systems"]["A"]["RF"]
    with (OUT / "errors_A_ctx.csv").open("w", newline="", encoding="utf8") as f:
        w = csv.writer(f)
        w.writerow(["turn_id", "word_start", "form", "d_type_ref", "y", "ctx", "form_prior"])
        for r, c, p in zip(rows, ctx, rf):
            if c != r["y"]:
                w.writerow([r["turn_id"], r["s"], r["form"], r["d_type"], r["y"], int(c), int(p)])
    lines = ["| form | n | pos | R1 F1 | form-prior F1 | CTX P/R/F1 (A) | CTX P/R/F1 (B-e2e) |", "|---|---|---|---|---|---|---|"]
    for form, d in res["tables"]["A/per_form"].items():
        b = res["tables"]["B/per_form"][form]

        def g(x):
            return "n/a" if x != x else f"{x:.2f}"

        def prf(c):
            return f"{g(c['precision'])}/{g(c['recall'])}/{g(c['f1'])}"

        lines.append(
            f"| {form} | {d['R1']['n']} | {d['R1']['support_pos']} | "
            f"{g(d['R1']['f1'])} | {g(d['RF']['f1'])} | {prf(d['CTX'])} | {prf(b['CTX'])} |"
        )
    (OUT / "per_form_table.md").write_text("\n".join(lines) + "\n", "utf8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
