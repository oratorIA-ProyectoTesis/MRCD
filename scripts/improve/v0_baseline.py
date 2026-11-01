"""Step 1: V0 (current baseline) recomputed ONLY from the frozen ASR caches: always_present / always_absent / existing rule /
ASR lexical rule / base checkpoint with the 'any window argmax' rule, per corpus, class and partition (dev/tune/test).
Writes results/mejoras_20261030/v0/{V0_metrics.json,V0_BASELINE.md,config.json,predictions_*.csv}."""
import csv, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from common import OUT, write_cfg  # noqa
from evalkit import *  # noqa

V0 = OUT / "v0"; V0.mkdir(parents=True, exist_ok=True)
COMPS = {**COMPARATORS, "mrcd_checkpoint_V0": pred_argmax}


def main():
    allm, md = [], ["# V0 on frozen ASR caches (OMP=4)\n", "Cell: TP/FP/FN/TN. F1 CI = 95% bootstrap by speaker/actor. Rows with <30 positives are UNSTABLE.\n"]
    spec = [("dihana", CLASSES4, ["dev", "tune", "test"]), ("epic", ["filler_word"], ["dev(all 136)"]), ("ameresco", ["prolongation"], ["pilot32", "new66"])]
    for corpus, classes, parts in spec:
        rows = unit_table(corpus, "zero")
        part_rows = {p: [r for r in rows if r["split"] == p] for p in parts}
        part_rows["all"] = rows
        for c in classes:
            md += [f"\n#### {corpus} / {c}\n", "| comparator | partition | n | pos | TP/FP/FN/TN | P | R | F1 |", "|---|---|---|---|---|---|---|---|"]
            for p, rr in part_rows.items():
                if not rr: continue
                t = eval_table(rr, COMPS, [c], corpus, p, n_boot=1000 if corpus == "ameresco" else 2000)
                allm += t; md += [fmt_row(m) for m in t]
        if corpus == "dihana":
            with (V0 / "predictions_dihana.csv").open("w", encoding="utf-8-sig", newline="") as f:
                w = csv.writer(f); w.writerow(["unit_id", "speaker", "split", "dur_s", "padded"] + [f"{k}_{c}" for k in ("y", "V0", "asr", "rule") for c in CLASSES4])
                for r in rows:
                    w.writerow([r["unit_id"], r["group"], r["split"], round(r["dur_s"], 3), int(r["padded"])] + [r["ref"][c] for c in CLASSES4] + [pred_argmax(r, c) for c in CLASSES4]
                               + [r["asr"][c] if r["asr"][c] is not None else "" for c in CLASSES4] + [r["rule_existing"][c] for c in CLASSES4])
    (V0 / "V0_metrics.json").write_text(json.dumps(allm, indent=1, default=float))
    (V0 / "V0_BASELINE.md").write_text("\n".join(md), encoding="utf8")
    write_cfg(V0, "v0_baseline", __file__, {"n_boot": 2000, "units_per_partition": {m["corpus"] + "/" + m["partition"]: m["n"] for m in allm if m["comparator"] == "always_present" and m["class"] in ("filler_word", "prolongation")}})
    print("\n".join(md))


if __name__ == "__main__":
    main()
