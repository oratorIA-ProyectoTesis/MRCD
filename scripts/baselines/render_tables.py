"""Render markdown tables from metrics_audio.json and text/text_metrics.json into results/baselines_20261030/_tables.md."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from bl_common import OUT

f = lambda x: "n/a" if x != x else f"{x:.2f}"


def row(m):
    ci = m.get("ci95_by_group")
    c = f"[{f(ci['f1'][0])}-{f(ci['f1'][1])}]" if ci else ""
    flag = "" if m["support_pos"] >= 30 else " UNSTABLE(<30 pos)"
    return f"| {m['comparator']} | {m['partition']} | {m['n']} | {m['support_pos']} | {m['tp']}/{m['fp']}/{m['fn']}/{m['tn']} | {f(m['precision'])} | {f(m['recall'])} | {f(m['f1'])} {c}{flag} |"


HDR = "| comparator | partition | n | pos | TP/FP/FN/TN | P | R | F1 [95% CI, bootstrap by group] |\n|---|---|---|---|---|---|---|---|"
a = json.loads((OUT / "metrics_audio.json").read_text("utf8"))
t = json.loads((OUT / "text" / "text_metrics.json").read_text("utf8"))
o = []
order = {
    "always_present": 0,
    "always_absent": 1,
    "rule_asr_lexical": 2,
    "rule_existing_autolabeler": 3,
    "mrcd_checkpoint": 4,
}
prt = {"dev": 0, "tune": 1, "test": 2, "pilot32": 0, "new66": 1, "all": 3}
corp = []
for m in a["metrics"]:
    k = (m["corpus"], m["class"])
    if k not in corp:
        corp.append(k)
for corpus, cls in corp:
    ms = [m for m in a["metrics"] if m["corpus"] == corpus and m["class"] == cls]
    ms.sort(key=lambda m: (prt.get(m["partition"], 0), order.get(m["comparator"], 9)))
    o.append(f"\n#### {corpus} / {cls}\n\n{HDR}")
    o += [row(m) for m in ms]
o.append("\n### Text-only tasks (T)\n")
for corpus, task in sorted({(m["corpus"], m["task"]) for m in t}):
    ms = [m for m in t if m["corpus"] == corpus and m["task"] == task]
    ms.sort(
        key=lambda m: (
            ["dev", "tune", "test", "all"].index(m["partition"]),
            m["comparator"],
        )
    )
    o.append(f"\n#### {corpus} / {task}\n\n{HDR}")
    o += [row(m) for m in ms]
ex = a["extra"]
o.append(
    "\n### AMERESCO per conversation (pool of 98; pos = <alargamiento/> segments)\n\n| conversation | comparator | n | pos | TP/FP/FN/TN | P | R | F1 |\n|---|---|---|---|---|---|---|---|"
)
for g, d in ex["ameresco_per_conversation"].items():
    for c, m in d.items():
        o.append(
            f"| {g} | {c} | {m['n']} | {m['support_pos']} | {m['tp']}/{m['fp']}/{m['fn']}/{m['tn']} | {f(m['precision'])} | {f(m['recall'])} | {f(m['f1'])} |"
        )
(OUT / "_tables.md").write_text("\n".join(o), encoding="utf8")
print(len(o), "lines")
