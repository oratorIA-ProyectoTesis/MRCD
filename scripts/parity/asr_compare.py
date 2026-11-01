"""Compare ASR word lists: pilot cache vs run1 vs run2 (same frozen model/params)."""
import json
from pathlib import Path
OUT = Path(r"C:\zArnes-tesis\mrcd\results\paridad_20261030")
jobs = json.loads((OUT / "jobs.json").read_text("utf8"))
def ld(p): return json.loads(Path(p).read_text("utf8"))
def cmp(a, b):
    same_text = [w["text"] for w in a] == [w["text"] for w in b]
    if len(a) != len(b) or not same_text: return {"text_equal": False, "n_a": len(a), "n_b": len(b), "max_time_diff": None}
    d = max([0.0] + [max(abs(x["start"] - y["start"]), abs(x["end"] - y["end"]), abs(x["prob"] - y["prob"])) for x, y in zip(a, b)])
    return {"text_equal": True, "n_a": len(a), "n_b": len(b), "max_time_prob_diff": d}
res = {}
for j in jobs:
    r1 = ld(OUT / "asr_run1" / f"{j['id']}_asr.json"); r2 = ld(OUT / "asr_run2" / f"{j['id']}_asr.json")
    row = {"run1_vs_run2": cmp(r1, r2)}
    if j["pilot_words"]: row["pilot_vs_run1"] = cmp(ld(j["pilot_words"]), r1); row["pilot_vs_run2"] = cmp(ld(j["pilot_words"]), r2)
    res[j["id"]] = row
summ = {k: {"clips": sum(k in r for r in res.values()), "identical_text": sum(r[k]["text_equal"] for r in res.values() if k in r),
            "max_time_prob_diff": max([r[k].get("max_time_prob_diff") or 0 for r in res.values() if k in r])} for k in ("run1_vs_run2", "pilot_vs_run1", "pilot_vs_run2")}
(OUT / "asr_variability.json").write_text(json.dumps({"summary": summ, "per_clip": res}, indent=1), "utf8"); print(json.dumps(summ))
