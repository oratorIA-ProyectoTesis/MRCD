"""Compare per-window tensors / probabilities between (tag, mode) pairs; recompute pilot metrics per path.
usage: compare.py <tagA> <tagB> [out_prefix]   (reads results/paridad_20261030/paths/<tag>.npz)"""
import os
import csv, json, sys
from pathlib import Path
import numpy as np
OUT = Path(r"C:\zArnes-tesis\mrcd\results\paridad_20261030"); P = Path(os.environ.get("MRCD_PILOT_ROOT", r"<PILOT_ROOT>"))
KEYS = ["ac", "pros", "ling", "lpos", "lmask", "probs"]
jobs = {j["id"]: j for j in json.loads((OUT / "jobs.json").read_text("utf8"))}

def table(tag, mode):
    z = np.load(OUT / "paths" / f"{tag}.npz"); meta = json.loads((OUT / "paths" / f"{tag}.json").read_text("utf8")); t = {}
    for jid in jobs:
        if f"{jid}/{mode}/probs" not in z: continue
        st = z[f"{jid}/engine/starts"] if mode == "engine" else z[f"{jid}/starts"]
        for i, s in enumerate(st):
            t[(jid, round(float(s), 3))] = {k: z[f"{jid}/{mode}/{k}"][i] for k in KEYS if f"{jid}/{mode}/{k}" in z}
    return t, meta["labels"]

def contained(jid, s):
    j = jobs[jid]; return s >= j["a"] - 1e-6 and s + 3 <= j["b"] + 1e-6

def compare(A, B, labels):
    ks = sorted(set(A) & set(B)); r = {"n_windows": len(ks), "n_only_A": len(set(A) - set(B)), "n_only_B": len(set(B) - set(A)), "tensors": {}}
    for k in KEYS:
        d = np.array([np.abs(A[w][k].astype(np.float64) - B[w][k].astype(np.float64)).max() for w in ks if k in A[w] and k in B[w]])
        m = np.array([np.abs(A[w][k].astype(np.float64) - B[w][k].astype(np.float64)).mean() for w in ks if k in A[w] and k in B[w]])
        if len(d): r["tensors"][k] = {"max_abs": float(d.max()), "mean_of_window_mean_abs": float(m.mean()),
                                      "n_windows_nonzero": int((d > 0).sum()), "n_windows_gt_1e-6": int((d > 1e-6).sum())}
    ch = [w for w in ks if A[w]["probs"].argmax() != B[w]["probs"].argmax()]
    r["argmax_changes"] = len(ch); r["argmax_change_windows"] = [[w[0], w[1], labels[A[w]["probs"].argmax()], labels[B[w]["probs"].argmax()]] for w in ch]
    # decision per clip over target-contained windows present in BOTH
    pi = labels.index("prolongation"); dA, dB = {}, {}
    for jid in jobs:
        ws = [w for w in ks if w[0] == jid and contained(jid, w[1])]
        if ws: dA[jid] = int(any(A[w]["probs"].argmax() == pi for w in ws)); dB[jid] = int(any(B[w]["probs"].argmax() == pi for w in ws))
    r["decision_changes"] = [j for j in dA if dA[j] != dB[j]]; r["n_clips"] = len(dA)
    return r

def metrics(tab, labels):
    gold = {x["sample_id"]: int(x["source_present"]) for x in csv.DictReader((P / "references.csv").open(encoding="utf-8-sig", newline=""))}
    pi = labels.index("prolongation"); pred = {}
    for (jid, s), v in tab.items():
        if jobs[jid]["kind"] != "pilot" or not contained(jid, s): continue
        pred[jid] = max(pred.get(jid, 0), int(v["probs"].argmax() == pi))
    tp = sum(gold[j] == 1 and p == 1 for j, p in pred.items()); fp = sum(gold[j] == 0 and p == 1 for j, p in pred.items())
    fn = sum(gold[j] == 1 and p == 0 for j, p in pred.items()); tn = sum(gold[j] == 0 and p == 0 for j, p in pred.items())
    return {"n": len(pred), "tp": tp, "fp": fp, "fn": fn, "tn": tn, "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0}

if __name__ == "__main__":
    tA, tB = sys.argv[1], sys.argv[2]; res = {}
    tabs = {}
    for tag in (tA, tB):
        for mode in ("batch", "bw", "engine", "slice"):
            try: tabs[(tag, mode)], labels = table(tag, mode)
            except Exception as e: pass
    pairs = [((tA, "batch"), (tB, "batch")), ((tA, "bw"), (tB, "bw")), ((tA, "engine"), (tB, "engine")),
             ((tA, "bw"), (tB, "engine")), ((tA, "batch"), (tB, "engine")), ((tA, "bw"), (tA, "batch")),
             ((tB, "bw"), (tB, "engine")), ((tB, "batch"), (tB, "engine")), ((tB, "batch"), (tB, "slice")), ((tA, "batch"), (tA, "slice"))]
    for a, b in pairs:
        if a in tabs and b in tabs: res[f"{a[0]}/{a[1]} vs {b[0]}/{b[1]}"] = compare(tabs[a], tabs[b], labels)
    res["metrics"] = {f"{t}/{m}": metrics(v, labels) for (t, m), v in tabs.items()}
    # separate by clip kind for the pair summary
    (OUT / f"parity_{tA}_vs_{tB}.json").write_text(json.dumps(res, indent=1), "utf8")
    for k, v in res.items():
        if k == "metrics": print(json.dumps(v)); continue
        print(k, "| win", v["n_windows"], "onlyA", v["n_only_A"], "onlyB", v["n_only_B"], "| argmax_chg", v["argmax_changes"], "| dec_chg", v["decision_changes"],
              "| max", {t: f"{d['max_abs']:.2e}" for t, d in v["tensors"].items()})
