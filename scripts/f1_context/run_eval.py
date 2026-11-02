"""Evaluate comparators and the contextual classifier (frozen protocol). Serial, CPU-light."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np

from mrcd.context_ambiguity.features import ACOUSTIC_KEYS, build_vocab, text_features
from mrcd.context_ambiguity.model import select_and_fit
from mrcd.context_eval.metrics import boot_f1_ci, confusion, holm, paired_f1_delta

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results/fundamentacion_20261101/f1_context"
FOLDS = (1, 2, 3, 4)
POOLED = ("este", "bueno", "o sea", "pues", "entonces", "mira/mire", "vale", "a ver")
CONTESTED = ("bueno", "a ver", "entonces", "o sea")
FILL_LIST = ("pues", "entonces", "mira/mire")
R2_LIST = ("pues", "entonces", "mira/mire", "bueno", "a ver", "o sea")


def view(row, turns, cond):
    """(words, s, e, acoustic dict|None) for a row under condition 'A' or 'B'; None if not recovered."""
    if cond == "A":
        return turns[row["turn_id"]]["words"], row["s"], row["e"], None
    r = row["asr0"]
    if r is None:
        return None
    return turns[row["turn_id"]]["asr0"]["norm"], r["s"], r["e"], r["ac"]


def turn_words(turns, tid, cond):
    return turns[tid]["words"] if cond == "A" else turns[tid]["asr0"]["norm"]


def feats_of(row, turns, cond, vocab, ac=True, punct=False):
    w, s, e, a = view(row, turns, cond)
    f = text_features(w, s, e, row["form"], vocab, row["punct"] if punct else None)
    if a is not None and ac:
        f.update({k: a[k] for k in ACOUSTIC_KEYS})
    return f


def rule_preds(rows, turns, cond):
    out = {k: np.zeros(len(rows), int) for k in ("R1", "R1L", "R2")}
    for i, r in enumerate(rows):
        v = view(r, turns, cond)
        if v is None:
            continue  # a rule cannot fire on a word the ASR does not contain
        w, s, e, a = v
        out["R1"][i] = 1
        out["R1L"][i] = int(r["form"] in FILL_LIST)
        ok = r["form"] in R2_LIST and len(w) != e - s
        if cond == "B":
            ok = ok and (a["a_gap_b"] + a["a_gap_a"] >= 0.10)
        out["R2"][i] = int(ok)
    return out


def run_cv(rows, turns, cond, ac=True, punct=False, form_only=False):
    """Outer leave-one-fold-out; returns OOF predictions (0 where not scored) and scores."""
    pred = np.zeros(len(rows), int)
    score = np.zeros(len(rows))
    infos = []
    for k in FOLDS:
        tr = [i for i, r in enumerate(rows) if r["fold"] != k and view(r, turns, cond)]
        te = [i for i, r in enumerate(rows) if r["fold"] == k and view(r, turns, cond)]
        vocab = build_vocab(
            turn_words(turns, t, cond) for t in {rows[i]["turn_id"] for i in tr}
        )
        if form_only:
            Xtr = [{f"form={rows[i]['form']}": 1.0} for i in tr]
            Xte = [{f"form={rows[i]['form']}": 1.0} for i in te]
        else:
            Xtr = [feats_of(rows[i], turns, cond, vocab, ac, punct) for i in tr]
            Xte = [feats_of(rows[i], turns, cond, vocab, ac, punct) for i in te]
        y = np.array([rows[i]["y"] for i in tr])
        m, info = select_and_fit(Xtr, y, [rows[i]["speaker"] for i in tr])
        s = m.score(Xte)
        for i, si in zip(te, s):
            score[i] = si
            pred[i] = int(si >= m.threshold)
        infos.append(
            {
                "fold": k,
                "C": info["C"],
                "threshold": info["threshold"],
                "inner_f1": info["inner_f1"],
                "n_train": len(tr),
            }
        )
    return pred, score, infos


def summarise(y, p, g, n_boot=2000):
    c = confusion(y, p)
    c["f1_ci"] = boot_f1_ci(y, p, g, n_boot)
    return c


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args()
    T = pickle.load(open(a.out / "_table.pkl", "rb"))
    turns = T["turns"]
    rows = [r for r in T["rows"] if r["form"] in POOLED]
    y = np.array([r["y"] for r in rows])
    g = np.array([r["speaker"] for r in rows])
    forms = np.array([r["form"] for r in rows])
    cont = np.isin(forms, CONTESTED)
    res: dict = {
        "n_rows": len(rows),
        "n_contested": int(cont.sum()),
        "pos_contested": int(y[cont].sum()),
        "pos_pooled": int(y.sum()),
    }

    rec = {}
    for f in POOLED:
        idx = [i for i, r in enumerate(rows) if r["form"] == f]
        sub = [i for i in idx if turns[rows[i]["turn_id"]]["asr1"] is not None]
        rec[f] = {
            "n": len(idx),
            "recovered_P0": sum(rows[i]["asr0"] is not None for i in idx),
            "n_in_P1_subset": len(sub),
            "recovered_P0_on_P1_subset": sum(rows[i]["asr0"] is not None for i in sub),
            "recovered_P1_on_P1_subset": sum(rows[i]["asr1"] is not None for i in sub),
        }
    res["recovery"] = rec

    systems = {}
    for cond in ("A", "B"):
        sysp = rule_preds(rows, turns, cond)
        if cond == "B":
            sysp["V0"] = np.array(
                [int(r["asr0"]["v0"]) if r["asr0"] else 0 for r in rows]
            )
        sysp["RF"], _, _ = run_cv(rows, turns, cond, form_only=True)
        sysp["CTX"], _, inf = run_cv(rows, turns, cond)
        if cond == "B":
            sysp["CTX_text_only"], _, _ = run_cv(rows, turns, cond, ac=False)
        else:
            sysp["CTX_punct_LEAKY"], _, _ = run_cv(rows, turns, cond, punct=True)
        systems[cond] = sysp
        res[f"cv_info_{cond}"] = inf
    res["tables"] = {}
    for cond, sysp in systems.items():
        for scope, mask in (("CONTESTED", cont), ("POOLED", np.ones(len(rows), bool))):
            res["tables"][f"{cond}/{scope}"] = {
                n: summarise(y[mask], p[mask], g[mask]) for n, p in sysp.items()
            }
        if cond == "B":
            rm = np.array([r["asr0"] is not None for r in rows])
            for scope, mask in (("CONTESTED", cont & rm), ("POOLED", rm)):
                res["tables"][f"B-rec/{scope}"] = {
                    n: summarise(y[mask], p[mask], g[mask]) for n, p in sysp.items()
                }
        res["tables"][f"{cond}/per_form"] = {
            f: {n: confusion(y[forms == f], p[forms == f]) for n, p in sysp.items()}
            for f in POOLED
        }
    res["tests"] = {}
    for cond, sysp in systems.items():
        ts = []
        for other in ("R1", "R1L", "R2", "RF"):
            d = paired_f1_delta(y[cont], sysp["CTX"][cont], sysp[other][cont], g[cont])
            d["vs"] = other
            ts.append(d)
        for t, h in zip(ts, holm([t["p_boot"] for t in ts])):
            t["p_holm"] = h
        res["tests"][cond] = ts
        res["tests"][cond + "_pooled_descriptive"] = [
            {**paired_f1_delta(y, sysp["CTX"], sysp[o], g), "vs": o}
            for o in ("R1", "RF")
        ]
    (a.out / "results.json").write_text(
        json.dumps(res, indent=1, default=float), "utf8"
    )
    pickle.dump(
        {
            "systems": systems,
            "rows_idx": [(r["turn_id"], r["s"], r["form"]) for r in rows],
        },
        open(a.out / "_preds.pkl", "wb"),
    )
    print({k: res[k] for k in ("n_rows", "n_contested", "pos_contested", "pos_pooled")})


if __name__ == "__main__":
    main()
