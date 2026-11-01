"""E2 filler fusion (rule + V0 score; Vc on fold 4 only) and the Holm family E2. Protocol: S3/PROTOCOL.md section 3."""

import s3_common as K  # noqa: F401

import csv
import json
import pickle
import sys
import time

import numpy as np

sys.path.insert(0, str(K.REPO / "scripts" / "baselines"))
from evaluate_audio import asr_rules  # noqa: E402

from mrcd import fusion_text as F
from mrcd.context_eval import metrics as M
from mrcd.context_eval.partition import group_folds
from mrcd.decision import best_threshold


def run_cv(rule, score, y, g, fo, base_dec):
    """Nested: everything learned on the train rows of each outer fold."""
    n = len(y)
    out = {k: np.zeros(n, int) for k in ("V0thr", "F_OR", "F_STACK")}
    st_s = np.zeros(n)
    store, info = {}, []
    for f in sorted(set(fo.tolist())):
        te, tr = np.flatnonzero(fo == f), np.flatnonzero(fo != f)
        assert not (set(g[te]) & set(g[tr]))
        t, _ = best_threshold(score[tr], y[tr])
        out["V0thr"][te] = (score[te] >= t).astype(int)
        t_or = F.fit_or(rule[tr], score[tr], y[tr])
        out["F_OR"][te] = F.apply_or(rule[te], score[te], t_or)
        ss = F.select_fit_w(F.stack_features(rule[tr], score[tr]), y[tr], g[tr])
        st_s[te] = ss["model"].score(F.stack_features(rule[te], score[te]))
        out["F_STACK"][te] = (st_s[te] >= ss["threshold"]).astype(int)
        info.append(
            {
                "fold": int(f),
                "thr_score": t,
                "t_or": t_or,
                "stack_C": ss["C"],
                "stack_thr": ss["threshold"],
            }
        )
        store[int(f)] = {"tr": tr, "inner_oof": ss["inner_oof"], "thr": ss["threshold"]}
    return out, st_s, store, info


def describe(y, p, g):
    m = M.confusion(y, p)
    m["n_groups"] = int(len(set(g)))
    m["n_pos_groups"] = int(len({a for a, b in zip(g, y) if b}))
    m["f1_ci95"] = M.boot_f1_ci(y, p, g, 2000, K.SEED)
    return m


def main():
    t0 = time.time()
    pool = K.load_pool()
    outer = K.outer_folds(pool)
    base, _ = K.v0_vc(pool)
    T = K.turns_meta()
    ids = [u["unit_id"] for u in pool]
    y = np.array([int(T[u]["filler_present"]) for u in ids])
    g = np.array([u["speaker"] for u in pool])
    rule = np.array([asr_rules(u["text_B"])["filler_word"] for u in pool])
    sc = np.array([base[u]["v0s_filler_word"] for u in ids])
    v0 = np.array([base[u]["v0_filler_word"] for u in ids])
    out, st_s, store, info = run_cv(rule, sc, y, g, outer, v0)
    preds = {"rule_asr_lexical": rule, "V0_argmax": v0, **out}
    res = {
        "filler": {
            "n": len(y),
            "pos": int(y.sum()),
            "pos_speakers": int(len({a for a, b in zip(g, y) if b})),
            "folds": info,
            "models": {k: describe(y, p, g) for k, p in preds.items()},
        }
    }
    fam = []
    for name in ("rule_asr_lexical", "V0thr"):
        d = M.paired_f1_delta(y, out["F_OR"], preds[name], g, 5000, K.SEED)
        d["test"] = f"E2-filler: F_OR minus {name}"
        fam.append(d)
    res["filler"]["paired_descriptive"] = {
        f"{a}_minus_{b}": M.paired_f1_delta(y, preds[a], preds[b], g, 5000, K.SEED)
        for a, b in (
            ("F_STACK", "V0thr"),
            ("F_STACK", "rule_asr_lexical"),
            ("V0thr", "V0_argmax"),
        )
    }
    # Vc fusion: official fold 4 only, nested CV over its speakers (Vc-favourable, descriptive)
    f4 = np.array([u["fold"] == 4 for u in pool])
    idx = np.flatnonzero(f4)
    vcs = np.array([base[ids[i]]["vcs_filler_word"] for i in idx])
    vcd = np.array([base[ids[i]]["vc_filler_word"] for i in idx])
    fo4 = group_folds(g[idx], 5, K.SEED)
    out4, _, _, info4 = run_cv(rule[idx], vcs, y[idx], g[idx], fo4, vcd)
    p4 = {
        "rule": rule[idx],
        "Vc_frozen": vcd,
        "V0_argmax": v0[idx],
        "Vcthr": out4["V0thr"],
        "F_OR_Vc": out4["F_OR"],
        "F_STACK_Vc": out4["F_STACK"],
    }
    res["filler_vc_fold4"] = {
        "n": int(len(idx)),
        "pos": int(y[idx].sum()),
        "folds": info4,
        "models": {k: describe(y[idx], p, g[idx]) for k, p in p4.items()},
        "paired_descriptive": {
            f"{a}_minus_{b}": M.paired_f1_delta(
                y[idx], p4[a], p4[b], g[idx], 5000, K.SEED
            )
            for a, b in (
                ("F_OR_Vc", "Vc_frozen"),
                ("F_OR_Vc", "rule"),
                ("F_STACK_Vc", "Vc_frozen"),
            )
        },
    }
    # full E2 family (Holm over 4): the two repetition tests come from run_e1.py
    e1 = json.loads((K.S3 / "e1" / "results.json").read_text())
    for d in e1["family_E2_rep_partial"]:
        fam.insert(len([x for x in fam if x["test"].startswith("E2-rep")]), d)
    res["family_E2"] = K.holm_table(fam)
    res["config"] = {
        "protocol_sha256": json.loads((K.S3 / "PROTOCOL.sha256.json").read_text())[
            "sha256"
        ],
        "seconds": round(time.time() - t0, 1),
    }
    out_d = K.S3 / "e2"
    out_d.mkdir(exist_ok=True)
    (out_d / "results.json").write_text(
        json.dumps(res, indent=1, default=float), "utf8"
    )
    with (out_d / "predictions_filler.csv").open(
        "w", newline="", encoding="utf8"
    ) as fh:
        wr = csv.writer(fh)
        wr.writerow(
            ["unit_id", "speaker", "fold", "y", "v0_score", "stack_score", *preds]
        )
        for j, u in enumerate(ids):
            wr.writerow(
                [
                    u,
                    g[j],
                    int(outer[j]),
                    y[j],
                    f"{sc[j]:.6f}",
                    f"{st_s[j]:.6f}",
                    *[int(p[j]) for p in preds.values()],
                ]
            )
    (out_d / "_store.pkl").write_bytes(
        pickle.dumps(
            {
                "ids": ids,
                "y": y,
                "g": g,
                "fo": outer,
                "score": sc,
                "stack": st_s,
                "store": store,
            }
        )
    )
    print("done", res["config"]["seconds"])


if __name__ == "__main__":
    main()
