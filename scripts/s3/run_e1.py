"""E1 (repetition text route on P1) + E2 repetition fusion. Protocol: results/mejora_objetivos_20261101/S3/PROTOCOL.md."""

import s3_common as K  # noqa: F401  (pins OMP first)

import csv
import json
import pickle
import time

import numpy as np

from mrcd import fusion_text as F
from mrcd.asr_recovery.cvfold import nested_oof_fold_aware
from mrcd.asr_recovery.stats import boot_wf1_ci, paired_wf1_delta, wconfusion
from mrcd.context_eval.textfeat import featurize, rule_rep


def main():
    t0 = time.time()
    pool = K.load_pool()
    outer = K.outer_folds(pool)
    assert all(u["fold"] != 5 for u in pool)
    base, _ = K.v0_vc(pool)
    ids, y, w, g, ix = K.frame_units(pool)
    fo = outer[ix]
    n = len(ids)
    t0txt = [K.words_text(K.asr_words("P0", u)) for u in ids]
    t1txt = [K.words_text(K.asr_words("P1", u)) for u in ids]
    X0, X1 = featurize(t0txt, "rep_literal"), featurize(t1txt, "rep_literal")
    # parent ctx_LR (trained on P0/cache text of the full pool, same nesting as the parent experiment)
    idx_all, y_all = K.task_units(pool, "rep_literal")
    Xc = featurize([pool[i]["text_B"] for i in idx_all], "rep_literal")
    nest = nested_oof_fold_aware(
        lambda tr: Xc,
        y_all,
        np.array([pool[i]["speaker"] for i in idx_all]),
        outer[idx_all],
    )
    psel = {d["fold"]: d["sel"] for d in nest["per_fold"]}

    def parent_pred(X):
        sc, dc = np.zeros(n), np.zeros(n, int)
        for j in range(n):
            s = psel[int(fo[j])]
            sc[j] = float(s["model"].score(X[j : j + 1])[0])
            dc[j] = int(sc[j] >= s["threshold"])
        return sc, dc

    sc_p0, ctx_p0 = parent_pred(X0)
    sc_p1m, ctx_p0m_on_p1 = parent_pred(X1)
    rule1 = np.array([rule_rep(t) for t in t1txt])
    rule0 = np.array([rule_rep(t) for t in t0txt])
    v0 = np.array([base[u]["v0_repetition"] for u in ids])
    vc = np.array([base[u]["vc_repetition"] for u in ids])

    a1s, a1d, a2s, a2d = np.zeros(n), np.zeros(n, int), np.zeros(n), np.zeros(n, int)
    for_ = np.zeros(n, int)
    fst = np.zeros(n, int)
    fst_s = np.zeros(n)
    folds_info, train_store = [], {}
    for f in sorted(set(fo.tolist())):
        te, tr = np.flatnonzero(fo == f), np.flatnonzero(fo != f)
        assert not (set(g[te]) & set(g[tr]))
        s1 = F.select_fit_w(X1[tr], y[tr], g[tr], w[tr])
        a1s[te] = s1["model"].score(X1[te])
        a1d[te] = (a1s[te] >= s1["threshold"]).astype(int)
        s2 = F.select_fit_w(
            np.vstack([X0[tr], X1[tr]]),
            np.r_[y[tr], y[tr]],
            np.r_[g[tr], g[tr]],
            np.r_[w[tr], w[tr]],
        )
        a2s[te] = s2["model"].score(X1[te])
        a2d[te] = (a2s[te] >= s2["threshold"]).astype(int)
        inner = s1["inner_oof"]
        t_or = (
            F.fit_or(rule1[tr], inner, y[tr], w[tr])
            if np.isfinite(inner).all()
            else F.NEVER
        )
        for_[te] = F.apply_or(rule1[te], a1s[te], t_or)
        if np.isfinite(inner).all():
            ss = F.select_fit_w(F.stack_features(rule1[tr], inner), y[tr], g[tr], w[tr])
            fst_s[te] = ss["model"].score(F.stack_features(rule1[te], a1s[te]))
            fst[te] = (fst_s[te] >= ss["threshold"]).astype(int)
        folds_info.append(
            {
                "fold": int(f),
                "n_train": int(len(tr)),
                "n_test": int(len(te)),
                "A1_C": s1["C"],
                "A1_thr": s1["threshold"],
                "A2_C": s2["C"],
                "A2_thr": s2["threshold"],
                "t_or": t_or,
            }
        )
        train_store[int(f)] = {
            "tr_ids": [ids[i] for i in tr],
            "inner_oof": inner,
            "thr": s1["threshold"],
        }

    preds = {
        "rule_P1": rule1,
        "rule_P0": rule0,
        "V0": v0,
        "ctx_LR_P0_text": ctx_p0,
        "ctx_LR_P0model_on_P1": ctx_p0m_on_p1,
        "A1_ctxLR_P1": a1d,
        "A2_ctxLR_P0P1": a2d,
        "F_OR_rule_A1": for_,
        "F_STACK_rule_A1": fst,
    }
    res = {
        "n_units": n,
        "pos": int(y.sum()),
        "pos_speakers": int(len(set(g[y == 1]))),
        "speakers": int(len(set(g))),
        "weighted_negatives": float(w[y == 0].sum()),
        "folds": folds_info,
        "models": {},
    }
    for k, p in preds.items():
        m = wconfusion(y, p, w)
        m["f1w_ci95"] = boot_wf1_ci(y, p, g, w, 2000, K.SEED)
        res["models"][k] = m
    f4 = fo == 0  # placeholder replaced below
    fold4 = np.array([pool[i]["fold"] == 4 for i in ix])
    res["vc_fold4_only_descriptive"] = {
        "n": int(fold4.sum()),
        "pos": int(y[fold4].sum()),
        "units_note": "official DIHANA fold 4; Vc-favourable",
    }
    for k in ("Vc", "A1_ctxLR_P1", "rule_P1", "V0", "ctx_LR_P0_text"):
        p = vc if k == "Vc" else preds[k]
        m = wconfusion(y[fold4], p[fold4], w[fold4])
        m["f1w_ci95"] = boot_wf1_ci(
            y[fold4], p[fold4], g[fold4], w[fold4], 2000, K.SEED
        )
        res["vc_fold4_only_descriptive"][k] = m
    res["vc_fold4_only_descriptive"]["paired_A1_minus_Vc"] = paired_wf1_delta(
        y[fold4], a1d[fold4], vc[fold4], g[fold4], w[fold4], 5000, K.SEED
    )

    fam = []
    for name in ("rule_P1", "V0", "ctx_LR_P0_text", "ctx_LR_P0model_on_P1"):
        d = paired_wf1_delta(y, a1d, preds[name], g, w, 5000, K.SEED)
        d["test"] = f"E1: A1 minus {name}"
        fam.append(d)
    res["family_E1"] = K.holm_table(fam)
    res["descriptive_paired"] = {
        f"{a}_minus_{b}": paired_wf1_delta(y, preds[a], preds[b], g, w, 5000, K.SEED)
        for a, b in (
            ("A2_ctxLR_P0P1", "A1_ctxLR_P1"),
            ("A1_ctxLR_P1", "rule_P0"),
            ("F_STACK_rule_A1", "A1_ctxLR_P1"),
        )
    }
    fam2 = []
    for name in ("rule_P1", "A1_ctxLR_P1"):
        d = paired_wf1_delta(y, for_, preds[name], g, w, 5000, K.SEED)
        d["test"] = f"E2-rep: F_OR minus {name}"
        fam2.append(d)
    res["family_E2_rep_partial"] = (
        fam2  # Holm over the full E2 family (4 tests) is applied in run_e2.py
    )
    pos = y == 1
    res["recall_split_by_P0_survival"] = None
    res["config"] = {
        "protocol_sha256": json.loads((K.S3 / "PROTOCOL.sha256.json").read_text())[
            "sha256"
        ],
        "inputs": K.file_hashes([K.REC / "subset.json"]),
        "seconds": round(time.time() - t0, 1),
    }
    out = K.S3 / "e1"
    out.mkdir(exist_ok=True)
    (out / "results.json").write_text(json.dumps(res, indent=1, default=float), "utf8")
    with (out / "predictions.csv").open("w", newline="", encoding="utf8") as fh:
        wr = csv.writer(fh)
        wr.writerow(
            [
                "unit_id",
                "speaker",
                "fold",
                "w",
                "y",
                "A1_score",
                "A2_score",
                "Fstack_score",
                *preds,
            ]
        )
        for j, u in enumerate(ids):
            wr.writerow(
                [
                    u,
                    g[j],
                    int(fo[j]),
                    f"{w[j]:.4f}",
                    y[j],
                    f"{a1s[j]:.6f}",
                    f"{a2s[j]:.6f}",
                    f"{fst_s[j]:.6f}",
                    *[int(p[j]) for p in preds.values()],
                ]
            )
    (out / "_store.pkl").write_bytes(
        pickle.dumps(
            {
                "ids": ids,
                "y": y,
                "w": w,
                "g": g,
                "fo": fo,
                "a1s": a1s,
                "train": train_store,
            }
        )
    )
    print("done", res["config"]["seconds"])


if __name__ == "__main__":
    main()
