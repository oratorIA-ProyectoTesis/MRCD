"""O2 (ASR-cached signals) and O3 (acoustic DTW) on ALL label-defined turns of folds 1-4, nested speaker-grouped CV (parent partition).

Usage: python s4_eval_o2_o3.py   (needs features/acoustic.pkl from s3)
"""

import _pin  # noqa: F401  (MUST be first)
import csv
import hashlib
import json
import pickle
import platform
import time

import numpy as np

from mrcd.asr_recovery import acoustic as AC
from mrcd.asr_recovery import signals as SG
from mrcd.asr_recovery.common import (
    ASR,
    WAV,
    run_config,
    LITERAL_SUBTYPES,
    OUT,
    PARENT,
    SEED,
    TASKS,
    load_pool,
    outer_folds,
    sha,
    task_units,
)
from mrcd.asr_recovery.cvfold import NanFill, nested_oof_fold_aware
from mrcd.asr_recovery.stats import boot_wf1_ci, paired_wf1_delta, wconfusion
from mrcd.context_eval.survival import turn_survives
from mrcd.context_eval.textfeat import featurize, rule_rep, rule_rev


def load_parent_decisions():
    d = {}
    with (PARENT / "run_main" / "predictions.csv").open(encoding="utf8") as f:
        for r in csv.DictReader(f):
            if r["task"] in TASKS and r["model"] in ("rule", "ctx_LR"):
                d[(r["task"], r["condition"], r["model"], r["unit_id"])] = int(
                    r["decision"]
                )
    return d


def main():
    t0 = time.time()
    pool = load_pool()
    outer = outer_folds(pool)
    parent = load_parent_decisions()
    ac = pickle.loads((OUT / "features" / "acoustic.pkl").read_bytes())
    recs = {
        u["unit_id"]: (lambda j: {"words": j["words"], "n_samples": j["n_samples"]})(
            json.loads((ASR / f"{u['unit_id']}.json").read_text("utf8"))
        )
        for u in pool
    }
    res = {"tasks": {}}
    rows = []
    for task in TASKS:
        idx, y = task_units(pool, task)
        g = np.array([pool[i]["speaker"] for i in idx])
        fo = outer[idx]
        uid = [pool[i]["unit_id"] for i in idx]
        textsB = [pool[i]["text_B"] for i in idx]
        Xc = featurize(textsB, task)
        rule_fn = rule_rev if task == "revision" else rule_rep
        rule_B = np.array([rule_fn(t) for t in textsB])
        A_rule = np.array([parent[(task, "A", "rule", u)] for u in uid])
        A_ctx = np.array([parent[(task, "A", "ctx_LR", u)] for u in uid])
        P_ctx = np.array([parent[(task, "B", "ctx_LR", u)] for u in uid])
        P_rule = np.array([parent[(task, "B", "rule", u)] for u in uid])
        kind, key = (
            ("revision", "rev_marks")
            if task == "revision"
            else ("repetition", "rep_marks")
        )
        subs = LITERAL_SUBTYPES if task == "rep_literal" else None
        vis = np.array(
            [
                bool(y[j])
                and turn_survives(kind, pool[i]["ref"][key], pool[i]["text_B"], subs)
                for j, i in enumerate(idx)
            ]
        )
        destroyed = (y == 1) & ~vis
        R = [recs[u] for u in uid]
        Xac_a = np.array([ac["feats"][u][0] for u in uid], float)
        Xac_v = np.array([ac["feats"][u][1] for u in uid], float)

        def sig_X(tr, with_c):
            st = SG.SignalStats.fit([R[j] for j in tr])
            S = SG.signal_matrix(R, st)
            return np.hstack([Xc, S]) if with_c else S

        def ac_X(tr, which, with_c):
            fill = NanFill().fit(which[tr])
            M = fill.transform(which)
            return np.hstack([Xc, M]) if with_c else M

        builders = {
            "ctx_LR": lambda tr: Xc,
            "O2_S": lambda tr: sig_X(tr, False),
            "O2_C+S": lambda tr: sig_X(tr, True),
            "O3_AC_asr": lambda tr: ac_X(tr, Xac_a, False),
            "O3_AC_vad": lambda tr: ac_X(tr, Xac_v, False),
            "O3_C+AC_asr": lambda tr: ac_X(tr, Xac_a, True),
            "O3_C+AC_vad": lambda tr: ac_X(tr, Xac_v, True),
        }
        decs = {"rule_B": rule_B}
        scores = {}
        pf = {}
        for name, mk in builders.items():
            o = nested_oof_fold_aware(mk, y, g, fo)
            decs[name] = o["decision"]
            scores[name] = o["score"]
            pf[name] = [
                {k: v for k, v in d.items() if k != "sel"} for d in o["per_fold"]
            ]
        parity = int((decs["ctx_LR"] != P_ctx).sum())
        parity_rule = int((rule_B != P_rule).sum())
        F1 = {k: wconfusion(y, v)["f1"] for k, v in decs.items()}
        F1["ctx_LR_A"] = wconfusion(y, A_ctx)["f1"]
        F1["rule_A"] = wconfusion(y, A_rule)["f1"]
        tr_ = {
            "task": task,
            "n_units": int(len(idx)),
            "pos": int(y.sum()),
            "pos_speakers": int(len(set(g[y == 1]))),
            "visible_pos": int((vis).sum()),
            "destroyed_pos": int(destroyed.sum()),
            "parity_mismatch_ctx_LR_B_vs_parent": parity,
            "parity_mismatch_rule_B_vs_parent": parity_rule,
            "f1": F1,
            "models": {},
        }
        gain_den = F1["ctx_LR_A"] - F1["ctx_LR"]
        for name, d in decs.items():
            m = wconfusion(y, d)
            m["f1_ci95"] = boot_wf1_ci(y, d, g, None, 2000, SEED)
            m["recall_on_destroyed"] = (
                float(d[destroyed].mean()) if destroyed.any() else float("nan")
            )
            m["recall_on_visible"] = (
                float(d[(y == 1) & vis].mean())
                if ((y == 1) & vis).any()
                else float("nan")
            )
            m["fp_vs_ctx_LR_B"] = {
                "new_fp": int(((d == 1) & (y == 0) & (decs["ctx_LR"] == 0)).sum()),
                "removed_fp": int(((d == 0) & (y == 0) & (decs["ctx_LR"] == 1)).sum()),
                "recovered_tp_new": int(
                    ((d == 1) & (y == 1) & (decs["ctx_LR"] == 0)).sum()
                ),
                "lost_tp": int(((d == 0) & (y == 1) & (decs["ctx_LR"] == 1)).sum()),
            }
            if name not in ("ctx_LR",):
                m["paired_vs_ctx_LR_B"] = paired_wf1_delta(
                    y, d, decs["ctx_LR"], g, None, 5000, SEED
                )
                m["paired_vs_rule_B"] = paired_wf1_delta(
                    y, d, rule_B, g, None, 5000, SEED
                )
                m["recovered_share_of_A_minus_B_ctx_LR"] = (
                    (F1[name] - F1["ctx_LR"]) / gain_den if gain_den > 0.02 else None
                )
            if name in pf:
                m["per_fold"] = pf[name]
            tr_["models"][name] = m
            for j, u in enumerate(uid):
                rows.append(
                    {
                        "task": task,
                        "model": name,
                        "unit_id": u,
                        "speaker": g[j],
                        "outer_fold": int(fo[j]),
                        "y": int(y[j]),
                        "destroyed_by_asr": int(destroyed[j]),
                        "score": "" if name not in scores else f"{scores[name][j]:.6f}",
                        "decision": int(d[j]),
                    }
                )
        tr_["paired_ctx_A_minus_B"] = paired_wf1_delta(
            y, A_ctx, decs["ctx_LR"], g, None, 2000, SEED
        )
        res["tasks"][task] = tr_
    wav_h = hashlib.sha256("".join(sha(WAV / f"{u['unit_id']}.wav") for u in pool).encode()).hexdigest()
    res["config"] = run_config(
        {
            "acoustic.pkl": sha(OUT / "features" / "acoustic.pkl"),
            "parent_predictions.csv": sha(PARENT / "run_main" / "predictions.csv"),
            "wav16k_all_pool_units": wav_h,
            "asr_cache_pool_units": hashlib.sha256("".join(sha(ASR / f"{u['unit_id']}.json") for u in pool).encode()).hexdigest(),
        },
        {
            "n_signal_features": len(SG.NAMES),
            "n_acoustic_features": [len(AC.NAMES_ASR), len(AC.NAMES_VAD)],
            "seconds": round(time.time() - t0, 1),
        },
    )
    run = OUT / "run_o2_o3"
    run.mkdir(exist_ok=True)
    (run / "config.json").write_text(json.dumps(res["config"], indent=1), "utf8")
    (run / "results.json").write_text(json.dumps(res, indent=1, default=float), "utf8")
    with (run / "predictions.csv").open("w", newline="", encoding="utf8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(
        json.dumps(
            {
                t: {
                    "F1": res["tasks"][t]["f1"],
                    "parity": res["tasks"][t]["parity_mismatch_ctx_LR_B_vs_parent"],
                }
                for t in TASKS
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
