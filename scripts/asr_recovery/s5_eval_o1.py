"""O1 evaluation: P0 (control, re-decoded or cache per timing.json) vs P1 / P2 on the frozen subset. Usage: python s5_eval_o1.py"""

import _pin  # noqa: F401  (MUST be first)
import csv
import hashlib
import json
import platform
import time

import numpy as np

from mrcd.asr_recovery.common import (
    ASR,
    LITERAL_SUBTYPES,
    OUT,
    PARENT,
    SEED,
    run_config,
    TASKS,
    load_pool,
    outer_folds,
    sha,
    task_units,
)
from mrcd.asr_recovery.cvfold import nested_oof_fold_aware
from mrcd.asr_recovery.decode import P1_PROMPT, VARIANTS, words_text
from mrcd.asr_recovery.stats import (
    boot_wf1_ci,
    corpus_wer,
    paired_rate_delta,
    paired_wf1_delta,
    prompt_echo,
    rate_ci,
    wconfusion,
)
from mrcd.context_eval.survival import turn_survives
from mrcd.context_eval.textfeat import featurize, rule_rep, rule_rev

VARS = ("P0", "P1", "P2")


def load_variant(v, ids):
    out = {}
    for u in ids:
        if v == "P0cache":
            j = json.loads((ASR / f"{u}.json").read_text("utf8"))
        else:
            j = json.loads((OUT / "asr" / v / f"{u}.json").read_text("utf8"))
        out[u] = j
    return out


def main():
    t0 = time.time()
    pool = load_pool()
    outer = outer_folds(pool)
    by_id = {u["unit_id"]: i for i, u in enumerate(pool)}
    frame = json.loads((OUT / "subset.json").read_text("utf8"))["frame"]
    timing = json.loads((OUT / "timing.json").read_text("utf8"))
    ids = [s["unit_id"] for s in frame]
    W = {s["unit_id"]: s["w"] for s in frame}
    KIND = {s["unit_id"]: s["kind"] for s in frame}
    dec_words = {v: load_variant(v, ids) for v in (*VARS, "P0cache")}
    parent = {}
    with (PARENT / "run_main" / "predictions.csv").open(encoding="utf8") as f:
        for r in csv.DictReader(f):
            if r["task"] in TASKS and r["model"] in ("rule", "ctx_LR"):
                parent[(r["task"], r["condition"], r["model"], r["unit_id"])] = int(
                    r["decision"]
                )
    res = {"tasks": {}}
    rows = []
    for task in TASKS:
        idx_all, y_all = task_units(pool, task)
        y_of = {pool[i]["unit_id"]: int(v) for i, v in zip(idx_all, y_all)}
        fo_all = outer[idx_all]
        g_all = np.array([pool[i]["speaker"] for i in idx_all])
        Xc = featurize([pool[i]["text_B"] for i in idx_all], task)
        nest = nested_oof_fold_aware(lambda tr: Xc, y_all, g_all, fo_all)
        sel = {d["fold"]: d["sel"] for d in nest["per_fold"]}
        rule_fn = rule_rev if task == "revision" else rule_rep
        fold_of = {pool[i]["unit_id"]: int(outer[i]) for i in range(len(pool))}
        units = [u for u in ids if u in y_of]
        y = np.array([y_of[u] for u in units])
        w = np.array([W[u] for u in units])
        g = np.array([pool[by_id[u]]["speaker"] for u in units])
        neg = np.array([KIND[u] == "neg" for u in units])
        kind, key = (
            ("revision", "rev_marks")
            if task == "revision"
            else ("repetition", "rep_marks")
        )
        subs = LITERAL_SUBTYPES if task == "rep_literal" else None
        tr_ = {
            "n_units": len(units),
            "pos": int(y.sum()),
            "pos_speakers": int(len(set(g[y == 1]))),
            "neg_sampled": int(neg.sum()),
            "p0_control": timing["p0_control"],
            "variants": {},
        }
        A = {
            m: np.array([parent[(task, "A", m, u)] for u in units])
            for m in ("rule", "ctx_LR")
        }
        texts, decs, surv = {}, {}, {}
        for v in (*VARS, "P0cache"):
            tx = [words_text(dec_words[v][u]["words"]) for u in units]
            texts[v] = tx
            X = featurize(tx, task)
            sc = np.zeros(len(units))
            dc = np.zeros(len(units), int)
            for j, u in enumerate(units):
                f = fold_of[u]
                sc[j] = float(sel[f]["model"].score(X[j : j + 1])[0])
                dc[j] = int(sc[j] >= sel[f]["threshold"])
            decs[v] = {
                "rule": np.array([rule_fn(t) for t in tx]),
                "ctx_LR": dc,
                "score": sc,
            }
            surv[v] = np.array(
                [
                    bool(y[j])
                    and turn_survives(kind, pool[by_id[u]]["ref"][key], tx[j], subs)
                    for j, u in enumerate(units)
                ]
            )
        # sanity: P0cache decisions of the refitted models equal the parent OOF decisions
        pos_in_task = {u: j for j, u in enumerate(pool[i]["unit_id"] for i in idx_all)}
        tr_["sanity_P0cache_ctx_LR_vs_oof_mismatch"] = int(
            sum(
                decs["P0cache"]["ctx_LR"][j] != nest["decision"][pos_in_task[u]]
                for j, u in enumerate(units)
            )
        )
        tr_["sanity_P0cache_ctx_LR_vs_parent_mismatch"] = int(
            sum(
                decs["P0cache"]["ctx_LR"][j] != parent[(task, "B", "ctx_LR", u)]
                for j, u in enumerate(units)
            )
        )
        posm = y == 1
        ref_text = [pool[by_id[u]]["text_A"] for u in units]
        for v in (*VARS, "P0cache"):
            d = {"survival": rate_ci(surv[v][posm].astype(float), g[posm], 2000, SEED)}
            if v != "P0":
                d["survival_paired_vs_P0"] = paired_rate_delta(
                    surv[v][posm], surv["P0"][posm], g[posm]
                )
            d["survival_on_P0_destroyed"] = {
                "n": int((posm & ~surv["P0"]).sum()),
                "recovered": int((posm & ~surv["P0"] & surv[v]).sum()),
            }
            d["models"] = {}
            for m in ("rule", "ctx_LR"):
                dd = decs[v][m]
                met = wconfusion(y, dd, w)
                met["f1w_ci95"] = boot_wf1_ci(y, dd, g, w, 2000, SEED)
                met["fp_on_sampled_negatives"] = {
                    "n_neg": int(neg.sum()),
                    "fp": int(dd[neg].sum()),
                    "rate": float(dd[neg].mean()) if neg.any() else float("nan"),
                }
                dP0 = decs["P0"][m]
                met["recall_on_P0_destroyed_pos"] = (
                    float(dd[posm & ~surv["P0"]].mean())
                    if (posm & ~surv["P0"]).any()
                    else float("nan")
                )
                met["recall_on_P0_visible_pos"] = (
                    float(dd[posm & surv["P0"]].mean())
                    if (posm & surv["P0"]).any()
                    else float("nan")
                )
                if v != "P0":
                    met["paired_f1w_vs_P0"] = paired_wf1_delta(
                        y, dd, dP0, g, w, 5000, SEED
                    )
                    met["changes_vs_P0"] = {
                        "new_fp": int(((dd == 1) & (y == 0) & (dP0 == 0)).sum()),
                        "removed_fp": int(((dd == 0) & (y == 0) & (dP0 == 1)).sum()),
                        "new_tp": int(((dd == 1) & (y == 1) & (dP0 == 0)).sum()),
                        "lost_tp": int(((dd == 0) & (y == 1) & (dP0 == 1)).sum()),
                    }
                    f_a = wconfusion(y, A[m], w)["w_f1"]
                    f_0 = wconfusion(y, dP0, w)["w_f1"]
                    den = f_a - f_0
                    met["recovered_share_of_A_minus_P0"] = (
                        (met["w_f1"] - f_0) / den if den > 0.02 else None
                    )
                d["models"][m] = met
            # WER / damage
            if v != "P0":
                d["wer_P0_vs_variant_on_neg"] = corpus_wer(
                    [(texts["P0"][j], texts[v][j]) for j in np.flatnonzero(neg)]
                )
            d["wer_vs_reference_text_A_on_neg"] = corpus_wer(
                [(ref_text[j], texts[v][j]) for j in np.flatnonzero(neg)]
            )
            d["wer_vs_reference_text_A_all"] = corpus_wer(
                [(ref_text[j], texts[v][j]) for j in range(len(units))]
            )
            d["empty_outputs"] = int(sum(len(t.strip()) == 0 for t in texts[v]))
            d["prompt_echo"] = int(sum(prompt_echo(t, P1_PROMPT) for t in texts[v]))
            d["mean_words"] = float(
                np.mean([len(dec_words[v][u]["words"]) for u in units])
            )
            tr_["variants"][v] = d
        tr_["A"] = {m: wconfusion(y, A[m], w) for m in A}
        res["tasks"][task] = tr_
        for v in (*VARS, "P0cache"):
            for m in ("rule", "ctx_LR"):
                for j, u in enumerate(units):
                    rows.append(
                        {
                            "task": task,
                            "variant": v,
                            "model": m,
                            "unit_id": u,
                            "speaker": g[j],
                            "kind": KIND[u],
                            "w": f"{w[j]:.4f}",
                            "y": int(y[j]),
                            "survives": int(surv[v][j]),
                            "decision": int(decs[v][m][j]),
                            "score": f"{decs[v]['score'][j]:.6f}"
                            if m == "ctx_LR"
                            else "",
                            "text": texts[v][j],
                        }
                    )
    res["config"] = run_config(
        {
            "subset.json": sha(OUT / "subset.json"),
            "timing.json": sha(OUT / "timing.json"),
            "parent_predictions.csv": sha(PARENT / "run_main" / "predictions.csv"),
            **{
                f"asr_{v}": hashlib.sha256("".join(sorted(sha(OUT / "asr" / v / f"{u}.json") for u in ids)).encode()).hexdigest()
                for v in VARS
            },
        },
        {
            "variants": {k: {kk: (list(vv) if isinstance(vv, tuple) else vv) for kk, vv in v.items()} for k, v in VARIANTS.items()},
            "seconds": round(time.time() - t0, 1),
        },
    )
    run = OUT / "run_o1"
    run.mkdir(exist_ok=True)
    (run / "config.json").write_text(json.dumps(res["config"], indent=1), "utf8")
    (run / "results.json").write_text(json.dumps(res, indent=1, default=float), "utf8")
    with (run / "predictions.csv").open("w", newline="", encoding="utf8") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    print("done", res["config"]["seconds"])


if __name__ == "__main__":
    main()
