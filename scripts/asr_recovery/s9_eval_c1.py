"""Annex 3 evaluation: P0 vs C1 (fluent control prompt) vs P1 on the frozen subset. Usage: python s9_eval_c1.py"""

import _pin  # noqa: F401  (MUST be first)
import csv
import hashlib
import json
import time

import numpy as np

from mrcd.asr_recovery.common import (
    LITERAL_SUBTYPES,
    OUT,
    PARENT,
    SEED,
    TASKS,
    load_pool,
    outer_folds,
    run_config,
    sha,
    task_units,
)
from mrcd.asr_recovery.cvfold import nested_oof_fold_aware
from mrcd.asr_recovery.decode import C1_PROMPT, P1_PROMPT, words_text
from mrcd.asr_recovery.stats import (
    _f1,
    _gcounts,
    boot_wf1_ci,
    corpus_wer,
    paired_rate_delta,
    paired_wf1_delta,
    prompt_echo,
    rate_ci,
    wconfusion,
)
from mrcd.context_eval.metrics import holm
from mrcd.context_eval.reference import _norm_tokens
from mrcd.context_eval.survival import turn_survives
from mrcd.context_eval.textfeat import featurize, rule_rep, rule_rev

VARS = ("P0", "C1", "P1")


def adjacent_pairs(text):
    t = _norm_tokens(text)
    return sum(a == b for a, b in zip(t, t[1:]))


def share_ci(y, dC, dP, d0, g, w, n_boot=2000, seed=SEED):
    """(F1w(C1)-F1w(P0)) / (F1w(P1)-F1w(P0)) with speaker bootstrap (denominator > 0.02 only)."""
    CC, CP, C0 = (_gcounts(y, d, g, w) for d in (dC, dP, d0))

    def f(c):
        return float(_f1(c.sum(0)))

    den = f(CP) - f(C0)
    pt = (f(CC) - f(C0)) / den if den > 0.02 else None
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(CC), (n_boot, len(CC)))
    num = _f1(CC[idx].sum(1)) - _f1(C0[idx].sum(1))
    dd = _f1(CP[idx].sum(1)) - _f1(C0[idx].sum(1))
    ok = np.isfinite(num) & np.isfinite(dd) & (dd > 0.02)
    ci = (
        [
            float(np.percentile((num / dd)[ok], 2.5)),
            float(np.percentile((num / dd)[ok], 97.5)),
        ]
        if ok.sum() > 100
        else None
    )
    return {"share": pt, "ci95": ci, "n_boot_valid": int(ok.sum())}


def main():
    t0 = time.time()
    pool = load_pool()
    outer = outer_folds(pool)
    by_id = {u["unit_id"]: i for i, u in enumerate(pool)}
    frame = json.loads((OUT / "subset.json").read_text("utf8"))["frame"]
    ids = [s["unit_id"] for s in frame]
    W = {s["unit_id"]: s["w"] for s in frame}
    KIND = {s["unit_id"]: s["kind"] for s in frame}
    dec = {
        v: {
            u: json.loads((OUT / "asr" / v / f"{u}.json").read_text("utf8"))
            for u in ids
        }
        for v in VARS
    }
    res = {"tasks": {}, "family": []}
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
        posm = y == 1
        kind, key = (
            ("revision", "rev_marks")
            if task == "revision"
            else ("repetition", "rep_marks")
        )
        subs = LITERAL_SUBTYPES if task == "rep_literal" else None
        ref_text = [pool[by_id[u]]["text_A"] for u in units]
        texts, decs, surv = {}, {}, {}
        for v in VARS:
            tx = [words_text(dec[v][u]["words"]) for u in units]
            texts[v] = tx
            X = featurize(tx, task)
            dc = np.zeros(len(units), int)
            for j, u in enumerate(units):
                f = fold_of[u]
                dc[j] = int(
                    float(sel[f]["model"].score(X[j : j + 1])[0]) >= sel[f]["threshold"]
                )
            decs[v] = {"rule": np.array([rule_fn(t) for t in tx]), "ctx_LR": dc}
            surv[v] = np.array(
                [
                    bool(y[j])
                    and turn_survives(kind, pool[by_id[u]]["ref"][key], tx[j], subs)
                    for j, u in enumerate(units)
                ]
            )
        tr_ = {
            "n_units": len(units),
            "pos": int(y.sum()),
            "neg_sampled": int(neg.sum()),
            "variants": {},
            "contrasts": {},
        }
        for v in VARS:
            d = {
                "survival": rate_ci(surv[v][posm].astype(float), g[posm], 2000, SEED),
                "models": {},
            }
            for m in ("rule", "ctx_LR"):
                dd = decs[v][m]
                met = wconfusion(y, dd, w)
                met["f1w_ci95"] = boot_wf1_ci(y, dd, g, w, 2000, SEED)
                met["fp_on_sampled_negatives"] = {
                    "n_neg": int(neg.sum()),
                    "fp": int(dd[neg].sum()),
                }
                d["models"][m] = met
            d["wer_vs_reference_text_A_on_neg"] = corpus_wer(
                [(ref_text[j], texts[v][j]) for j in np.flatnonzero(neg)]
            )
            d["empty_outputs"] = int(sum(len(t.strip()) == 0 for t in texts[v]))
            d["echo_of_P1_prompt"] = int(
                sum(prompt_echo(t, P1_PROMPT) for t in texts[v])
            )
            d["echo_of_C1_prompt"] = int(
                sum(prompt_echo(t, C1_PROMPT) for t in texts[v])
            )
            d["adjacent_identical_pairs_on_sampled_negatives"] = int(
                sum(adjacent_pairs(texts[v][j]) for j in np.flatnonzero(neg))
            )
            d["units_with_adjacent_pair_on_sampled_negatives"] = int(
                sum(adjacent_pairs(texts[v][j]) > 0 for j in np.flatnonzero(neg))
            )
            d["mean_words"] = float(np.mean([len(dec[v][u]["words"]) for u in units]))
            tr_["variants"][v] = d
        for a, b in (("C1", "P0"), ("P1", "C1"), ("P1", "P0")):
            c = {
                "survival_paired": paired_rate_delta(
                    surv[a][posm], surv[b][posm], g[posm]
                ),
                "models": {},
            }
            for m in ("rule", "ctx_LR"):
                da, db = decs[a][m], decs[b][m]
                c["models"][m] = {
                    "paired_f1w": paired_wf1_delta(y, da, db, g, w, 5000, SEED),
                    "new_fp": int(((da == 1) & (y == 0) & (db == 0)).sum()),
                    "removed_fp": int(((da == 0) & (y == 0) & (db == 1)).sum()),
                    "new_tp": int(((da == 1) & (y == 1) & (db == 0)).sum()),
                    "lost_tp": int(((da == 0) & (y == 1) & (db == 1)).sum()),
                }
            tr_["contrasts"][f"{a}-{b}"] = c
        for m in ("rule", "ctx_LR"):
            tr_.setdefault("share_of_P1_gain_reproduced_by_C1", {})[m] = share_ci(
                y, decs["C1"][m], decs["P1"][m], decs["P0"][m], g, w
            )
        res["tasks"][task] = tr_
        for v in VARS:
            for m in ("rule", "ctx_LR"):
                for j, u in enumerate(units):
                    rows.append(
                        dict(
                            task=task,
                            variant=v,
                            model=m,
                            unit_id=u,
                            speaker=g[j],
                            kind=KIND[u],
                            w=f"{w[j]:.4f}",
                            y=int(y[j]),
                            survives=int(surv[v][j]),
                            decision=int(decs[v][m][j]),
                            text=texts[v][j],
                        )
                    )
    fam = []
    for task in TASKS:
        for pair in ("P1-C1", "C1-P0"):
            pr = res["tasks"][task]["contrasts"][pair]["models"]["ctx_LR"]["paired_f1w"]
            fam.append({"task": task, "contrast": pair, **pr})
    adj = holm([f["p_boot"] for f in fam])
    for f, a in zip(fam, adj):
        f["p_holm"] = a
        f["verdict"] = (
            "supported after Holm"
            if a < 0.05 and (f["ci"][0] > 0 or f["ci"][1] < 0)
            else "not demonstrated"
        )
    res["family"] = fam
    res["config"] = run_config(
        {
            "subset.json": sha(OUT / "subset.json"),
            "annex3": sha(OUT / "PROTOCOL_ANNEX_20261031_3_C1_CONTROL.md"),
            "parent_predictions.csv": sha(PARENT / "run_main" / "predictions.csv"),
            **{
                f"asr_{v}": hashlib.sha256(
                    "".join(
                        sorted(sha(OUT / "asr" / v / f"{u}.json") for u in ids)
                    ).encode()
                ).hexdigest()
                for v in VARS
            },
        },
        {"seconds": round(time.time() - t0, 1)},
    )
    run = OUT / "run_c1"
    run.mkdir(exist_ok=True)
    (run / "config.json").write_text(json.dumps(res["config"], indent=1), "utf8")
    (run / "results.json").write_text(json.dumps(res, indent=1, default=float), "utf8")
    with (run / "predictions.csv").open("w", newline="", encoding="utf8") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    for f in fam:
        print(
            f"{f['task']:11s} {f['contrast']:6s} d={f['delta']:+.3f} ci=[{f['ci'][0]:+.3f},{f['ci'][1]:+.3f}] p={f['p_boot']:.4f} holm={f['p_holm']:.4f} {f['verdict']}"
        )
    print("done", res["config"]["seconds"])


if __name__ == "__main__":
    main()
