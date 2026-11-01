"""Main experiment (EXPLORATORY): contextual repetition / revision on DIHANA folds 1-4, nested speaker-grouped CV.

Protocol: results/ejecucion_material4_20261030/experimentos/PROTOCOL.md (frozen; sha256 in PROTOCOL.sha256.json).
Fold 5 is never loaded. Usage: python scripts/rep_rev/run_experiment.py
"""

import os

os.environ["OMP_NUM_THREADS"] = "4"  # before numpy / torch (project rule)

import csv  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import pickle  # noqa: E402
import platform  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

REPO = Path(r"C:\zArnes-tesis\mrcd")
sys.path.insert(0, str(REPO))
import numpy as np  # noqa: E402

from mrcd.constants import LABELS  # noqa: E402
from mrcd.context_eval import metrics as M  # noqa: E402
from mrcd.context_eval.model import CS, nested_oof  # noqa: E402
from mrcd.context_eval.partition import assert_no_crossing, group_folds  # noqa: E402
from mrcd.context_eval.pool import assert_no_historical, build_pool  # noqa: E402
from mrcd.context_eval.survival import turn_survives  # noqa: E402
from mrcd.context_eval.textfeat import featurize, rule_rep, rule_rev  # noqa: E402
from mrcd.decision import any_window_argmax  # noqa: E402

SEED = 13
EXP = REPO / "results" / "ejecucion_material4_20261030" / "experimentos"
RUN = EXP / "run_main"
CORPUS = Path(
    (os.environ.get("MRCD_CORPORA_ROOT", r"<CORPORA_ROOT>") + r"\01_DIHANA\extraido\dihana")
)
TURNS = REPO / "results/baselines_20261030/dihana/turns.csv"
ASR = REPO / "results/mejoras_20261030/asr_cache/dihana"
FEAT = REPO / "results/mejoras_20261030/features"
HEAD = REPO / "results/mejoras_20261030/frozen_head_zero.json"
GATE_POS, GATE_SPK = 30, 15
TASKS = ("rep_literal", "rep_full", "revision")
FAMILY = [
    ("rep_literal", "A"),
    ("rep_literal", "B"),
    ("revision", "A"),
    ("revision", "B"),
]


def sha(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_pool():
    cache = EXP / "_cache" / "pool.pkl"
    if cache.exists():
        return pickle.loads(cache.read_bytes())
    pool = build_pool(CORPUS, TURNS, ASR)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_bytes(pickle.dumps(pool))
    return pool


def v0_vc(pool):
    """V0 for every unit (folds 1-4); Vc for every unit but flagged fair only on fold 4. Frozen caches, nothing is fitted."""
    from mrcd import adaptation as A

    feats = {}
    for p in sorted(FEAT.glob("dihana_zero_shard*.pkl")):
        feats.update(pickle.loads(p.read_bytes()))
    info = json.loads(HEAD.read_text("utf8"))
    ck, _ = A.load_adapted(
        info["checkpoint"], expected_sha256=info["checkpoint_sha256"]
    )
    classes = list(ck["classes"])
    out = {}
    for u in pool:
        rec = feats[u["unit_id"]]
        s = A.head_scores(
            ck["state"], rec["pooled"], len(classes), ck["head_kind"], ck["hidden"]
        )
        thr = info["thresholds"]["repetition"]["threshold"]
        out[u["unit_id"]] = {
            "V0_rep": any_window_argmax(rec["probs"], list(LABELS), "repetition"),
            "V0_rev": any_window_argmax(rec["probs"], list(LABELS), "revision"),
            "Vc_rep": int(s[:, classes.index("repetition")].max() >= thr),
            "Vc_score": float(s[:, classes.index("repetition")].max()),
        }
    return out, info


def task_units(pool, task):
    idx, y = [], []
    for i, u in enumerate(pool):
        v = u["ref"][task]
        if v is None:
            continue
        idx.append(i)
        y.append(int(v))
    return np.array(idx), np.array(y)


def descr(y, p, g, n_boot=2000):
    m = M.confusion(y, p)
    m["n_groups"] = int(len(set(g)))
    m["n_pos_groups"] = int(len({gg for gg, yy in zip(g, y) if yy}))
    m["f1_ci95"] = M.boot_f1_ci(y, p, g, n_boot, SEED)
    return m


def main():
    t0 = time.time()
    RUN.mkdir(parents=True, exist_ok=True)
    pool = load_pool()
    assert_no_historical(u["fold"] for u in pool)
    spk = [u["speaker"] for u in pool]
    outer = group_folds(spk, 5, SEED)
    assert_no_crossing(spk, outer, "speaker")
    assert_no_crossing([u["dialogue"] for u in pool], outer, "dialogue")
    base, head_info = v0_vc(pool)

    pred_rows, results = [], {"tasks": {}}
    store = {}  # (task, cond, model) -> dict(idx,y,score,dec)
    for task in TASKS:
        idx, y = task_units(pool, task)
        g = [pool[i]["speaker"] for i in idx]
        npos, nspk = int(y.sum()), len({gg for gg, yy in zip(g, y) if yy})
        gate = npos >= GATE_POS and nspk >= GATE_SPK
        res = {
            "n_units": int(len(idx)),
            "pos": npos,
            "pos_speakers": nspk,
            "gate_met": bool(gate),
            "cond": {},
        }
        feat_task = "revision" if task == "revision" else task
        for cond in ("A", "B"):
            texts = [pool[i]["text_" + cond] for i in idx]
            rule_fn = rule_rev if task == "revision" else rule_rep
            models = {"rule": (None, np.array([rule_fn(t) for t in texts]))}
            if cond == "B":
                key = "V0_rev" if task == "revision" else "V0_rep"
                models["V0"] = (
                    None,
                    np.array([base[pool[i]["unit_id"]][key] for i in idx]),
                )
                if task != "revision":
                    models["Vc"] = (
                        np.array([base[pool[i]["unit_id"]]["Vc_score"] for i in idx]),
                        np.array([base[pool[i]["unit_id"]]["Vc_rep"] for i in idx]),
                    )
            X = featurize(texts, feat_task)
            if gate:
                nest = nested_oof(X, y, g, outer[idx], calibrate=True)
                models["ctx_LR"] = (nest["score"], nest["decision"])
                res.setdefault("per_fold", {})[cond] = nest["per_fold"]
                store[(task, cond, "ctx_LR_cal")] = nest["calibrated"]
            cres = {}
            for name, (score, dec) in models.items():
                store[(task, cond, name)] = (idx, y, score, dec)
                row = descr(y, dec, g)
                if name == "Vc":
                    fair = np.array([pool[i]["fold"] == 4 for i in idx])
                    row = {"all_units_in_sample_for_folds_1_3": row}
                    row["fold4_only_vc_favourable"] = descr(
                        y[fair], dec[fair], np.array(g)[fair]
                    )
                    for mname in ("rule", "V0", "ctx_LR"):
                        if mname in models:
                            row["fold4_only_" + mname] = descr(
                                y[fair], models[mname][1][fair], np.array(g)[fair]
                            )
                            if mname != "Vc":
                                row["fold4_paired_vc_minus_" + mname] = (
                                    M.paired_f1_delta(
                                        y[fair],
                                        dec[fair],
                                        models[mname][1][fair],
                                        np.array(g)[fair],
                                        5000,
                                        SEED,
                                    )
                                )
                cres[name] = row
                for j, i in enumerate(idx):
                    pred_rows.append(
                        {
                            "task": task,
                            "condition": cond,
                            "model": name,
                            "unit_id": pool[i]["unit_id"],
                            "speaker": pool[i]["speaker"],
                            "fold_official": pool[i]["fold"],
                            "outer_fold": int(outer[i]),
                            "y": int(y[j]),
                            "score": "" if score is None else f"{score[j]:.6f}",
                            "decision": int(dec[j]),
                            "calibrated": ""
                            if name != "ctx_LR" or not gate
                            else f"{store[(task, cond, 'ctx_LR_cal')][j]:.6f}",
                            "vc_fair": int(pool[i]["fold"] == 4)
                            if name == "Vc"
                            else "",
                        }
                    )
            # paired contrasts, descriptive except the pre-declared family
            if gate:
                gg = np.array(g)
                cres["paired_ctx_minus_rule"] = M.paired_f1_delta(
                    y, models["ctx_LR"][1], models["rule"][1], gg, 5000, SEED
                )
                if "V0" in models:
                    cres["paired_ctx_minus_V0_descriptive"] = M.paired_f1_delta(
                        y, models["ctx_LR"][1], models["V0"][1], gg, 5000, SEED
                    )
                    cres["paired_rule_minus_V0_descriptive"] = M.paired_f1_delta(
                        y, models["rule"][1], models["V0"][1], gg, 5000, SEED
                    )
                # per official fold (descriptive)
                of = np.array([pool[i]["fold"] for i in idx])
                cres["per_official_fold"] = {
                    str(f): {
                        n: M.confusion(y[of == f], d[of == f])
                        for n, (_, d) in models.items()
                        if n != "Vc"
                    }
                    for f in (1, 2, 3, 4)
                }
                # calibration (cross-fitted Platt vs raw)
                s_raw, s_cal = models["ctx_LR"][0], store[(task, cond, "ctx_LR_cal")]
                gg = np.array(g)
                cres["calibration"] = {
                    "bins": 10,
                    "brier_raw": M.brier(y, s_raw),
                    "brier_cal": M.brier(y, s_cal),
                    "ece_raw": M.ece(y, s_raw),
                    "ece_cal": M.ece(y, s_cal),
                    "brier_raw_minus_cal": M.boot_diff(
                        M.brier, y, s_raw, s_cal, gg, 1000, SEED
                    ),
                    "ece_raw_minus_cal": M.boot_diff(
                        M.ece, y, s_raw, s_cal, gg, 1000, SEED
                    ),
                    "reliability_raw": M.reliability(y, s_raw),
                    "reliability_cal": M.reliability(y, s_cal),
                    "prevalence": float(y.mean()),
                    "note": "raw LR score, calibrator = Platt on inner-OOF scores of the outer train (cross-fitted)",
                }
            res["cond"][cond] = cres
        results["tasks"][task] = res

    # pre-declared family with Holm
    ps = [
        results["tasks"][t]["cond"][c]
        .get("paired_ctx_minus_rule", {})
        .get("p_boot", float("nan"))
        for t, c in FAMILY
    ]
    adj = M.holm(ps)
    results["family_holm"] = [
        {
            "task": t,
            "condition": c,
            "paired_ctx_minus_rule": results["tasks"][t]["cond"][c].get(
                "paired_ctx_minus_rule"
            ),
            "p_boot": p,
            "p_holm": a,
            "verdict": (
                "supported after Holm"
                if (
                    a == a
                    and a < 0.05
                    and (
                        results["tasks"][t]["cond"][c]["paired_ctx_minus_rule"]["ci"][0]
                        > 0
                        or results["tasks"][t]["cond"][c]["paired_ctx_minus_rule"][
                            "ci"
                        ][1]
                        < 0
                    )
                )
                else "not demonstrated"
            ),
        }
        for (t, c), p, a in zip(FAMILY, ps, adj)
    ]

    # error decomposition A vs B
    dec = {}
    for task in TASKS:
        if not results["tasks"][task]["gate_met"]:
            continue
        idx, y = task_units(pool, task)
        kind = "revision" if task == "revision" else "repetition"
        key = "rev_marks" if task == "revision" else "rep_marks"
        subs = {"A_literal_repeat"} if task == "rep_literal" else None
        vis = np.array(
            [
                bool(y[j])
                and turn_survives(kind, pool[i]["ref"][key], pool[i]["text_B"], subs)
                for j, i in enumerate(idx)
            ]
        )
        destroyed = (y == 1) & ~vis
        red = ~destroyed
        g = np.array([pool[i]["speaker"] for i in idx])
        d = {
            "positives": int(y.sum()),
            "visible_in_asr": int(vis.sum()),
            "destroyed_by_asr": int(destroyed.sum()),
            "models": {},
        }
        for m in ("rule", "ctx_LR"):
            pa, pb = store[(task, "A", m)][3], store[(task, "B", m)][3]
            fa, fb = M.confusion(y, pa)["f1"], M.confusion(y, pb)["f1"]
            fa_r, fb_r = (
                M.confusion(y[red], pa[red])["f1"],
                M.confusion(y[red], pb[red])["f1"],
            )
            tot = fa - fb
            within = fa_r - fb_r
            d["models"][m] = {
                "F1_A": fa,
                "F1_B": fb,
                "F1_A_on_reduced": fa_r,
                "F1_B_on_reduced": fb_r,
                "total_loss_A_minus_B": tot,
                "text_difference_on_reduced_units": within,
                "residual_attributed_to_destroyed_positives": tot - within,
                "recall_B_on_visible": float(pb[(y == 1) & vis].mean())
                if ((y == 1) & vis).any()
                else float("nan"),
                "recall_B_on_destroyed": float(pb[destroyed].mean())
                if destroyed.any()
                else float("nan"),
                "recall_A_on_visible": float(pa[(y == 1) & vis].mean())
                if ((y == 1) & vis).any()
                else float("nan"),
                "recall_A_on_destroyed": float(pa[destroyed].mean())
                if destroyed.any()
                else float("nan"),
                "flip_table_AxB_by_y": {
                    f"y{yy}": {
                        f"A{a}B{b}": int(((y == yy) & (pa == a) & (pb == b)).sum())
                        for a in (0, 1)
                        for b in (0, 1)
                    }
                    for yy in (0, 1)
                },
                "paired_F1_A_minus_B_ci": M.paired_f1_delta(y, pa, pb, g, 2000, SEED),
            }
        dec[task] = d
    results["decomposition"] = dec

    # FP / FN examples (deterministic hash order, no labels created)
    ex = {}
    for task in TASKS:
        if not results["tasks"][task]["gate_met"]:
            continue
        for m in ("rule", "ctx_LR"):
            _, y, _, d_ = store[(task, "B", m)]
            idx = store[(task, "B", m)][0]
            for kind, sel in (
                ("FP", (d_ == 1) & (y == 0)),
                ("FN", (d_ == 0) & (y == 1)),
            ):
                cand = [int(i) for i in idx[sel]]
                cand.sort(
                    key=lambda i: hashlib.sha256(
                        f"{SEED}|{task}|{m}|{kind}|{pool[i]['unit_id']}".encode()
                    ).hexdigest()
                )
                ex[f"{task}|{m}|{kind}"] = {
                    "total": len(cand),
                    "examples": [
                        {
                            "unit": pool[i]["unit_id"],
                            "text_B_asr": pool[i]["text_B"],
                            "text_A_reference": pool[i]["text_A"],
                            "marks": [
                                (mk["subtype"], mk["span"])
                                for mk in pool[i]["ref"]["rep_marks"]
                                + pool[i]["ref"]["rev_marks"]
                            ],
                        }
                        for i in cand[:8]
                    ],
                }
    results["error_examples"] = ex

    # breakdown of recall by sub-type of the human mark (descriptive)
    sub = {}
    for task, key, kindkey in (
        ("rep_full", "rep_subtypes", "rep"),
        ("revision", "rev_subtypes", "rev"),
    ):
        if not results["tasks"][task]["gate_met"]:
            continue
        idx, y = task_units(pool, task)
        for m in ("rule", "ctx_LR", "V0"):
            for cond in ("A", "B"):
                if (task, cond, m) not in store:
                    continue
                dd = store[(task, cond, m)][3]
                tab = {}
                for st in sorted({s for i in idx for s in pool[i]["ref"][key]}):
                    sel = np.array([st in pool[i]["ref"][key] for i in idx])
                    tab[st] = {"turns": int(sel.sum()), "detected": int(dd[sel].sum())}
                sub[f"{task}|{cond}|{m}"] = tab
    results["subtype_recall"] = sub

    # reproducibility
    import sklearn
    import scipy
    import torch

    results["config"] = {
        "seed": SEED,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "versions": {
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "sklearn": sklearn.__version__,
            "torch": torch.__version__,
        },
        "pool": {
            "turns": len(pool),
            "speakers": len(set(spk)),
            "dialogues": len({u["dialogue"] for u in pool}),
            "folds": sorted({u["fold"] for u in pool}),
            "outer_k": 5,
            "inner_k": 3,
            "Cs": list(CS),
            "speaker_crossings": 0,
            "dialogue_crossings": 0,
        },
        "protocol_sha256": json.loads((EXP / "PROTOCOL.sha256.json").read_text())[
            "sha256"
        ],
        "inputs_sha256": {
            "turns.csv": sha(TURNS),
            "frozen_head_zero.json": sha(HEAD),
            "head_checkpoint": head_info["checkpoint_sha256"],
            **{p.name: sha(p) for p in sorted(FEAT.glob("dihana_zero_shard*.pkl"))},
            "asr_cache_manifest": sha(ASR / "manifest.json")
            if (ASR / "manifest.json").exists()
            else None,
        },
        "code_sha256": {
            str(p.relative_to(REPO)): sha(p)
            for p in sorted((REPO / "mrcd" / "context_eval").glob("*.py"))
        }
        | {"scripts/rep_rev/run_experiment.py": sha(__file__)},
        "seconds": round(time.time() - t0, 1),
    }
    (RUN / "results.json").write_text(
        json.dumps(results, indent=1, default=float), encoding="utf8"
    )
    (RUN / "config.json").write_text(
        json.dumps(results["config"], indent=1), encoding="utf8"
    )
    with (RUN / "predictions.csv").open("w", newline="", encoding="utf8") as f:
        w = csv.DictWriter(f, fieldnames=list(pred_rows[0]))
        w.writeheader()
        w.writerows(pred_rows)
    print(
        json.dumps(
            {
                "seconds": results["config"]["seconds"],
                "gates": {t: results["tasks"][t]["gate_met"] for t in TASKS},
            }
        )
    )


if __name__ == "__main__":
    main()
