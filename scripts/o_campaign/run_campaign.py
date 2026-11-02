"""O-campaign runner. Stages: `classes` (nested search + tables + learning curve + final fit per class) and
`package` (promotion rule, versioned artifact, external zero-shot). Refuses to run before README_S8.md exists.

usage: python run_campaign.py classes [--classes filler_word prolongation repetition] [--budget N]
       python run_campaign.py package
"""

import _pin  # noqa: F401  (OMP_NUM_THREADS=4 before numpy)
import argparse
import csv
import json
import pickle
import sys
import time
from collections import Counter

import numpy as np

from common import MODELS, OUT, REPO, code_hashes, gate, git_head, verify_frozen

sys.path.insert(0, str(REPO / "scripts" / "s3"))

from mrcd.optimize import data as DT  # noqa: E402
from mrcd.optimize import evaluate as EV  # noqa: E402
from mrcd.optimize import product_pipeline as PP  # noqa: E402
from mrcd.optimize import search as SE  # noqa: E402
from mrcd.optimize import space as SP  # noqa: E402

S3 = REPO / "results" / "mejora_objetivos_20261101" / "S3"
FRACS = (0.25, 0.5, 0.75)


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return jsonable(o.tolist())
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    return o


def read_pred(path, cols, ids, y, w=None):
    rows = {r["unit_id"]: r for r in csv.DictReader(open(path, encoding="utf8"))}
    out = {c: np.array([int(float(rows[i][c])) for i in ids]) for c in cols}
    assert (np.array([int(rows[i]["y"]) for i in ids]) == y).all(), (
        "label mismatch with the S3 predictions"
    )
    if w is not None:
        assert np.allclose([float(rows[i]["w"]) for i in ids], w, atol=1e-4), (
            "weight mismatch with the S3 predictions"
        )
    return out


def comparators(cls, d):
    """Decisions of the fixed comparators on the identical units / outer partition, plus the Vc fold-4 subset."""
    import s3_common as K

    pool = K.load_pool()
    base, _ = K.v0_vc(pool)
    fold_of = {u["unit_id"]: u["fold"] for u in pool}
    ids = d.ids
    v0 = np.array([base[i][f"v0_{cls}"] for i in ids])
    out = {}
    if cls == "filler_word":
        p = read_pred(
            S3 / "e2" / "predictions_filler.csv",
            ["rule_asr_lexical", "V0_argmax", "V0thr", "F_STACK"],
            ids,
            d.y,
        )
        out = {
            "rule_asr_lexical": p["rule_asr_lexical"],
            "V0_argmax": p["V0_argmax"],
            "V0thr": p["V0thr"],
            "S3_best_F_STACK": p["F_STACK"],
        }
        assert (out["V0_argmax"] == v0).all()
    elif cls == "prolongation":
        import run_e4 as E4

        recs = [
            {"words": r["words"]["p0"], "n_samples": r["n_samples"]}
            for r in d.store.recs
        ]
        s = np.array([base[i]["v0s_prolongation"] for i in ids])
        dec, _, thr_dec, _ = E4.stack_cv(s, recs, d.y, d.g, d.outer)
        out = {"V0_argmax": v0, "V0thr": thr_dec, "S3_best_P_a_stack": dec}
    else:
        p = read_pred(
            S3 / "e1" / "predictions.csv",
            ["rule_P0", "rule_P1", "V0", "A1_ctxLR_P1", "A2_ctxLR_P0P1"],
            ids,
            d.y,
            d.w,
        )
        out = {
            "rule_P0": p["rule_P0"],
            "rule_P1": p["rule_P1"],
            "V0_argmax": p["V0"],
            "S3_ctxLR_P1_A1": p["A1_ctxLR_P1"],
            "S3_best_A2": p["A2_ctxLR_P0P1"],
        }
        assert (out["V0_argmax"] == v0).all()
    f4 = np.array([fold_of[i] == 4 for i in ids])
    vc = np.array([base[i].get(f"vc_{cls}", -1) for i in ids])
    return out, f4, vc


def run_folds(cls, d, cdir, budget):
    res = {}
    for f in sorted(set(d.outer.tolist())):
        p = cdir / f"fold{f}.pkl"
        if p.exists():
            res[f] = pickle.loads(p.read_bytes())
            continue
        t0 = time.time()
        r = SE.outer_fold(d, f, budget=budget)
        p.write_bytes(pickle.dumps(r))
        print(
            cls,
            "fold",
            f,
            r["config"]["key"],
            f"inner={r['inner_f1']:.3f}",
            r["stop"],
            r["n_evaluated"],
            f"{time.time() - t0:.0f}s",
            flush=True,
        )
        res[f] = r
    return res


def run_lc(cls, d, folds, cdir):
    out = {}
    for f, r in folds.items():
        p = cdir / f"lc{f}.pkl"
        if p.exists():
            out[f] = pickle.loads(p.read_bytes())
            continue
        out[f] = SE.learning_curve(d, f, SP.Config.from_dict(r["config"]), FRACS)
        p.write_bytes(pickle.dumps(out[f]))
    return out


def assemble(d, folds):
    n = len(d.y)
    dec, sc, cal, st = (
        np.full(n, -1),
        np.full(n, np.nan),
        np.full(n, np.nan),
        np.full(n, "", object),
    )
    for f, r in folds.items():
        te = r["te"]
        dec[te], sc[te], cal[te], st[te] = (
            r["decision"],
            r["score"],
            r["cal"],
            r["state"],
        )
    assert (dec >= 0).all()
    return dec, sc, cal, st


def tables(cls, d, folds, lc):
    dec, sc, cal, st = assemble(d, folds)
    y, g, w = d.y, d.g, d.w
    comps, f4, vc = comparators(cls, d)
    res = {"n": int(len(y)), "pos": int(y.sum()), "weighted": cls == "repetition"}
    res["optimised"] = EV.describe(y, dec, g, w)
    res["comparators"] = {k: EV.describe(y, v, g, w) for k, v in comps.items()}
    res["family_vs_comparators"] = EV.compare_family(y, dec, comps, g, w)
    res["vs_default"] = EV.paired(y, dec, comps["V0_argmax"], g, w)
    best = max(comps, key=lambda k: res["comparators"][k]["w_f1"])
    res["best_fixed_comparator"] = {
        "name": best,
        **EV.paired(y, dec, comps[best], g, w),
    }
    # Vc, fold 4 only (Vc-favourable, descriptive)
    ix = np.flatnonzero(f4)
    if len(ix) and (vc[ix] >= 0).all():
        res["vc_fold4_descriptive"] = {
            "n": int(len(ix)),
            "pos": int(y[ix].sum()),
            "optimised": EV.describe(y[ix], dec[ix], g[ix], w[ix]),
            "Vc": EV.describe(y[ix], vc[ix], g[ix], w[ix]),
            "delta": EV.paired(y[ix], dec[ix], vc[ix], g[ix], w[ix]),
        }
    res["coverage_risk"] = EV.coverage_risk(y, st, g, w)
    res["no_abstention_f1"] = res["optimised"]["w_f1"]
    # per-fold winners and stability
    pf = [
        {
            k: r[k]
            for k in (
                "fold",
                "config",
                "inner_f1",
                "thr",
                "t_or",
                "stop",
                "n_evaluated",
                "calibration",
                "band",
                "seconds",
                "n_train",
                "n_test",
            )
        }
        for r in folds.values()
    ]
    res["per_fold"] = pf
    keys = Counter(p["config"]["key"] for p in pf)
    res["stability"] = {
        "distinct_configs": len(keys),
        "modal_config_folds": keys.most_common(1)[0][1],
        "profile": Counter(p["config"]["profile"] for p in pf),
        "model": Counter(p["config"]["model"] for p in pf),
        "fusion": Counter(p["config"]["fusion"] for p in pf),
        "blocks": Counter("+".join(p["config"]["blocks"]) for p in pf),
        "calibration": Counter(p["calibration"] for p in pf),
    }
    res["selection_optimism"] = {
        "mean_inner_f1": float(np.mean([p["inner_f1"] for p in pf])),
        "outer_pooled_f1": res["optimised"]["w_f1"],
    }
    # learning curve (pooled over outer folds)
    curve = {}
    for fr in FRACS:
        pr = np.full(len(y), -1)
        for f in folds:
            e = lc[f][str(fr)]
            pr[e["te"]] = e["decision"]
        curve[str(fr)] = {"pred": pr, **EV.describe(y, pr, g, w, 1000)}
    curve["1.0"] = {"pred": dec, **EV.describe(y, dec, g, w, 1000)}
    res["learning_curve"] = {
        k: {kk: vv for kk, vv in v.items() if kk != "pred"} for k, v in curve.items()
    }
    res["learning_curve_delta_100_minus_75"] = EV.paired(
        y, curve["1.0"]["pred"], curve["0.75"]["pred"], g, w
    )
    res["learning_curve_delta_75_minus_25"] = EV.paired(
        y, curve["0.75"]["pred"], curve["0.25"]["pred"], g, w
    )
    return res, dec


def final_fit(cls, d, cdir):
    p = cdir / "final.pkl"
    if p.exists():
        return pickle.loads(p.read_bytes())
    tr = np.arange(len(d.y))
    sel = SE.search(d, tr, SP.SEED + 99)
    pipe = PP.fit_class_pipeline(
        d,
        tr,
        sel["best"],
        f"fit on DIHANA folds 1-4 ({len(tr)} rows); search stop={sel['stop']}, n={sel['n_evaluated']}",
    )
    out = {
        "pipe": pipe,
        "inner_f1": sel["best"]["inner_f1"],
        "stop": sel["stop"],
        "n_evaluated": sel["n_evaluated"],
        "trials": sel["trials"],
    }
    p.write_bytes(pickle.dumps(out))
    return out


def stage_classes(args):
    for cls in args.classes:
        cdir = OUT / "runs" / cls
        cdir.mkdir(parents=True, exist_ok=True)
        d = DT.load_dihana(cls)
        folds = run_folds(cls, d, cdir, args.budget)
        lc = run_lc(cls, d, folds, cdir)
        res, dec = tables(cls, d, folds, lc)
        fin = final_fit(cls, d, cdir)
        res["final_config"] = fin["pipe"].cfg.to_dict()
        res["final_search"] = {k: fin[k] for k in ("inner_f1", "stop", "n_evaluated")}
        (OUT / f"results_{cls}.json").write_text(
            json.dumps(jsonable(res), indent=1), "utf8"
        )
        with (OUT / f"predictions_{cls}.csv").open(
            "w", newline="", encoding="utf8"
        ) as fh:
            wr = csv.writer(fh)
            wr.writerow(["unit_id", "speaker", "fold", "w", "y", "decision"])
            for i in range(len(d.y)):
                wr.writerow(
                    [d.ids[i], d.g[i], int(d.outer[i]), d.w[i], d.y[i], int(dec[i])]
                )
        print(
            cls,
            "F1",
            round(res["optimised"]["w_f1"], 4),
            "vs default",
            res["vs_default"]["delta"],
            flush=True,
        )


def ext_metrics(pipe, recs, y, g, cls):
    from mrcd.decision import any_window_argmax
    from mrcd.constants import LABELS

    out = pipe.predict(recs)
    v0 = np.array([any_window_argmax(r["probs"], list(LABELS), cls) for r in recs])
    res = {
        "n": int(len(y)),
        "pos": int(y.sum()),
        "optimised": EV.describe(y, out["decision"], g),
        "V0_argmax": EV.describe(y, v0, g),
        "delta_vs_V0": EV.paired(y, out["decision"], v0, g),
        "coverage_risk": EV.coverage_risk(y, out["state"], g),
        "note": "zero-shot; corpus already inspected in earlier phases; not used for any selection",
    }
    if cls == "filler_word":
        from mrcd.optimize.blocks import rule_filler

        rule = np.array([rule_filler(r["text"]["p0"]) for r in recs])
        res["rule_asr_lexical"] = EV.describe(y, rule, g)
        res["delta_vs_rule"] = EV.paired(y, out["decision"], rule, g)
    return res


def stage_package(args):
    pipes, per = {}, {}
    for cls in SP.CLASSES:
        pipes[cls] = pickle.loads((OUT / "runs" / cls / "final.pkl").read_bytes())[
            "pipe"
        ]
        per[cls] = json.loads((OUT / f"results_{cls}.json").read_text("utf8"))
    prom = EV.promotion({c: per[c]["vs_default"] for c in SP.CLASSES})
    meta = {
        "promotion": jsonable(prom),
        "protocol_sha256": verify_frozen()["sha256"],
        "code_sha256": code_hashes(),
        "git_head": git_head(),
        "status": "promoted-candidate" if prom["promote"] else "opt-in",
        "caveat": "exploratory; DIHANA folds 1-4 only; scores are not calibrated probabilities",
    }
    man = PP.save_pipelines(pipes, MODELS, meta)
    (OUT / "promotion.json").write_text(
        json.dumps(jsonable({"promotion": prom, "manifest": man}), indent=1), "utf8"
    )
    ext = {}
    for corpus, cls in (("epic", "filler_word"), ("ameresco", "prolongation")):
        recs, y, g = DT.load_external(corpus, cls)
        ext[corpus] = ext_metrics(pipes[cls], recs, y, g, cls)
    (OUT / "external_zero_shot.json").write_text(
        json.dumps(jsonable(ext), indent=1), "utf8"
    )
    print("promote:", prom["promote"], prom["reason"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["classes", "package"])
    ap.add_argument("--classes", nargs="+", default=list(SP.CLASSES))
    ap.add_argument("--budget", type=int, default=SP.BUDGET)
    a = ap.parse_args()
    gate()
    fz = verify_frozen()
    print("protocol", fz["sha256"][:12], "frozen", fz["frozen_utc"], flush=True)
    {"classes": stage_classes, "package": stage_package}[a.stage](a)


if __name__ == "__main__":
    main()
