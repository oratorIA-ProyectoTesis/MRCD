"""M2 evaluation: service document on short turns vs the validated per-unit recipe (same pipelines), frozen gate.

    .venv\\Scripts\\python.exe scripts\\mejora_servicio\\m2_eval.py --profile o2026_11 [--route rule_p1]

Per-unit recipe (the validated one, computed from the stored per-turn features and per-turn decodes):
  filler_word / prolongation: the o2026_11 JSON pipelines on the stored unit records (P0 per-turn words);
  repetition: o2026_11 -> V0 any-window argmax on the stored window probabilities (all short turns with a label);
              o2026_11b -> the selected route on the per-turn P1 text (frame turns only, frame weights).
Service: a turn is positive for a class when its document has at least one in-scope event of that class.
"""

import common as K  # noqa: F401  (pins OMP before numpy)

import argparse
import json

import numpy as np

from mrcd.constants import LABELS
from mrcd.context_eval.textfeat import rule_rep
from mrcd.decision import any_window_argmax
from mrcd.optimize import data as OD
from mrcd.service import decision_profile as DP
from mrcd.service import s3_features as S3F

import freeze
from m2_run import short_units

M2 = K.OUT / "m2"
BINS = [(2.0, 3.0), (3.0, 5.0), (5.0, 8.0), (8.0, 13.0)]


def p1_text(uid: str) -> str:
    d = json.loads((K.RC.OUT / "asr" / "P1" / f"{uid}.json").read_text("utf8"))
    return " ".join(w["text"] for w in d["words"])


def gate(y, ps, pr, g, w, tol=-0.02) -> dict:
    d0, dist = K.paired_dist(y, ps, pr, g, w)
    ci = K.ci(dist) if len(dist) else [float("nan")] * 2
    return {
        "n": int(len(y)),
        "positives": int(np.sum(y)),
        "delta": d0,
        "ci95": ci,
        "service": K.describe(y, ps, g, w),
        "recipe": K.describe(y, pr, g, w),
        "pass": bool(d0 >= tol and ci[1] >= 0),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", required=True, choices=[DP.PROFILE_O, DP.PROFILE_OB])
    ap.add_argument("--route", default=None)
    args = ap.parse_args()
    freeze.verify()
    tag = args.profile + (f"_{args.route}" if args.route else "")
    docs = {}
    for f in sorted((M2 / "runs" / tag).glob("*.jsonl")):
        for line in f.read_text("utf8").splitlines():
            if line.strip():
                d = json.loads(line)
                docs[d["unit_id"]] = d
    units = short_units()
    missing = [u["unit_id"] for u in units if u["unit_id"] not in docs]
    if missing:
        raise SystemExit(f"{len(missing)} short turns have no document yet")
    pool = K.RC.load_pool()
    by = {u["unit_id"]: u for u in pool}
    T = K.turns_meta()
    feats = OD.load_features()
    pipes = DP.o_pipelines()
    ids = [u["unit_id"] for u in units]
    dur = np.array([u["dur_s"] for u in units])
    g = np.array([u["speaker"] for u in units])
    recs = [OD._rec(uid, feats, ("p0",), None) for uid in ids]
    res = {"profile": tag, "n_units": len(ids), "n_speakers": int(len(set(g.tolist())))}
    res["documents"] = {
        "n": len(ids),
        "completed": sum(1 for u in ids if docs[u]["status"] == "completed"),
        "invalid": sum(1 for u in ids if docs[u]["valid_errors"]),
        "error_codes": sorted({docs[u]["error"] for u in ids if docs[u]["error"]}),
        "short_audio_warning": sum(
            1 for u in ids if "short_audio_reduced_confidence" in docs[u]["warnings"]
        ),
    }
    svc = {
        c: np.array(
            [
                int(any(e["class"] == c and e["in_scope"] for e in docs[u]["events"]))
                for u in ids
            ]
        )
        for c in K.CLASSES
    }
    data = {}
    for cls in ("filler_word", "prolongation"):
        y = np.array([int(T[u][K.LABEL_COL[cls]]) for u in ids])
        pr = np.asarray(pipes[cls].predict(recs)["decision"]).astype(int)
        data[cls] = (np.ones(len(ids), bool), y, svc[cls], pr, np.ones(len(ids)))
    if args.profile == DP.PROFILE_O:
        lab = [by[u]["ref"]["rep_literal"] for u in ids]
        keep = np.array([v is not None for v in lab])
        y = np.array([int(v) if v is not None else 0 for v in lab])
        pr = np.array(
            [
                any_window_argmax(feats[u]["probs"], list(LABELS), "repetition")
                for u in ids
            ]
        ).astype(int)
        data["repetition"] = (keep, y, svc["repetition"], pr, np.ones(len(ids)))
    else:
        fr = {
            r["unit_id"]: r
            for r in json.loads((K.OUT / "m1" / "frame.json").read_text("utf8"))
        }
        keep = np.array([u in fr for u in ids])
        y = np.array([fr[u]["y_repetition"] if u in fr else 0 for u in ids])
        w = np.array([fr[u]["w"] if u in fr else 0.0 for u in ids])
        texts = [p1_text(u) if u in fr else "" for u in ids]
        if args.route == "rule_p1":
            pr = np.array([rule_rep(t) for t in texts]).astype(int)
        else:
            pr = np.asarray(S3F.load_route().decide(texts)[1]).astype(int)
        data["repetition"] = (keep, y, svc["repetition"], pr, w)
    res["band_2_13"], res["bins"] = {}, {}
    for cls, (keep, y, ps, pr, w) in data.items():
        res["band_2_13"][cls] = gate(y[keep], ps[keep], pr[keep], g[keep], w[keep])
        res["bins"][cls] = {}
        for a, b in BINS:
            m = keep & (dur >= a) & (dur < b)
            d0, _ = K.paired_dist(y[m], ps[m], pr[m], g[m], w[m], n_boot=200)
            res["bins"][cls][f"{a:g}-{b:g}"] = {
                "n": int(m.sum()),
                "positives": int(y[m].sum()),
                "delta": d0,
                "service_f1w": K.ST.wconfusion(y[m], ps[m], w[m])["w_f1"],
                "recipe_f1w": K.ST.wconfusion(y[m], pr[m], w[m])["w_f1"],
            }
        m3 = keep & (dur >= 3.0)
        res.setdefault("band_3_13", {})[cls] = gate(y[m3], ps[m3], pr[m3], g[m3], w[m3])
    docs_ok = (
        res["documents"]["completed"] == res["documents"]["n"]
        and res["documents"]["invalid"] == 0
    )
    primary = docs_ok and all(r["pass"] for r in res["band_2_13"].values())
    if not primary:
        minimum, why = (
            13.0,
            "primary gate on 2-13 s failed (or errors/invalid documents): the 13 s rejection stays",
        )
    elif all(res["bins"][c]["2-3"]["delta"] >= -0.02 for c in data):
        minimum, why = (
            2.0,
            "primary gate passed and every class keeps delta >= -0.02 on the 2-3 s bin",
        )
    elif all(r["pass"] for r in res["band_3_13"].values()):
        minimum, why = (
            3.0,
            "primary gate passed, the 2-3 s bin has a class with delta < -0.02, the 3-13 s band passes",
        )
    else:
        minimum, why = (
            13.0,
            "the 2-3 s bin fails and the 3-13 s band does not pass the gate: the 13 s rejection stays",
        )
    res["decision"] = {
        "documents_ok": bool(docs_ok),
        "primary_pass": bool(primary),
        "minimum_s": minimum,
        "why": why,
    }
    (M2 / f"metrics_m2_{tag}.json").write_text(
        json.dumps(res, indent=1, default=float), "utf8"
    )
    print(json.dumps(res["decision"], indent=1))
    for cls, r in res["band_2_13"].items():
        print(
            cls,
            round(r["service"]["w_f1"], 3),
            round(r["recipe"]["w_f1"], 3),
            round(r["delta"], 3),
            [round(x, 3) for x in r["ci95"]],
            r["pass"],
        )


if __name__ == "__main__":
    main()
