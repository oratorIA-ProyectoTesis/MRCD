"""M1 evaluation: turn-level decisions from the service documents, gates G1, G2, G5 and the frozen selection rule.

    .venv\\Scripts\\python.exe scripts\\mejora_servicio\\m1_eval.py

Writes ``m1/metrics_m1.json`` (all numbers) and ``m1/decision_m1.json`` (gates and the selected candidate).
"""

import common as K  # noqa: F401  (pins OMP before numpy)

import json

import numpy as np

import freeze

M1 = K.OUT / "m1"
SYSTEMS = ("base", "cand_a", "cand_b")
CANDS = ("cand_a", "cand_b")


def turn_preds(sessions, runs, mode):
    pred = {s: {} for s in SYSTEMS}
    status = {s: [] for s in SYSTEMS}
    for sess in sessions:
        r = runs[sess["session_id"]]
        for s in SYSTEMS:
            doc = r["runs"][s]
            status[s].append((doc["status"], len(doc["valid_errors"])))
            got = K.assign_events(sess["turns"], doc["events"], mode)
            for uid, cl in got.items():
                pred[s][uid] = cl
    return pred, status


def main():
    freeze.verify()
    fr = json.loads((M1 / "frame.json").read_text("utf8"))
    sessions = json.loads((M1 / "sessions.json").read_text("utf8"))
    runs = {}
    for s in sessions:
        f = M1 / "runs" / f"{s['session_id']}.json"
        if not f.exists():
            raise SystemExit(f"missing run {f.name}")
        runs[s["session_id"]] = json.loads(f.read_text("utf8"))
    ids = [r["unit_id"] for r in fr]
    w = np.array([r["w"] for r in fr])
    g = np.array([r["speaker"] for r in fr])
    Y = {c: np.array([r[f"y_{c}"] for r in fr]) for c in K.CLASSES}
    out = {
        "n_turns": len(ids),
        "n_speakers": int(len(set(g.tolist()))),
        "n_sessions": len(sessions),
    }
    for mode in ("max_overlap", "any_overlap"):
        pred, status = turn_preds(sessions, runs, mode)
        P = {
            s: {c: np.array([int(c in pred[s][u]) for u in ids]) for c in K.CLASSES}
            for s in SYSTEMS
        }
        res = {
            "documents": {
                s: {
                    "n": len(status[s]),
                    "completed": sum(1 for st, _ in status[s] if st == "completed"),
                    "invalid": sum(1 for _, e in status[s] if e),
                }
                for s in SYSTEMS
            },
            "describe": {
                s: {c: K.describe(Y[c], P[s][c], g, w) for c in K.CLASSES}
                for s in SYSTEMS
            },
        }
        # G1: repetition, candidate minus base, Holm over the two candidates
        rows = []
        for c in CANDS:
            d0, dist = K.paired_dist(
                Y["repetition"], P[c]["repetition"], P["base"]["repetition"], g, w
            )
            rows.append(
                {
                    "test": f"G1 {c} - base, repetition F1w",
                    "cand": c,
                    "delta": d0,
                    "dist": dist,
                }
            )
        K.holm_ci_family(rows)
        res["G1"] = [{k: v for k, v in r.items() if k != "dist"} for r in rows]
        # G2: filler_word and prolongation, candidate minus base
        res["G2"] = []
        for c in CANDS:
            for cls in ("filler_word", "prolongation"):
                d0, dist = K.paired_dist(Y[cls], P[c][cls], P["base"][cls], g, w)
                ci = K.ci(dist) if len(dist) else [0.0, 0.0]
                identical = bool((P[c][cls] == P["base"][cls]).all())
                res["G2"].append(
                    {
                        "cand": c,
                        "class": cls,
                        "delta": d0,
                        "ci95": ci,
                        "turn_decisions_identical": identical,
                        "pass": bool(d0 >= -0.01 and ci[1] >= 0),
                    }
                )
        # G5: repetition false-positive rate on reference negatives (weighted), candidate minus base <= +1 pp
        neg = Y["repetition"] == 0
        res["G5"] = []
        for c in CANDS:
            d = K.paired_rate_diff(
                P[c]["repetition"][neg], P["base"]["repetition"][neg], g[neg], w[neg]
            )
            d["cand"] = c
            d["pass"] = bool(d["delta"] <= 0.01)
            res["G5"].append(d)
        # selection descriptive: cand_b minus cand_a
        d0, dist = K.paired_dist(
            Y["repetition"], P["cand_b"]["repetition"], P["cand_a"]["repetition"], g, w
        )
        res["b_minus_a"] = {
            "delta": d0,
            "ci95": K.ci(dist),
            "p_boot": K.p_two_sided(dist),
        }
        # identity of the non-repetition events between base and candidates (same P0 decode)
        same = {c: 0 for c in CANDS}
        for sess in sessions:
            r = runs[sess["session_id"]]["runs"]
            key = lambda d: [  # noqa: E731
                (e["class"], e["start_s"], e["end_s"], e["score"])
                for e in d["events"]
                if e["class"] != "repetition"
            ]
            for c in CANDS:
                same[c] += int(key(r[c]) == key(r["base"]))
        res["non_repetition_events_identical_sessions"] = same
        out[mode] = res
    # gates on the PRIMARY mapping only
    prim = out["max_overlap"]
    gates = {}
    for c in CANDS:
        g1 = next(r for r in prim["G1"] if r["cand"] == c)
        g2 = [r for r in prim["G2"] if r["cand"] == c]
        g5 = next(r for r in prim["G5"] if r["cand"] == c)
        docs_ok = (
            prim["documents"][c]["completed"] == prim["documents"][c]["n"]
            and prim["documents"][c]["invalid"] == 0
        )
        gates[c] = {
            "G1": bool(g1["holm_lower_gt_0"]),
            "G2": all(r["pass"] for r in g2),
            "G5": bool(g5["pass"]),
            "documents_ok": bool(docs_ok),
        }
        gates[c]["statistical_gates_pass"] = all(gates[c].values())
    passing = [c for c in CANDS if gates[c]["statistical_gates_pass"]]
    if len(passing) == 2:
        bma = prim["b_minus_a"]
        chosen = "cand_b" if bma["ci95"][0] > 0 else "cand_a"
        why = "both pass; b chosen only if b - a has a 95% CI above 0, else the simpler a (rule)"
    elif len(passing) == 1:
        chosen, why = passing[0], "only this candidate passes G1, G2 and G5"
    else:
        chosen, why = None, "no candidate passes G1, G2 and G5: nothing is adopted"
    decision = {
        "gates_g1_g2_g5": gates,
        "selected": chosen,
        "selected_route": {"cand_a": "rule_p1", "cand_b": "rep_ctx_p1", None: None}[
            chosen
        ],
        "why": why,
        "pending": "G3 (incremental == offline) and G4 (wait after 2 min <= 10 s) are measured on the selected candidate",
    }
    (M1 / "metrics_m1.json").write_text(
        json.dumps(out, indent=1, default=float), "utf8"
    )
    (M1 / "decision_m1.json").write_text(json.dumps(decision, indent=1), "utf8")
    print(json.dumps(decision, indent=1))


if __name__ == "__main__":
    main()
