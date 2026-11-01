"""S2 analysis (protocol: results/mejora_objetivos_20261101/S2/PROTOCOL.md). Paired, group-bootstrap, Holm over the fixed family of 16.

usage: python analyze.py --device cuda|cpu --branches C0 P1 A B C [D_X] --out DIR [--family-branches A B C D]
CPU 'C0' / 'P1' are the cached CPU int8 transcripts (3b off_p0/off_p1 for U_W, asr_recovery P0/P1 for U_S); other CPU branches are read from S2/asr/cpu.
"""

import os
import sys

NUMPY_FIRST = "numpy" in sys.modules
os.environ["OMP_NUM_THREADS"] = "4"
os.environ.setdefault("HF_HUB_OFFLINE", "1")
if NUMPY_FIRST:
    raise SystemExit("numpy imported before the OMP pin")
from pathlib import Path as _P  # noqa: E402

_REPO = _P(os.environ.get("MRCD_REPO") or _P(__file__).resolve().parents[2])
sys.path[:0] = [str(_REPO), str(_REPO / "scripts" / "cierre3b")]

import argparse  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

from asr_analyze import align, norm_text  # noqa: E402  (frozen WER normalization / alignment of the 3b report)
from mrcd.asr_profiles import detect_asr_anomalies  # noqa: E402
from mrcd.asr_recovery.common import LITERAL_SUBTYPES, load_pool  # noqa: E402
from mrcd.asr_recovery.noloop import is_loop_clip, loop_excess  # noqa: E402
from mrcd.asr_recovery.stats import corpus_wer  # noqa: E402
from mrcd.context_eval.survival import turn_survives  # noqa: E402
from mrcd.context_eval.textfeat import rule_rep  # noqa: E402

REPO = _REPO
S2 = REPO / "results/mejora_objetivos_20261101/S2"
W_UNITS = REPO / "results/cierre_oe1_oe3_20261031/3b/asr/units.json"
CACHE_W = REPO / "results/cierre_oe1_oe3_20261031/3b/asr/pred"
CACHE_S = REPO / "results/asr_recovery_20261031/asr"
SEED, NBOOT = 20261101, 5000
FAMILY_BRANCHES = ("A", "B", "C", "D")
METRICS = ("loop_rate", "wer", "survival", "rule_fp")


def words_of(path):
    return json.loads(Path(path).read_text("utf8"))["words"]


def load_branch(device, branch, kind, uids):
    """-> {uid: [word dicts]} for the given set kind ('W' or 'S')."""
    out = {}
    for u in uids:
        if device == "cpu" and branch in ("C0", "P1"):
            p = (
                CACHE_W / {"C0": "off_p0", "P1": "off_p1"}[branch]
                if kind == "W"
                else CACHE_S / {"C0": "P0", "P1": "P1"}[branch]
            ) / f"{u}.json"
        else:
            p = S2 / "asr" / device / branch / f"{u}.json"
        if not p.exists():
            return None
        out[u] = words_of(p)
    return out


def text_of(words):
    return " ".join(w["text"] for w in words)


def boot_ratio(g, n1, n0, d, seed=SEED, nboot=NBOOT):
    """Paired difference of ratio-of-sums (branch - reference) with a group bootstrap; two-sided bootstrap p."""
    g, n1, n0, d = (np.asarray(x) for x in (g, n1, n0, d))
    _, inv = np.unique(g.astype(str), return_inverse=True)
    k = int(inv.max()) + 1
    S1, S0, D = (np.zeros(k) for _ in range(3))
    np.add.at(S1, inv, n1)
    np.add.at(S0, inv, n0)
    np.add.at(D, inv, d)
    delta = float(S1.sum() / D.sum() - S0.sum() / D.sum())
    if k < 2:
        return {"delta": delta, "ci": [None, None], "p": None, "n_groups": k}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, k, (nboot, k))
    den = D[idx].sum(1)
    ok = den > 0
    dd = (S1[idx].sum(1)[ok] - S0[idx].sum(1)[ok]) / den[ok]
    B = len(dd)
    p = min(
        1.0, 2 * min((np.sum(dd <= 0) + 1) / (B + 1), (np.sum(dd >= 0) + 1) / (B + 1))
    )
    return {
        "delta": delta,
        "ci": [float(np.percentile(dd, 2.5)), float(np.percentile(dd, 97.5))],
        "p": float(p),
        "n_groups": k,
    }


def rate(g, n, d, seed=SEED, nboot=NBOOT):
    r = boot_ratio(g, n, np.zeros(len(n)), d, seed, nboot)
    return {"value": r["delta"], "ci": r["ci"], "n_groups": r["n_groups"]}


def holm(pvals: dict, m: int) -> dict:
    """Holm step-down over a family of fixed size m (missing comparisons count as p=1 placeholders at the end)."""
    items = sorted(((p, k) for k, p in pvals.items() if p is not None))
    adj, run = {}, 0.0
    for i, (p, k) in enumerate(items):
        run = max(run, min(1.0, (m - i) * p))
        adj[k] = run
    return adj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", required=True)
    ap.add_argument("--branches", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--control", default="C0")
    a = ap.parse_args()
    outd = Path(a.out)
    outd.mkdir(parents=True, exist_ok=True)

    uw = json.loads(W_UNITS.read_text("utf8"))
    W = {u["unit_id"]: u for u in uw}
    pool = load_pool()
    P = {u["unit_id"]: u for u in pool}
    frame = json.loads(
        (REPO / "results/asr_recovery_20261031/subset.json").read_text("utf8")
    )["frame"]
    S_pos = [s["unit_id"] for s in frame if s["rep_literal"] == 1]
    S_neg = [s["unit_id"] for s in frame if s["kind"] == "neg"]
    S_all = [s["unit_id"] for s in frame]
    assert len(S_pos) == 214 and len(S_neg) == 679 and len(W) == 135, (
        len(S_pos),
        len(S_neg),
        len(W),
    )
    refs = {u: norm_text(W[u]["ref_raw"]) for u in W}

    per, branches_ok = {}, []
    for b in a.branches:
        wd = load_branch(a.device, b, "W", list(W))
        sd = load_branch(a.device, b, "S", S_all)
        if wd is None or sd is None:
            print("MISSING decoded units for", a.device, b, "-> skipped")
            continue
        branches_ok.append(b)
        rows = {}
        for u in W:
            h = norm_text(text_of(wd[u]))
            s, dl, ins, _ = align(refs[u], h)
            lenient = any(
                x["kind"] == "repeated_ngram_loop"
                for x in detect_asr_anomalies(text_of(wd[u]), "")
            )
            rows[u] = {
                "group": W[u]["group"],
                "corpus": W[u]["corpus"],
                "n_ref": len(refs[u]),
                "err": s + dl + ins,
                "excess": loop_excess(h, refs[u]),
                "loop": int(is_loop_clip(h, refs[u])),
                "lenient": int(lenient),
            }
        srows = {}
        for u in S_all:
            t = text_of(sd[u])
            r = P[u]
            srows[u] = {
                "group": r["speaker"],
                "surv": int(
                    bool(r["ref"]["rep_literal"] == 1)
                    and turn_survives(
                        "repetition", r["ref"]["rep_marks"], t, LITERAL_SUBTYPES
                    )
                ),
                "fp": int(rule_rep(t)),
                "empty": int(not sd[u]),
                "n_words": len(sd[u]),
            }
        per[b] = {"W": rows, "S": srows, "S_text": {u: text_of(sd[u]) for u in S_neg}}

    def arrays(b, metric):
        if metric in ("loop_rate", "wer"):
            R = per[b]["W"]
            us = list(W)
            g = [R[u]["group"] for u in us]
            if metric == "loop_rate":
                return (
                    us,
                    g,
                    np.array([R[u]["loop"] for u in us], float),
                    np.ones(len(us)),
                )
            return (
                us,
                g,
                np.array([R[u]["err"] for u in us], float),
                np.array([R[u]["n_ref"] for u in us], float),
            )
        R = per[b]["S"]
        us = S_pos if metric == "survival" else S_neg
        key = "surv" if metric == "survival" else "fp"
        return (
            us,
            [R[u]["group"] for u in us],
            np.array([R[u][key] for u in us], float),
            np.ones(len(us)),
        )

    summary = {}
    for b in branches_ok:
        R, S = per[b]["W"], per[b]["S"]
        d = {"W": {}, "S": {}}
        for name, sel in (
            ("all", None),
            ("ameresco", "ameresco"),
            ("dihana", "dihana"),
        ):
            us = [u for u in W if sel is None or W[u]["corpus"] == sel]
            g = [R[u]["group"] for u in us]
            d["W"][name] = {
                "n_units": len(us),
                "n_groups": len(set(g)),
                "loop_clips": int(sum(R[u]["loop"] for u in us)),
                "loop_rate": rate(g, [R[u]["loop"] for u in us], np.ones(len(us))),
                "lenient_flag_clips": int(sum(R[u]["lenient"] for u in us)),
                "excess_tokens": int(sum(R[u]["excess"] for u in us)),
                "wer": rate(g, [R[u]["err"] for u in us], [R[u]["n_ref"] for u in us]),
                "ci_note": "4 recordings: CI not informative"
                if name == "ameresco"
                else "",
            }
        for metric in ("survival", "rule_fp"):
            us, g, n, den = arrays(b, metric)
            d["S"][metric] = {"k": int(n.sum()), "n": int(den.sum()), **rate(g, n, den)}
        neg_pairs = [(P[u]["text_A"], per[b]["S_text"][u]) for u in S_neg]
        d["S"]["wer_vs_ref_A_on_negatives"] = corpus_wer(neg_pairs)
        d["S"]["empty_outputs"] = int(sum(S[u]["empty"] for u in S_all))
        d["S"]["mean_words_per_unit"] = float(np.mean([S[u]["n_words"] for u in S_all]))
        wall = []
        rec_dir = S2 / "asr" / a.device / b
        if rec_dir.exists():
            for p in rec_dir.glob("*.json"):
                if p.name != "_meta.json":
                    j = json.loads(p.read_text("utf8"))
                    wall.append((j["wall_s"], j["duration_s"], j.get("diag")))
            if wall:
                d["timing"] = {
                    "rtf": sum(x[0] for x in wall) / sum(x[1] for x in wall),
                    "units": len(wall),
                    "looped_chunks": sum(
                        (x[2] or {}).get("looped_chunks", 0) for x in wall
                    ),
                    "unresolved_chunks": sum(
                        (x[2] or {}).get("unresolved", 0) for x in wall
                    ),
                    "redecoded_chunks": sum(
                        len((x[2] or {}).get("redecoded", [])) for x in wall
                    ),
                }
        summary[b] = d

    paired = {}
    for b in branches_ok:
        for ref in (a.control, "P1"):
            if b == ref or b == a.control or ref not in branches_ok:
                continue
            for m in METRICS:
                us1, g, n1, d1 = arrays(b, m)
                us0, g0, n0, d0 = arrays(ref, m)
                assert us1 == us0 and np.array_equal(d1, d0)
                paired[f"{b}_vs_{ref}|{m}"] = boot_ratio(g, n1, n0, d1)
    fam_p = {
        k: v["p"]
        for k, v in paired.items()
        if k.endswith("")
        and k.split("_vs_")[1].startswith(a.control + "|")
        and k.split("_vs_")[0].split("_")[0] in FAMILY_BRANCHES
    }
    adj = holm(fam_p, 16)
    for k, v in adj.items():
        paired[k]["holm_p_family16"] = v

    # decision rule (protocol section 4.1-4.2)
    decisions = {}
    c0 = summary[a.control]["W"]["all"]["loop_clips"]
    for b in branches_ok:
        if b in (a.control, "P1"):
            continue
        n_loop = summary[b]["W"]["all"]["loop_clips"]
        pw = paired[f"{b}_vs_{a.control}|wer"]
        ps = paired[f"{b}_vs_{a.control}|survival"]
        pf = paired[f"{b}_vs_{a.control}|rule_fp"]
        reduces = c0 >= 3 and n_loop <= c0 - 3
        wer_ok = pw["delta"] <= 0.005 and pw["ci"][1] is not None and pw["ci"][1] < 0.02
        surv_ok = (
            ps["delta"] >= -0.02 and ps["ci"][0] is not None and ps["ci"][0] > -0.05
        )
        fp_ok = pf["delta"] <= 0.01
        decisions[b] = {
            "control_loop_clips": c0,
            "branch_loop_clips": n_loop,
            "evaluable": c0 >= 3,
            "reduces_loops": bool(reduces),
            "wer_ok": bool(wer_ok),
            "survival_ok": bool(surv_ok),
            "rule_fp_ok": bool(fp_ok),
            "qualifies": bool(reduces and wer_ok and surv_ok and fp_ok),
        }
    (outd / f"analysis_{a.device}.json").write_text(
        json.dumps(
            {
                "branches": branches_ok,
                "summary": summary,
                "paired": paired,
                "decisions": decisions,
                "seed": SEED,
                "nboot": NBOOT,
            },
            indent=1,
        ),
        "utf8",
    )

    L = [f"# S2 tables - device {a.device} (EXPLORATORY; control {a.control})", ""]
    L += [
        "## U_W: loops and WER vs human reference (loop clip = excess >= 20 tokens beyond the reference)",
        "",
        "| branch | loop clips all (AMERESCO / DIHANA) | lenient flag clips | excess tokens | WER all [95% CI] | WER AMERESCO (4 rec.) | WER DIHANA |",
        "|---|---|---|---|---|---|---|",
    ]
    for b in branches_ok:
        w = summary[b]["W"]
        f = lambda x: (
            f"{x['value']:.3f} [{x['ci'][0]:.3f}, {x['ci'][1]:.3f}]"
            if x["ci"][0] is not None
            else f"{x['value']:.3f}"
        )
        L.append(
            f"| {b} | {w['all']['loop_clips']}/135 ({w['ameresco']['loop_clips']}/75, {w['dihana']['loop_clips']}/60) | {w['all']['lenient_flag_clips']} | {w['all']['excess_tokens']} | {f(w['all']['wer'])} | {w['ameresco']['wer']['value']:.3f} | {w['dihana']['wer']['value']:.3f} |"
        )
    L += [
        "",
        "## U_S: survival of literal repetition (214 positives) and rule false positives (679 sampled negatives)",
        "",
        "| branch | survival k/n [95% CI] | rule FP k/n [95% CI] | WER vs ref A on negatives | empty outputs | mean words/unit | RTF (all units) | looped chunks / re-decoded / unresolved |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for b in branches_ok:
        s = summary[b]["S"]
        t = summary[b].get("timing", {})
        L.append(
            f"| {b} | {s['survival']['k']}/{s['survival']['n']} = {s['survival']['value']:.3f} [{s['survival']['ci'][0]:.3f}, {s['survival']['ci'][1]:.3f}] | "
            f"{s['rule_fp']['k']}/{s['rule_fp']['n']} = {s['rule_fp']['value']:.4f} [{s['rule_fp']['ci'][0]:.4f}, {s['rule_fp']['ci'][1]:.4f}] | {s['wer_vs_ref_A_on_negatives']:.3f} | {s['empty_outputs']} | {s['mean_words_per_unit']:.2f} | "
            f"{t.get('rtf', float('nan')):.3f} | {t.get('looped_chunks', '-')}/{t.get('redecoded_chunks', '-')}/{t.get('unresolved_chunks', '-')} |"
        )
    L += [
        "",
        f"## Paired differences (branch - reference), group bootstrap {NBOOT}x, Holm over a fixed family of 16 (rows vs {a.control} only)",
        "",
        "| comparison | metric | delta [95% CI] | groups | p | Holm p |",
        "|---|---|---|---|---|---|",
    ]
    for k, v in paired.items():
        cmp, m = k.split("|")
        ci = (
            f"[{v['ci'][0]:+.4f}, {v['ci'][1]:+.4f}]"
            if v["ci"][0] is not None
            else "n/a"
        )
        L.append(
            f"| {cmp} | {m} | {v['delta']:+.4f} {ci} | {v['n_groups']} | {('%.4f' % v['p']) if v['p'] is not None else 'n/a'} | {('%.4f' % v['holm_p_family16']) if 'holm_p_family16' in v else '(outside family)'} |"
        )
    L += [
        "",
        "## Decision rule (protocol 4.1-4.2)",
        "",
        "| branch | control loop clips | branch loop clips | reduces loops | WER ok | survival ok | rule FP ok | qualifies |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for b, d in decisions.items():
        L.append(
            f"| {b} | {d['control_loop_clips']} | {d['branch_loop_clips']} | {d['reduces_loops']} | {d['wer_ok']} | {d['survival_ok']} | {d['rule_fp_ok']} | {d['qualifies']} |"
        )
    (outd / f"tables_{a.device}.md").write_text("\n".join(L) + "\n", "utf8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
