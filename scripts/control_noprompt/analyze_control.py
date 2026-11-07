"""No-prompt control analysis (results/control_noprompt_20261103/PROTOCOL.md). Writes metrics.json (aggregates only, no corpus text).

Reuses the exact survival / rule-FP / loop / bootstrap / Holm code of S2 (scripts/s2_noloop/analyze.py) and the C1 extras
(WER vs reference text A on negatives, adjacent identical pairs) of scripts/asr_recovery/s9_eval_c1.py.
usage: python analyze_control.py
"""

import os
import sys

NUMPY_FIRST = "numpy" in sys.modules
os.environ["OMP_NUM_THREADS"] = "4"
os.environ.setdefault("HF_HUB_OFFLINE", "1")
if NUMPY_FIRST:
    raise SystemExit("numpy imported before the OMP pin")
from pathlib import Path  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [
    str(REPO),
    str(REPO / "scripts" / "cierre3b"),
    str(REPO / "scripts" / "s2_noloop"),
]
import importlib  # noqa: E402

# S2 analysis module (bootstrap and Holm, unchanged); it must be imported BEFORE numpy (it checks the OMP pin).
_S2 = importlib.import_module("analyze")
NBOOT, SEED, boot_ratio, holm = _S2.NBOOT, _S2.SEED, _S2.boot_ratio, _S2.holm

import csv  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

from asr_analyze import align, norm_text  # noqa: E402  (frozen 3b normalization / alignment)
from mrcd.asr_profiles import detect_asr_anomalies  # noqa: E402
from mrcd.asr_recovery.common import LITERAL_SUBTYPES, OUT, load_pool  # noqa: E402
from mrcd.asr_recovery.noloop import is_loop_clip, loop_excess  # noqa: E402
from mrcd.asr_recovery.stats import corpus_wer  # noqa: E402
from mrcd.context_eval.reference import _norm_tokens  # noqa: E402
from mrcd.context_eval.survival import turn_survives  # noqa: E402
from mrcd.context_eval.textfeat import rule_rep  # noqa: E402
from mrcd.context_eval.transcript_reference import is_nonlexical_filler  # noqa: E402

DEST = REPO / "results" / "control_noprompt_20261103"
SRC = {
    "p0": OUT / "asr" / "P0",
    "p1": OUT / "asr" / "P1",
    "C1": OUT / "asr" / "C1",
    "p_none": DEST / "runs" / "asr" / "p_none",
}
# Code check only (never a result): CONTROL_DRYRUN=<dir> reads C1 in place of p_none and writes there.
DRYRUN = os.environ.get("CONTROL_DRYRUN")
if DRYRUN:
    SRC["p_none"] = SRC["C1"]
NONLEX_TAGS = ("f:a", "f:e", "f:m", "f:sin_identificar")
FAMILY_CONTRASTS = (("p0", "p_none"), ("p1", "p_none"))
FAMILY_METRICS = ("survival_rep", "survival_filler", "wer", "loop_rate", "rule_fp")
DESCRIPTIVE_CONTRASTS = (("p_none", "C1"), ("p1", "p0"))
DESCRIPTIVE_METRICS = (
    "survival_rev",
    "wer_neg",
    "ins_neg",
    "filler_emission_absent",
    "adj_pairs_neg",
)


def sha_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def n_fillers(text):
    t = _norm_tokens(text)
    return sum(
        is_nonlexical_filler(x, t[i + 1] if i + 1 < len(t) else None)
        for i, x in enumerate(t)
    )


def adjacent_pairs(text):
    t = _norm_tokens(text)
    return sum(a == b for a, b in zip(t, t[1:]))


def main():
    t0 = time.time()
    frame = json.loads((OUT / "subset.json").read_text("utf8"))["frame"]
    ids = [s["unit_id"] for s in frame]
    kind = {s["unit_id"]: s["kind"] for s in frame}
    assert len(ids) == 1025 and all(int(s["fold"]) in (1, 2, 3, 4) for s in frame)
    pool = {u["unit_id"]: u for u in load_pool()}
    rows = {
        r["turn_id"]: r
        for r in csv.DictReader(
            (REPO / "results/baselines_20261030/dihana/turns.csv").open(
                encoding="utf-8-sig"
            )
        )
        if int(r["fold"]) in (1, 2, 3, 4) and r["turn_id"] in kind
    }
    fill_pos = {
        u: int(
            sum(
                v
                for k, v in json.loads(rows[u]["raw_tag_counts"]).items()
                if k in NONLEX_TAGS
            )
            > 0
        )
        for u in ids
    }
    rep_pos = [u for u in ids if pool[u]["ref"]["rep_literal"] == 1]
    rev_pos = [u for u in ids if pool[u]["ref"]["revision"] == 1]
    neg = [u for u in ids if kind[u] == "neg"]
    f_pos = [u for u in ids if fill_pos[u]]
    f_abs = [u for u in ids if not fill_pos[u]]
    assert (len(rep_pos), len(neg), len(f_pos)) == (214, 679, 308), (
        len(rep_pos),
        len(neg),
        len(f_pos),
    )
    refA = {u: norm_text(pool[u]["text_A"]) for u in ids}
    g = {u: pool[u]["speaker"] for u in ids}

    per = {}
    for v, d in SRC.items():
        files = [d / f"{u}.json" for u in ids]
        missing = [p.name for p in files if not p.exists()]
        if missing:
            raise SystemExit(f"{v}: {len(missing)} units not decoded")
        recs = {u: json.loads(p.read_text("utf8")) for u, p in zip(ids, files)}
        r = {}
        for u in ids:
            t = " ".join(w["text"] for w in recs[u]["words"])
            h = norm_text(t)
            s_, dl, ins, _ = align(refA[u], h)
            r[u] = {
                "survival_rep": int(
                    pool[u]["ref"]["rep_literal"] == 1
                    and turn_survives(
                        "repetition", pool[u]["ref"]["rep_marks"], t, LITERAL_SUBTYPES
                    )
                ),
                "survival_rev": int(
                    pool[u]["ref"]["revision"] == 1
                    and turn_survives("revision", pool[u]["ref"]["rev_marks"], t, None)
                ),
                "n_fill": n_fillers(t),
                "rule_fp": int(rule_rep(t)),
                "err": s_ + dl + ins,
                "ins": ins,
                "n_ref": len(refA[u]),
                "loop": int(is_loop_clip(h, refA[u])),
                "excess": loop_excess(h, refA[u]),
                "lenient": int(
                    any(
                        x["kind"] == "repeated_ngram_loop"
                        for x in detect_asr_anomalies(t, "")
                    )
                ),
                "adj": adjacent_pairs(t),
                "empty": int(not recs[u]["words"]),
                "n_words": len(recs[u]["words"]),
                "sec": recs[u].get("sec"),
                "dur": recs[u]["n_samples"] / 16000,
                "text_neg": t if kind[u] == "neg" else None,
            }
        per[v] = r

    def arrays(v, metric):
        R = per[v]
        if metric == "survival_rep":
            us, n, d = (
                rep_pos,
                [R[u]["survival_rep"] for u in rep_pos],
                [1] * len(rep_pos),
            )
        elif metric == "survival_rev":
            us, n, d = (
                rev_pos,
                [R[u]["survival_rev"] for u in rev_pos],
                [1] * len(rev_pos),
            )
        elif metric == "survival_filler":
            us, n, d = f_pos, [int(R[u]["n_fill"] > 0) for u in f_pos], [1] * len(f_pos)
        elif metric == "filler_emission_absent":
            us, n, d = f_abs, [int(R[u]["n_fill"] > 0) for u in f_abs], [1] * len(f_abs)
        elif metric == "wer":
            us, n, d = ids, [R[u]["err"] for u in ids], [R[u]["n_ref"] for u in ids]
        elif metric == "wer_neg":
            us, n, d = neg, [R[u]["err"] for u in neg], [R[u]["n_ref"] for u in neg]
        elif metric == "ins_neg":
            us, n, d = neg, [R[u]["ins"] for u in neg], [R[u]["n_ref"] for u in neg]
        elif metric == "loop_rate":
            us, n, d = ids, [R[u]["loop"] for u in ids], [1] * len(ids)
        elif metric == "rule_fp":
            us, n, d = neg, [R[u]["rule_fp"] for u in neg], [1] * len(neg)
        elif metric == "adj_pairs_neg":
            us, n, d = neg, [R[u]["adj"] for u in neg], [1] * len(neg)
        else:
            raise KeyError(metric)
        return us, [g[u] for u in us], np.array(n, float), np.array(d, float)

    summary = {}
    for v in SRC:
        R = per[v]
        s = {}
        for m in FAMILY_METRICS + DESCRIPTIVE_METRICS:
            us, gg, n, d = arrays(v, m)
            r = boot_ratio(gg, n, np.zeros(len(n)), d)
            s[m] = {
                "k": float(n.sum()),
                "n": float(d.sum()),
                "n_units": len(us),
                "n_groups": r["n_groups"],
                "value": r["delta"],
                "ci": r["ci"],
            }
        s["loop_clips"] = int(sum(R[u]["loop"] for u in ids))
        s["loop_excess_tokens"] = int(sum(R[u]["excess"] for u in ids))
        s["lenient_loop_flag_units"] = int(sum(R[u]["lenient"] for u in ids))
        s["wer_vs_ref_A_on_negatives_c1_function"] = corpus_wer(
            [(pool[u]["text_A"], R[u]["text_neg"]) for u in neg]
        )
        s["units_with_adjacent_pair_on_negatives"] = int(
            sum(R[u]["adj"] > 0 for u in neg)
        )
        s["filler_tokens_total"] = int(sum(R[u]["n_fill"] for u in ids))
        s["empty_outputs"] = int(sum(R[u]["empty"] for u in ids))
        s["mean_words_per_unit"] = float(np.mean([R[u]["n_words"] for u in ids]))
        secs = [R[u]["sec"] for u in ids if R[u]["sec"] is not None]
        s["decode_rtf_per_unit_sum"] = (
            float(sum(secs) / sum(R[u]["dur"] for u in ids)) if secs else None
        )
        summary[v] = s

    def paired(a, b, metrics):
        out = {}
        for m in metrics:
            us1, g1, n1, d1 = arrays(a, m)
            us0, _, n0, d0 = arrays(b, m)
            assert us1 == us0 and np.array_equal(d1, d0)
            r = boot_ratio(g1, n1, n0, d1)
            if (
                m.startswith("survival")
                or m.endswith("_absent")
                or m in ("rule_fp", "loop_rate")
            ):
                r["gained"] = int(((n1 > 0) & (n0 == 0)).sum())
                r["lost"] = int(((n1 == 0) & (n0 > 0)).sum())
            out[m] = r
        return out

    contrasts = {}
    for a, b in FAMILY_CONTRASTS:
        contrasts[f"{a}-{b}"] = paired(a, b, FAMILY_METRICS + DESCRIPTIVE_METRICS)
    for a, b in DESCRIPTIVE_CONTRASTS:
        contrasts[f"{a}-{b}"] = paired(a, b, FAMILY_METRICS + DESCRIPTIVE_METRICS)
    fam = {
        f"{a}-{b}|{m}": contrasts[f"{a}-{b}"][m]["p"]
        for a, b in FAMILY_CONTRASTS
        for m in FAMILY_METRICS
    }
    adj = holm(fam, len(fam))
    for k, p in adj.items():
        c, m = k.split("|")
        contrasts[c][m]["p_holm_family10"] = p

    # Decision rules (PROTOCOL section 6), primary metric = literal-repetition survival.
    h1 = contrasts["p0-p_none"]["survival_rep"]
    h2 = contrasts["p1-p_none"]["survival_rep"]
    if h1["p_holm_family10"] < 0.05 and h1["ci"][1] < 0:
        h1v = "refuted: p0 is worse than p_none (CI upper < 0, Holm < 0.05)"
    elif h1["delta"] >= -0.02 and h1["ci"][0] > -0.05:
        h1v = "supported (no harm: point >= -0.02 and CI lower > -0.05)"
        if h1["p_holm_family10"] < 0.05 and h1["ci"][0] > 0:
            h1v += "; p0 also superior (CI lower > 0, Holm < 0.05)"
    else:
        h1v = "not demonstrated (harm not excluded)"
    if h2["p_holm_family10"] < 0.05 and h2["ci"][0] > 0:
        h2v = "supported (CI lower > 0, Holm < 0.05)"
    elif h2["p_holm_family10"] < 0.05 and h2["ci"][1] < 0:
        h2v = "refuted: p1 is worse than p_none"
    else:
        h2v = "not demonstrated"
    loop_eval = summary["p0"]["loop_clips"] >= 3 or summary["p_none"]["loop_clips"] >= 3

    env = (
        {"n": 0, "identical": 0}
        if DRYRUN
        else json.loads((DEST / "runs" / "env_check_p0.json").read_text("utf8"))
    )
    res = {
        "protocol_sha256": None
        if DRYRUN
        else json.loads((DEST / "PROTOCOL.sha256.json").read_text("utf8"))["sha256"],
        "frame": {
            "units": len(ids),
            "rep_literal_pos": len(rep_pos),
            "revision_pos": len(rev_pos),
            "sampled_negatives": len(neg),
            "filler_pos": len(f_pos),
            "filler_absent": len(f_abs),
            "speakers": len(set(g.values())),
            "dihana_folds": [1, 2, 3, 4],
        },
        "env_check_p0_redecode": {"n": env["n"], "identical": env["identical"]},
        "summary": summary,
        "contrasts": contrasts,
        "holm_family": sorted(fam),
        "decisions": {
            "H1_p0_ge_p_none": h1v,
            "H2_p1_gt_p_none": h2v,
            "loop_comparison_evaluable": bool(loop_eval),
        },
        "bootstrap": {"seed": SEED, "n_boot": NBOOT, "groups": "speaker"},
        "inputs_sha256": {
            v: hashlib.sha256(
                "".join(sorted(sha_file(SRC[v] / f"{u}.json") for u in ids)).encode()
            ).hexdigest()
            for v in SRC
        },
        "code_sha256": {
            str(p.relative_to(REPO)).replace("\\", "/"): sha_file(p)
            for p in (
                Path(__file__),
                REPO / "scripts/control_noprompt/decode_pnone.py",
                REPO / "scripts/s2_noloop/analyze.py",
                REPO / "mrcd/asr_recovery/decode.py",
            )
        },
        "seconds": round(time.time() - t0, 1),
    }
    (Path(DRYRUN) if DRYRUN else DEST).mkdir(parents=True, exist_ok=True)
    ((Path(DRYRUN) if DRYRUN else DEST) / "metrics.json").write_text(
        json.dumps(res, indent=1, default=float), "utf8"
    )
    for c in contrasts:
        for m in FAMILY_METRICS:
            r = contrasts[c][m]
            print(
                f"{c:12s} {m:16s} d={r['delta']:+.4f} ci=[{r['ci'][0]:+.4f},{r['ci'][1]:+.4f}] p={r['p']:.4f} holm={r.get('p_holm_family10', float('nan')):.4f}"
            )
    print(res["decisions"])


if __name__ == "__main__":
    main()
