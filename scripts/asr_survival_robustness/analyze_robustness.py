"""Robustness analyses requested by simulated reviewer R2 (results/asr_survival_robustness_20261104/ADDENDUM.md).

Writes metrics.json and unit_flags.csv (aggregates and 0/1 flags only, no corpus text). Reuses the frozen helpers of
scripts/asr_survival_models/analyze_models.py (survival, tags, tiers, bootstrap, Holm).
usage: python analyze_robustness.py                       (CPU, mrcd venv)
       ASR_DRYRUN=<dir> python analyze_robustness.py       (code check only: stand-in systems, output in <dir>, never a result)
"""

import os
import sys

if "numpy" in sys.modules:
    raise SystemExit("numpy imported before the OMP pin")
os.environ["OMP_NUM_THREADS"] = "4"
os.environ.setdefault("HF_HUB_OFFLINE", "1")
from pathlib import Path  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "scripts" / "asr_survival_models"), str(REPO)]
import analyze_models as AM  # noqa: E402  (imports S2 before numpy: OMP pin check)

import csv  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

from mrcd.asr_recovery.common import LITERAL_SUBTYPES, OUT, load_pool  # noqa: E402
from mrcd.asr_recovery.stats import corpus_wer  # noqa: E402
from mrcd.context_eval.survival import turn_survives  # noqa: E402
from mrcd.context_eval.textfeat import rule_rep  # noqa: E402

NBOOT, SEED, boot_ratio, holm, tier = AM.NBOOT, AM.SEED, AM.boot_ratio, AM.holm, AM.tier
DEST = REPO / "results" / "asr_survival_robustness_20261104"
PREV = REPO / "results" / "asr_survival_models_20261104" / "runs" / "asr"
CTRL = REPO / "results" / "control_noprompt_20261103" / "runs" / "asr" / "p_none"
CHM_REF = REPO / "results" / "external_chm150_20261102"
DRYRUN = os.environ.get("ASR_DRYRUN")

# --- DIHANA systems -----------------------------------------------------------------------------------------------------
SYSTEMS = {  # name -> (directory, kind)
    "small_P0": (OUT / "asr" / "P0", "words"),
    "small_P1": (OUT / "asr" / "P1", "words"),
    "small_C1": (OUT / "asr" / "C1", "words"),
    "small_pnone": (CTRL, "words"),
    "sg_P0": (DEST / "runs/asr/sg_P0", "words"),
    "sg_P1": (DEST / "runs/asr/sg_P1", "words"),
    "sg_C1": (DEST / "runs/asr/sg_C1", "words"),
    "sg_pnone": (DEST / "runs/asr/sg_p_none", "words"),
    "lv3_P0": (PREV / "lv3_P0", "words"),
    "lv3_P1": (PREV / "lv3_P1", "words"),
    "lv3_C1": (PREV / "lv3_C1", "words"),
    "lv3_pnone": (PREV / "lv3_p_none", "words"),
    "cw_large": (PREV / "cw_large", "cw"),
}
if DRYRUN:  # code check only
    for k in ("sg_P0", "sg_P1", "sg_C1", "sg_pnone"):
        SYSTEMS[k] = (SYSTEMS["small_" + k.split("_")[1]][0], "words")

VARIANTS = ("P0", "P1", "C1", "pnone")
DEV_FAMILY = [(f"sg_{v}", f"small_{v}") for v in VARIANTS] + [
    (f"lv3_{v}", f"sg_{v}") for v in VARIANTS
]
FAMILY_METRICS = ("survival_rep", "survival_filler")
ALL_METRICS = (
    "survival_rep",
    "survival_rev",
    "survival_filler",
    "wer",
    "rule_fp",
    "filler_emission_absent",
)
MIN_RECOVERED = 30


def sha_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load_text(rec, kind):
    if kind == "words":
        return " ".join(w["text"] for w in rec["words"])
    return AM.cw_score_text(rec["text"])


# --- agreement statistics ---------------------------------------------------------------------------------------------
def kappa(a, b):
    a, b = np.asarray(a, int), np.asarray(b, int)
    n = len(a)
    po = float((a == b).mean())
    pe = float(a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean()))
    if pe >= 1.0:
        return None
    return (po - pe) / (1 - pe)


def agreement_stats(a, b):
    a, b = np.asarray(a, int), np.asarray(b, int)
    n11 = int(((a == 1) & (b == 1)).sum())
    n10 = int(((a == 1) & (b == 0)).sum())
    n01 = int(((a == 0) & (b == 1)).sum())
    n00 = int(((a == 0) & (b == 0)).sum())
    return {
        "n": len(a),
        "both": n11,
        "a_only": n10,
        "b_only": n01,
        "neither": n00,
        "observed_agreement": (n11 + n00) / len(a),
        "kappa": kappa(a, b),
        "p_b_given_a": n11 / (n11 + n10) if n11 + n10 else None,
        "p_a_given_b": n11 / (n11 + n01) if n11 + n01 else None,
        "jaccard_positive": n11 / (n11 + n10 + n01) if n11 + n10 + n01 else None,
    }


def cluster_boot(groups, fn, nboot=NBOOT, seed=SEED):
    """Percentile 95% CI of fn(index_array) resampling groups (speakers) with replacement."""
    groups = np.asarray(groups).astype(str)
    uniq, inv = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(inv == i) for i in range(len(uniq))]
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(nboot):
        pick = rng.integers(0, len(uniq), len(uniq))
        idx = np.concatenate([members[i] for i in pick])
        v = fn(idx)
        if v is not None and not (isinstance(v, float) and np.isnan(v)):
            vals.append(v)
    if len(vals) < nboot * 0.9:
        return {"ci": [None, None], "n_valid": len(vals)}
    return {
        "ci": [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))],
        "n_valid": len(vals),
    }


def agreement_with_ci(a, b, groups):
    a, b = np.asarray(a, int), np.asarray(b, int)
    out = agreement_stats(a, b)

    def cond(idx, x, y):
        s = x[idx].sum()
        return float((x[idx] & y[idx]).sum() / s) if s else None

    out["kappa_ci"] = cluster_boot(groups, lambda i: kappa(a[i], b[i]))["ci"]
    out["observed_agreement_ci"] = cluster_boot(
        groups, lambda i: float((a[i] == b[i]).mean())
    )["ci"]
    out["p_b_given_a_ci"] = cluster_boot(groups, lambda i: cond(i, a, b))["ci"]
    out["p_a_given_b_ci"] = cluster_boot(groups, lambda i: cond(i, b, a))["ci"]
    return out


def reading_agree(ci):
    lo, hi = ci
    if lo is None:
        return "not estimable"
    if lo >= 0.75:
        return "corroborated (CI lower >= 0.75)"
    if hi < 0.50:
        return "not corroborated (CI upper < 0.50)"
    return "inconclusive"


def sub_rate(g, flag_by_unit, units):
    n = np.array([flag_by_unit[u] for u in units], float)
    r = boot_ratio([g[u] for u in units], n, np.zeros(len(n)), np.ones(len(n)))
    return {
        "k": int(n.sum()),
        "n": len(units),
        "value": r["delta"],
        "ci": r["ci"],
        "n_groups": r["n_groups"],
    }


# --- DIHANA ----------------------------------------------------------------------------------------------------------
def dihana():
    frame = json.loads((OUT / "subset.json").read_text("utf8"))["frame"]
    ids = [s["unit_id"] for s in frame]
    kind = {s["unit_id"]: s["kind"] for s in frame}
    assert len(ids) == 1025
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
                if k in AM.NONLEX_TAGS
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
    assert (len(rep_pos), len(rev_pos), len(neg), len(f_pos)) == (214, 161, 679, 308)
    g = {u: pool[u]["speaker"] for u in ids}

    per, texts_hash = {}, {}
    for v, (d, k) in SYSTEMS.items():
        files = [d / f"{u}.json" for u in ids]
        missing = [p.name for p in files if not p.exists()]
        if missing:
            raise SystemExit(f"{v}: {len(missing)} units not decoded")
        recs = {u: json.loads(p.read_text("utf8")) for u, p in zip(ids, files)}
        r = {}
        for u in ids:
            t = load_text(recs[u], k)
            tag_fill = AM.cw_filled_tags(recs[u]["text"]) if k == "cw" else 0
            _, dl, ins, _ = None, None, None, None
            r[u] = {
                "text": t,
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
                "n_fill": AM.n_fillers(t) + tag_fill,
                "rule": int(rule_rep(t)),
            }
        per[v] = r
        texts_hash[v] = hashlib.sha256(
            "".join(sorted(sha_file(d / f"{u}.json") for u in ids)).encode()
        ).hexdigest()

    def arr(v, m):
        R = per[v]
        spec = {
            "survival_rep": (rep_pos, lambda u: R[u]["survival_rep"]),
            "survival_rev": (rev_pos, lambda u: R[u]["survival_rev"]),
            "survival_filler": (f_pos, lambda u: int(R[u]["n_fill"] > 0)),
            "filler_emission_absent": (f_abs, lambda u: int(R[u]["n_fill"] > 0)),
            "rule_fp": (neg, lambda u: R[u]["rule"]),
        }[m]
        us = spec[0]
        return (
            us,
            [g[u] for u in us],
            np.array([spec[1](u) for u in us], float),
            np.ones(len(us)),
        )

    metrics = (
        "survival_rep",
        "survival_rev",
        "survival_filler",
        "filler_emission_absent",
        "rule_fp",
    )
    summary = {}
    for v in SYSTEMS:
        s = {}
        for m in metrics:
            us, gg, n, d = arr(v, m)
            r = boot_ratio(gg, n, np.zeros(len(n)), d)
            s[m] = {
                "k": float(n.sum()),
                "n": float(d.sum()),
                "value": r["delta"],
                "ci": r["ci"],
            }
        for m in ("survival_rep", "survival_rev", "survival_filler"):
            s[m]["reading"] = tier(s[m]["ci"])
        s["wer_vs_ref_A_all_units"] = corpus_wer(
            [(pool[u]["text_A"], per[v][u]["text"]) for u in ids]
        )
        s["empty_outputs"] = int(sum(not per[v][u]["text"].strip() for u in ids))
        summary[v] = s

    def paired(a, b, ms):
        out = {}
        for m in ms:
            us1, g1, n1, d1 = arr(a, m)
            us0, _, n0, _ = arr(b, m)
            assert us1 == us0
            r = boot_ratio(g1, n1, n0, d1)
            r["gained"] = int(((n1 > 0) & (n0 == 0)).sum())
            r["lost"] = int(((n1 == 0) & (n0 > 0)).sum())
            out[m] = r
        return out

    contrasts = {f"{a}-{b}": paired(a, b, metrics) for a, b in DEV_FAMILY}
    # descriptive extras (outside the Holm family): P1 - P0 on small GPU, and the original size gap on the mixed pairs
    for a, b in (
        ("sg_P1", "sg_P0"),
        ("lv3_P0", "small_P0"),
        ("lv3_pnone", "small_pnone"),
        ("lv3_P1", "small_P1"),
        ("lv3_C1", "small_C1"),
    ):
        contrasts[f"{a}-{b}"] = paired(a, b, metrics)
    fam = {
        f"{a}-{b}|{m}": contrasts[f"{a}-{b}"][m]["p"]
        for a, b in DEV_FAMILY
        for m in FAMILY_METRICS
    }
    assert len(fam) == 16
    for k, p in holm(fam, len(fam)).items():
        c, m = k.split("|")
        contrasts[c][m]["p_holm_family16"] = p

    def sig(c, m, sign):
        r = contrasts[c][m]
        h = r.get("p_holm_family16")
        if h is None or r["ci"][0] is None:
            return False
        return h < 0.05 and (r["ci"][0] > 0 if sign > 0 else r["ci"][1] < 0)

    # --- decisions D5 (device confound), frozen in ADDENDUM section 6
    d5 = {}
    for m in FAMILY_METRICS:
        dev = contrasts["sg_P0-small_P0"][m]
        size = contrasts["lv3_P0-sg_P0"][m]
        orig = contrasts["lv3_P0-small_P0"][m]
        size_sig_neg = sig("lv3_P0-sg_P0", m, -1)
        dev_sig = sig("sg_P0-small_P0", m, +1) or sig("sg_P0-small_P0", m, -1)
        if size_sig_neg and not dev_sig:
            verdict = "size effect at matched device (device not detectable)"
        elif size_sig_neg and dev_sig:
            verdict = "both size and device contribute"
        elif not size_sig_neg and dev_sig:
            verdict = "device effect; size not detectable at matched device"
        else:
            verdict = "neither detectable (inconclusive)"
        d5[f"P0_{m}"] = {
            "device_sg_minus_small": dev["delta"],
            "size_lv3_minus_sg": size["delta"],
            "mixed_original_lv3_minus_small": orig["delta"],
            "size_share_of_original": (size["delta"] / orig["delta"])
            if orig["delta"]
            else None,
            "verdict": verdict,
        }
    pn = contrasts["lv3_pnone-sg_pnone"]["survival_rep"]
    pn_dev = contrasts["sg_pnone-small_pnone"]["survival_rep"]
    d5["pnone_no_size_effect_rep"] = {
        "size_lv3_minus_sg": pn["delta"],
        "size_ci": pn["ci"],
        "holm": pn.get("p_holm_family16"),
        "no_size_effect": not sig("lv3_pnone-sg_pnone", "survival_rep", -1)
        and not sig("lv3_pnone-sg_pnone", "survival_rep", +1),
        "device_sg_minus_small": pn_dev["delta"],
        "device_ci": pn_dev["ci"],
        "device_holm": pn_dev.get("p_holm_family16"),
    }
    d5["erasure_reading_sg"] = {
        v: {
            m: summary[v][m]["reading"]
            for m in ("survival_rep", "survival_rev", "survival_filler")
        }
        for v in ("sg_P0", "sg_pnone", "sg_P1", "sg_C1")
    }
    d5["P1_minus_P0_on_sg"] = {
        m: {
            "delta": contrasts["sg_P1-sg_P0"][m]["delta"],
            "ci": contrasts["sg_P1-sg_P0"][m]["ci"],
        }
        for m in FAMILY_METRICS
    }

    # --- agreement (D6), frozen in ADDENDUM section 6
    S0, S1, CW = "small_P0", "small_P1", "cw_large"
    surv = {v: {u: per[v][u]["survival_rep"] for u in ids} for v in (S0, S1, CW)}
    rule = {v: {u: per[v][u]["rule"] for u in ids} for v in (S0, S1, CW)}
    rec = [u for u in rep_pos if surv[S1][u] == 1 and surv[S0][u] == 0]
    lost = [u for u in rep_pos if surv[S1][u] == 0 and surv[S0][u] == 1]
    both = [u for u in rep_pos if surv[S1][u] == 1 and surv[S0][u] == 1]
    neither = [u for u in rep_pos if surv[S1][u] == 0 and surv[S0][u] == 0]
    d6a = {
        "n_recovered_P1_not_P0": len(rec),
        "n_lost_P1_not_P0": len(lost),
        "n_both": len(both),
        "n_neither": len(neither),
        "cw_survival_given_recovered": sub_rate(g, surv[CW], rec) if rec else None,
        "cw_survival_given_both": sub_rate(g, surv[CW], both) if both else None,
        "cw_survival_given_neither": sub_rate(g, surv[CW], neither)
        if neither
        else None,
        "cw_survival_given_lost": sub_rate(g, surv[CW], lost) if lost else None,
    }
    if rec and len(rec) >= MIN_RECOVERED:
        d6a["reading"] = reading_agree(d6a["cw_survival_given_recovered"]["ci"])
    else:
        d6a["reading"] = f"insufficient (n recovered < {MIN_RECOVERED})"
    # turn-level rule repetitions that P1 shows and P0 lacks (all 1025 units, then split by reference class)
    new_rule = [u for u in ids if rule[S1][u] == 1 and rule[S0][u] == 0]
    d6a_rule = {
        "n_P1_rule_not_P0": len(new_rule),
        "n_on_reference_positives": sum(u in set(rep_pos) for u in new_rule),
        "n_on_reference_negatives": sum(kind[u] == "neg" for u in new_rule),
        "cw_rule_given_P1_not_P0_all": sub_rate(g, rule[CW], new_rule)
        if new_rule
        else None,
        "cw_rule_given_P1_not_P0_negatives": sub_rate(
            g, rule[CW], [u for u in new_rule if kind[u] == "neg"]
        )
        if any(kind[u] == "neg" for u in new_rule)
        else None,
    }
    agree = {}
    for name, (x, y) in {
        "P1_vs_CW": (S1, CW),
        "P0_vs_CW": (S0, CW),
        "P1_vs_P0": (S1, S0),
    }.items():
        for setname, units in (
            ("negatives", neg),
            ("all_units", ids),
            ("reference_positives_rep", rep_pos),
        ):
            agree[f"{name}|rule_rep|{setname}"] = agreement_with_ci(
                [rule[x][u] for u in units],
                [rule[y][u] for u in units],
                [g[u] for u in units],
            )
    # P1-only fires on negatives (neither P0 nor CW fire): candidate prompt insertions
    p1_only_neg = [
        u for u in neg if rule[S1][u] == 1 and rule[S0][u] == 0 and rule[CW][u] == 0
    ]
    fp_paired = paired(S1, S0, ("rule_fp",))["rule_fp"]
    d6b = {
        "negatives": len(neg),
        "P1_fires": int(sum(rule[S1][u] for u in neg)),
        "P0_fires": int(sum(rule[S0][u] for u in neg)),
        "CW_fires": int(sum(rule[CW][u] for u in neg)),
        "P1_fires_and_CW_fires": int(sum(rule[S1][u] & rule[CW][u] for u in neg)),
        "P1_only_neither_P0_nor_CW": len(p1_only_neg),
        "P1_minus_P0_rule_fp_paired": {
            k: fp_paired[k] for k in ("delta", "ci", "p", "gained", "lost")
        },
        "kappa_P1_vs_CW_ci_excludes_0": (
            agree["P1_vs_CW|rule_rep|negatives"]["kappa_ci"][0] is not None
            and agree["P1_vs_CW|rule_rep|negatives"]["kappa_ci"][0] > 0
        ),
    }
    flags = {
        "ids": ids,
        "kind": kind,
        "rep_pos": set(rep_pos),
        "rev_pos": set(rev_pos),
        "fill_pos": fill_pos,
        "per": per,
    }
    return {
        "frame": {
            "units": len(ids),
            "rep_pos": 214,
            "rev_pos": 161,
            "neg": 679,
            "filler_pos": 308,
            "speakers": len(set(g.values())),
        },
        "summary": summary,
        "contrasts": contrasts,
        "holm_family16": sorted(fam),
        "D5_device": d5,
        "D6a_recovered_repetitions": d6a,
        "D6a_rule_level": d6a_rule,
        "D6_agreement_pairs": agree,
        "D6b_negatives": d6b,
        "inputs_sha256": texts_hash,
    }, flags


# --- CHM150 -----------------------------------------------------------------------------------------------------------
def chm():
    fr = json.loads((DEST / "chm_frame.json").read_text("utf8"))
    ids = fr["ids"]
    refs = json.loads((CHM_REF / "reference.json").read_text("utf8"))
    g = {u: refs[u]["speaker"] for u in ids}
    pos = [u for u in ids if refs[u]["rep_literal"] == 1]
    neg = [u for u in ids if refs[u]["rep_literal"] == 0]
    assert (len(pos), len(neg)) == (76, 424)
    cached = {
        u: json.loads((CHM_REF / "runs" / f"{u}.json").read_text("utf8")) for u in ids
    }
    sysmap = {
        "small_P0": lambda u: " ".join(w["text"] for w in cached[u]["asr_P0"]["words"]),
        "small_P1": lambda u: " ".join(w["text"] for w in cached[u]["asr_P1"]["words"]),
    }
    dirs = {
        "small_pnone": DEST / "runs/chm150/small_pnone",
        "lv3_pnone": DEST / "runs/chm150/lv3_pnone",
        "cw_large": DEST / "runs/chm150/cw_large",
    }
    if DRYRUN:
        dirs = {k: None for k in dirs}
        for k in dirs:
            sysmap[k] = sysmap["small_P0"]
    else:
        for k, d in dirs.items():
            miss = [u for u in ids if not (d / f"{u}.json").exists()]
            if miss:
                raise SystemExit(f"CHM150 {k}: {len(miss)} units not decoded")
            kind_ = "cw" if k == "cw_large" else "words"
            sysmap[k] = lambda u, d=d, kind_=kind_: load_text(
                json.loads((d / f"{u}.json").read_text("utf8")), kind_
            )
    per = {}
    for v, f in sysmap.items():
        r = {}
        for u in ids:
            t = f(u)
            r[u] = {
                "text": t,
                "surv": int(
                    refs[u]["rep_literal"] == 1
                    and turn_survives("repetition", refs[u]["rep_marks"], t)
                ),
                "rule": int(rule_rep(t)),
            }
        per[v] = r

    def arr(v, m):
        us = pos if m == "survival_rep" else neg
        return (
            us,
            [g[u] for u in us],
            np.array(
                [per[v][u]["surv" if m == "survival_rep" else "rule"] for u in us],
                float,
            ),
            np.ones(len(us)),
        )

    summary = {}
    for v in per:
        s = {}
        for m in ("survival_rep", "rule_fp"):
            us, gg, n, d = arr(v, m)
            r = boot_ratio(gg, n, np.zeros(len(n)), d)
            s[m] = {
                "k": float(n.sum()),
                "n": float(d.sum()),
                "value": r["delta"],
                "ci": r["ci"],
                "n_groups": r["n_groups"],
            }
        s["survival_rep"]["reading"] = tier(s["survival_rep"]["ci"])
        s["empty_outputs"] = int(sum(not per[v][u]["text"].strip() for u in ids))
        summary[v] = s
    fam_pairs = [
        ("small_P1", "small_P0"),
        ("small_pnone", "small_P0"),
        ("lv3_pnone", "small_pnone"),
        ("cw_large", "small_pnone"),
        ("cw_large", "small_P0"),
    ]
    contrasts = {}
    for a, b in fam_pairs:
        out = {}
        for m in ("survival_rep", "rule_fp"):
            us1, g1, n1, d1 = arr(a, m)
            _, _, n0, _ = arr(b, m)
            r = boot_ratio(g1, n1, n0, d1)
            r["gained"] = int(((n1 > 0) & (n0 == 0)).sum())
            r["lost"] = int(((n1 == 0) & (n0 > 0)).sum())
            out[m] = r
        contrasts[f"{a}-{b}"] = out
    fam = {
        f"{a}-{b}|survival_rep": contrasts[f"{a}-{b}"]["survival_rep"]["p"]
        for a, b in fam_pairs
    }
    for k, p in holm(fam, len(fam)).items():
        c, m = k.split("|")
        contrasts[c][m]["p_holm_family5"] = p
    persists = lambda v: (
        summary[v]["survival_rep"]["ci"][1] is not None
        and summary[v]["survival_rep"]["ci"][1] < AM.THRESH_PERSISTS
    )
    n_w = sum(persists(v) for v in ("small_pnone", "lv3_pnone"))
    cwc = contrasts["cw_large-small_pnone"]["survival_rep"]
    cw_att = (
        summary["cw_large"]["survival_rep"]["ci"][0] is not None
        and summary["cw_large"]["survival_rep"]["ci"][0] >= AM.THRESH_PERSISTS
    )
    d7 = {
        "whisper_noprompt_erasure_persists": {
            v: persists(v) for v in ("small_pnone", "lv3_pnone")
        },
        "D7a_whisper": "yes" if n_w == 2 else ("partly" if n_w == 1 else "no"),
        "cw_substantially_attenuated": cw_att,
        "cw_minus_small_pnone_holm": cwc.get("p_holm_family5"),
        "D7b_cw": "yes"
        if (cw_att and cwc["ci"][0] > 0 and cwc["p_holm_family5"] < 0.05)
        else (
            "partly"
            if (cw_att or (cwc["ci"][0] > 0 and cwc["p_holm_family5"] < 0.05))
            else "no"
        ),
    }
    agree = {
        "P1_vs_CW|rule_rep|negatives": agreement_with_ci(
            [per["small_P1"][u]["rule"] for u in neg],
            [per["cw_large"][u]["rule"] for u in neg],
            [g[u] for u in neg],
        )
    }
    flags = {"ids": ids, "per": per, "pos": set(pos)}
    return {
        "frame": {
            "units": len(ids),
            "pos": len(pos),
            "neg_sampled": len(neg),
            "speakers": len(set(g.values())),
            "seed": fr["seed"],
        },
        "summary": summary,
        "contrasts": contrasts,
        "holm_family5": sorted(fam),
        "D7": d7,
        "agreement_P1_vs_CW": agree,
        "inputs_sha256": {
            k: (
                None
                if d is None
                else hashlib.sha256(
                    "".join(sorted(sha_file(d / f"{u}.json") for u in ids)).encode()
                ).hexdigest()
            )
            for k, d in dirs.items()
        },
    }, flags


def main():
    t0 = time.time()
    dih, fl = dihana()
    ch, fc = chm()
    chk = {}
    for name, p in (("sg_P0", DEST / "runs" / "env_check_sg_P0.json"),):
        chk[name] = (
            None
            if (DRYRUN or not p.exists())
            else (lambda j: {"n": j["n"], "identical": j["identical"]})(
                json.loads(p.read_text("utf8"))
            )
        )
    res = {
        "addendum_sha256": None
        if DRYRUN
        else json.loads((DEST / "ADDENDUM.sha256.json").read_text("utf8"))["sha256"],
        "dryrun": bool(DRYRUN),
        "bootstrap": {"seed": SEED, "n_boot": NBOOT, "groups": "speaker"},
        "determinism_checks": chk,
        "dihana": dih,
        "chm150": ch,
        "code_sha256": {
            str(p.relative_to(REPO)).replace("\\", "/"): sha_file(p)
            for p in (
                Path(__file__),
                REPO / "scripts/asr_survival_robustness/decode_small_gpu.py",
                REPO / "scripts/asr_survival_robustness/decode_chm150.py",
                REPO / "scripts/asr_survival_robustness/decode_chm150_crisper.py",
                REPO / "scripts/asr_survival_models/analyze_models.py",
            )
        },
        "seconds": round(time.time() - t0, 1),
    }
    outdir = Path(DRYRUN) if DRYRUN else DEST
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "metrics.json").write_text(
        json.dumps(res, indent=1, default=float), "utf8"
    )
    with (outdir / "unit_flags.csv").open("w", newline="", encoding="utf8") as f:
        w = csv.writer(f)
        names = list(fl["per"])
        w.writerow(
            ["corpus", "unit_id", "kind", "rep_pos"]
            + [f"{n}:{m}" for n in names for m in ("rep", "rev", "fill", "rule")]
        )
        for u in fl["ids"]:
            w.writerow(
                ["dihana", u, fl["kind"][u], int(u in fl["rep_pos"])]
                + [
                    x
                    for n in names
                    for x in (
                        fl["per"][n][u]["survival_rep"],
                        fl["per"][n][u]["survival_rev"],
                        int(fl["per"][n][u]["n_fill"] > 0),
                        fl["per"][n][u]["rule"],
                    )
                ]
            )
        cn = list(fc["per"])
        for u in fc["ids"]:
            w.writerow(
                ["chm150", u, "pos" if u in fc["pos"] else "neg", int(u in fc["pos"])]
                + [
                    x
                    for n in cn
                    for x in (fc["per"][n][u]["surv"], "", "", fc["per"][n][u]["rule"])
                ]
            )
    for v, s in dih["summary"].items():
        print(
            f"{v:12s} rep {s['survival_rep']['k']:.0f}/{s['survival_rep']['n']:.0f}={s['survival_rep']['value']:.3f} {s['survival_rep']['ci']} fill {s['survival_filler']['value']:.3f} rev {s['survival_rev']['value']:.3f}"
        )
    print(
        json.dumps(
            {
                k: dih[k]
                for k in (
                    "D5_device",
                    "D6a_recovered_repetitions",
                    "D6a_rule_level",
                    "D6b_negatives",
                )
            },
            indent=1,
            default=float,
        )
    )
    print(json.dumps(ch["D7"], indent=1))
    for v, s in ch["summary"].items():
        print(
            f"CHM {v:12s} rep {s['survival_rep']['k']:.0f}/{s['survival_rep']['n']:.0f}={s['survival_rep']['value']:.3f} {s['survival_rep']['ci']} {s['survival_rep']['reading']}"
        )


if __name__ == "__main__":
    main()
