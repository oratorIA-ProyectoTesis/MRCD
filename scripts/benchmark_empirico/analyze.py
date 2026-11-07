"""Empirical benchmark of MRCD (service default) against every competitor that runs offline here.
Frozen protocol: results/benchmark_empirico_20261107/PROTOCOL.md. All transcripts are CACHED (nothing decoded here).
Writes metrics.json + unit_flags.csv (DIHANA, 0/1 flags only) + chm150_flags.local.csv (CHM150, local, not committed).
usage: python analyze.py [--check]     (--check: load inputs, assert plumbing identities, compute no metrics)
Inputs are read from the main checkout by absolute path; code comes from this worktree.
"""

import os

os.environ["OMP_NUM_THREADS"] = "4"
import csv  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

WT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WT))
import numpy as np  # noqa: E402

from mrcd.asr_recovery import stats as S  # noqa: E402
from mrcd.context_eval.metrics import holm  # noqa: E402
from mrcd.context_eval.textfeat import rule_rep  # noqa: E402
from mrcd.optimize.blocks import rule_filler  # noqa: E402

MAIN = Path("/path/to/mrcd")
R = MAIN / "results"
DEST = WT / "results" / "benchmark_empirico_20261107"
NBOOT, SEED = 5000, 13
CHECK = "--check" in sys.argv

# ---- cached inputs ------------------------------------------------------------------------------------------------
DIH = {
    "small_P0": R / "mejoras_20261030/asr_cache/dihana",
    "small_P1": R / "asr_recovery_20261031/asr/P1",
    "small_pnone": R / "control_noprompt_20261103/runs/asr/p_none",
    "lv3_P0": R / "asr_survival_models_20261104/runs/asr/lv3_P0",
    "lv3_P1": R / "asr_survival_models_20261104/runs/asr/lv3_P1",
    "lv3_pnone": R / "asr_survival_models_20261104/runs/asr/lv3_p_none",
    "cw_large": R / "asr_survival_models_20261104/runs/asr/cw_large",
}
CHM_ROBUST = R / "asr_survival_robustness_20261104/runs/chm150"
CHM_RUNS = R / "external_chm150_20261102/runs"
TAG_RE = re.compile(r"\[([^\[\]]*)\]")
FP_TAGS = frozenset(
    {"uh", "um", "eh", "em", "er", "ah", "ahh", "hm", "hmm", "mm", "mmm"}
)


def words_text(d):
    return " ".join(w["text"] for w in d["words"])


def cw_strip(raw):
    return " ".join(TAG_RE.sub(" ", raw).split())


def cw_tagtext(raw):
    return " ".join(TAG_RE.sub(lambda m: " " + m.group(1) + " ", raw).split())


def cw_fp_tags(raw):
    return sum(m.group(1).strip().lower() in FP_TAGS for m in TAG_RE.finditer(raw))


def load_text(dirp, uid, kind):
    d = json.loads((Path(dirp) / f"{uid}.json").read_text("utf8"))
    return d["text"] if kind == "cw" else words_text(d)


def dec_rep(text, kind):
    return rule_rep(cw_strip(text) if kind == "cw" else text)


def dec_fill(text, kind):
    if kind == "cw":
        return int(bool(rule_filler(cw_strip(text))) or cw_fp_tags(text) > 0)
    return rule_filler(text)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read_csv(p):
    return list(csv.DictReader(open(p, encoding="utf-8-sig")))


# ---- frames -------------------------------------------------------------------------------------------------------
OC = R / "mejora_objetivos_20261101/O_campaign"
pr = {r["unit_id"]: r for r in read_csv(OC / "predictions_repetition.csv")}
pf = {r["unit_id"]: r for r in read_csv(OC / "predictions_filler_word.csv")}
s3e1 = {
    r["unit_id"]: r
    for r in read_csv(R / "mejora_objetivos_20261101/S3/e1/predictions.csv")
}
s3e2 = {
    r["unit_id"]: r
    for r in read_csv(R / "mejora_objetivos_20261101/S3/e2/predictions_filler.csv")
}
frame_ids = [
    u["unit_id"]
    for u in json.loads((R / "asr_recovery_20261031/subset.json").read_text("utf8"))[
        "frame"
    ]
]
assert len(frame_ids) == 1025 and all(u in pf for u in frame_ids)
assert all(
    int(pf[u]["fold"]) != 5 for u in frame_ids
)  # the pool is folds 1-4 by construction
rep_ids = [u for u in frame_ids if u in pr]
assert len(rep_ids) == 1019 and len(pr) == 1019

opaf = {}
opaf_failed = set()
if (DEST / "runs" / "opaf_flags.csv").exists():
    # ANNEX 1: a detector crash (PraatError) is "no detection" = predicted negative; crashes are counted and a
    # ran-only sensitivity is reported.
    _orows = read_csv(DEST / "runs" / "opaf_flags.csv")
    opaf = {r["unit_id"]: (int(int(r["nrFP"]) >= 1) if r["nrFP"] != "NA" else 0) for r in _orows}
    opaf_failed = {r["unit_id"] for r in _orows if r["nrFP"] == "NA"}

SYSTEMS_DIH = [
    "small_P0",
    "small_P1",
    "small_pnone",
    "lv3_P0",
    "lv3_P1",
    "lv3_pnone",
    "cw_large",
]
kind_of = {s: ("cw" if s == "cw_large" else "words") for s in SYSTEMS_DIH}

txt = {s: {u: load_text(DIH[s], u, kind_of[s]) for u in frame_ids} for s in SYSTEMS_DIH}
rep_dec = {
    s: {u: dec_rep(txt[s][u], kind_of[s]) for u in frame_ids} for s in SYSTEMS_DIH
}
fill_dec = {
    s: {u: dec_fill(txt[s][u], kind_of[s]) for u in frame_ids} for s in SYSTEMS_DIH
}
rep_dec["cw_large_tagtext"] = {
    u: rule_rep(cw_tagtext(txt["cw_large"][u])) for u in frame_ids
}

# sanity identities (plumbing only; both comparators were already published in the O/S3 campaigns)
assert all(rep_dec["small_P0"][u] == int(s3e1[u]["rule_P0"]) for u in rep_ids), (
    "rule_P0 plumbing mismatch"
)
assert all(rep_dec["small_P1"][u] == int(s3e1[u]["rule_P1"]) for u in rep_ids), (
    "rule_P1 plumbing mismatch"
)
assert all(
    fill_dec["small_P0"][u] == int(s3e2[u]["rule_asr_lexical"]) for u in frame_ids
), "filler rule mismatch"
assert all(int(pf[u]["y"]) == int(s3e2[u]["y"]) for u in frame_ids)

# ---- CHM150 frame -------------------------------------------------------------------------------------------------
chm = json.loads(
    (R / "asr_survival_robustness_20261104/chm_frame.json").read_text("utf8")
)
chm_ids = chm["ids"]
chm_ref = {
    r["utt_id"]: r for r in read_csv(R / "external_chm150_20261102/predictions.csv")
}
assert (
    len(chm_ids) == 500
    and sum(int(chm_ref[u]["y_rep_literal_ref"]) for u in chm_ids) == 76
)
N_NEG_ALL = sum(int(r["y_rep_literal_ref"]) == 0 for r in chm_ref.values())
assert N_NEG_ALL == 2587
W_NEG = N_NEG_ALL / 424.0


def chm_text(s, u):
    if s in ("small_P0", "small_P1"):
        d = json.loads((CHM_RUNS / f"{u}.json").read_text("utf8"))
        return words_text(d["asr_P0" if s == "small_P0" else "asr_P1"])
    d = json.loads((CHM_ROBUST / s / f"{u}.json").read_text("utf8"))
    return d["text"] if s == "cw_large" else words_text(d)


SYSTEMS_CHM = ["small_P0", "small_P1", "small_pnone", "lv3_pnone", "cw_large"]
chm_dec = {
    s: {u: dec_rep(chm_text(s, u), kind_of.get(s, "words")) for u in chm_ids}
    for s in SYSTEMS_CHM
}
chm_dec["cw_large_tagtext"] = {
    u: rule_rep(cw_tagtext(chm_text("cw_large", u))) for u in chm_ids
}
assert all(
    chm_dec["small_P0"][u] == int(chm_ref[u]["pred_rule_P0"]) for u in chm_ids
), "CHM rule_P0 mismatch"
assert all(
    chm_dec["small_P1"][u] == int(chm_ref[u]["pred_rule_P1"]) for u in chm_ids
), "CHM rule_P1 mismatch"

if CHECK:
    print(
        "CHECK OK: frames",
        len(frame_ids),
        len(rep_ids),
        len(chm_ids),
        "opaf",
        len(opaf),
    )
    sys.exit(0)


# ---- metrics ------------------------------------------------------------------------------------------------------
def arr(ids, d):
    return np.array([d[u] for u in ids], int)


def describe(y, p, g, w):
    m = S.wconfusion(y, p, w)
    return {
        "tp": int(((y == 1) & (p == 1)).sum()),
        "fp": int(((y == 0) & (p == 1)).sum()),
        "fn": int(((y == 1) & (p == 0)).sum()),
        "tn": int(((y == 0) & (p == 0)).sum()),
        "w_precision": m["w_precision"],
        "w_recall": m["w_recall"],
        "w_f1": m["w_f1"],
        "f1_ci95": S.boot_wf1_ci(y, p, g, w, NBOOT, SEED),
        "n": int(len(y)),
        "pos": int(y.sum()),
    }


def verdict(ci):
    lo, hi = ci
    return (
        "MRCD wins" if lo > 0 else ("MRCD loses" if hi < 0 else "tie (CI includes 0)")
    )


def frame_block(name, ids, y, g, w, mrcd, comps, refs):
    out = {
        "frame": name,
        "n": len(ids),
        "pos": int(y.sum()),
        "n_groups": len(set(g.tolist())),
        "mrcd_service_default": describe(y, mrcd, g, w),
        "competitors": {},
        "mrcd_reference_routes": {},
        "sensitivity": {},
    }
    fam = []
    for s, p in comps["family"].items():
        d = S.paired_wf1_delta(y, mrcd, p, g, w, NBOOT, SEED)  # MRCD minus competitor
        fam.append((s, d))
        out["competitors"][s] = {
            **describe(y, p, g, w),
            "delta_mrcd_minus_competitor": d["delta"],
            "delta_ci95": d["ci"],
            "p_boot": d["p_boot"],
        }
    adj = holm([d["p_boot"] for _, d in fam])
    for (s, d), a in zip(fam, adj):
        out["competitors"][s]["p_holm"] = a
        out["competitors"][s]["verdict"] = verdict(d["ci"])
        out["competitors"][s]["verdict_holm_ok"] = bool(a < 0.05) or (
            d["ci"][0] <= 0 <= d["ci"][1]
        )
    for s, p in comps.get("sensitivity", {}).items():
        d = S.paired_wf1_delta(y, mrcd, p, g, w, NBOOT, SEED)
        out["sensitivity"][s] = {
            **describe(y, p, g, w),
            "delta_mrcd_minus_competitor": d["delta"],
            "delta_ci95": d["ci"],
        }
    best = max(comps["family"], key=lambda s: out["competitors"][s]["w_f1"])
    out["best_competitor_by_point_f1"] = best
    for s, p in refs.items():
        d = S.paired_wf1_delta(
            y, p, comps["family"][best], g, w, NBOOT, SEED
        )  # reference route minus best
        out["mrcd_reference_routes"][s] = {
            **describe(y, p, g, w),
            "delta_vs_best_competitor": d["delta"],
            "delta_ci95": d["ci"],
        }
    return out


res = {
    "protocol_sha256": sha(DEST / "PROTOCOL.md"),
    "nboot": NBOOT,
    "seed": SEED,
    "delta_convention": "MRCD minus competitor (positive = MRCD better)",
    "frames": {},
}

# F-DIH-REP
y = np.array([int(pr[u]["y"]) for u in rep_ids])
w = np.array([float(pr[u]["w"]) for u in rep_ids])
g = np.array([pr[u]["speaker"] for u in rep_ids])
mr = np.array([int(s3e1[u]["V0"]) for u in rep_ids])
comps = {
    "family": {s: arr(rep_ids, rep_dec[s]) for s in SYSTEMS_DIH},
    "sensitivity": {"cw_large_tagtext": arr(rep_ids, rep_dec["cw_large_tagtext"])},
}
refs = {
    "mrcd_opt_in_O_pipeline_oof": np.array([int(pr[u]["decision"]) for u in rep_ids]),
    "mrcd_opt_in_rep_ctx_p1_A1_oof": np.array(
        [int(s3e1[u]["A1_ctxLR_P1"]) for u in rep_ids]
    ),
}
res["frames"]["DIH_REP"] = frame_block(
    "DIHANA folds 1-4, repetition literal, 1019 units, weighted",
    rep_ids,
    y,
    g,
    w,
    mr,
    comps,
    refs,
)

# F-DIH-FILL
y = np.array([int(pf[u]["y"]) for u in frame_ids])
w = np.ones(len(frame_ids))
g = np.array([pf[u]["speaker"] for u in frame_ids])
mf = np.array([int(pf[u]["decision"]) for u in frame_ids])
fam = {s: arr(frame_ids, fill_dec[s]) for s in SYSTEMS_DIH}
if opaf:
    fam["openopaf_praat"] = np.array([opaf[u] for u in frame_ids])
refs = {
    "mrcd_library_v0_argmax": np.array([int(s3e2[u]["V0_argmax"]) for u in frame_ids])
}
res["frames"]["DIH_FILL"] = frame_block(
    "DIHANA folds 1-4, repetition-sampled frame, filler_word, 1025 units, unweighted",
    frame_ids,
    y,
    g,
    w,
    mf,
    {"family": fam},
    refs,
)

if opaf:
    ok = np.array([u not in opaf_failed for u in frame_ids])
    dd = S.paired_wf1_delta(y[ok], mf[ok], fam["openopaf_praat"][ok], g[ok], w[ok], NBOOT, SEED)
    res["frames"]["DIH_FILL"]["sensitivity"]["openopaf_ran_only"] = {
        **describe(y[ok], fam["openopaf_praat"][ok], g[ok], w[ok]),
        "n_failed_excluded": int((~ok).sum()),
        "mrcd_f1_same_units": describe(y[ok], mf[ok], g[ok], w[ok])["w_f1"],
        "delta_mrcd_minus_competitor": dd["delta"],
        "delta_ci95": dd["ci"],
    }
    res["frames"]["DIH_FILL"]["openopaf_failed_turns"] = len(opaf_failed)

# F-CHM-REP
y = np.array([int(chm_ref[u]["y_rep_literal_ref"]) for u in chm_ids])
w = np.where(y == 1, 1.0, W_NEG)
g = np.array([chm_ref[u]["speaker"] for u in chm_ids])
mc = np.array([int(chm_ref[u]["pred_V0_o2026_11"]) for u in chm_ids])
comps = {
    "family": {s: arr(chm_ids, chm_dec[s]) for s in SYSTEMS_CHM},
    "sensitivity": {"cw_large_tagtext": arr(chm_ids, chm_dec["cw_large_tagtext"])},
}
refs = {
    "mrcd_opt_in_rep_ctx_p1": np.array(
        [int(chm_ref[u]["pred_rep_ctx_p1"]) for u in chm_ids]
    )
}
res["frames"]["CHM_REP"] = frame_block(
    "CHM150 sampled frame (76 pos + 424 neg, neg weight 2587/424), repetition literal",
    chm_ids,
    y,
    g,
    w,
    mc,
    comps,
    refs,
)

res["inputs_sha256"] = {
    "predictions_repetition.csv": sha(OC / "predictions_repetition.csv"),
    "predictions_filler_word.csv": sha(OC / "predictions_filler_word.csv"),
    "S3_e1_predictions.csv": sha(R / "mejora_objetivos_20261101/S3/e1/predictions.csv"),
    "S3_e2_predictions_filler.csv": sha(
        R / "mejora_objetivos_20261101/S3/e2/predictions_filler.csv"
    ),
    "chm_frame.json": sha(R / "asr_survival_robustness_20261104/chm_frame.json"),
    "chm150_predictions.csv": sha(R / "external_chm150_20261102/predictions.csv"),
}
(DEST / "metrics.json").write_text(json.dumps(res, indent=1, default=float), "utf8")

# 0/1 flags only, no text
sysn = SYSTEMS_DIH + ["cw_large_tagtext"]
with open(DEST / "unit_flags.csv", "w", newline="", encoding="utf8") as f:
    wr = csv.writer(f)
    wr.writerow(
        ["frame", "unit_id", "y", "w", "mrcd_default"] + [f"rule:{s}" for s in sysn]
    )
    for u in rep_ids:
        wr.writerow(
            ["DIH_REP", u, pr[u]["y"], pr[u]["w"], s3e1[u]["V0"]]
            + [rep_dec[s][u] for s in sysn]
        )
    for u in frame_ids:
        wr.writerow(
            ["DIH_FILL", u, pf[u]["y"], 1, pf[u]["decision"]]
            + [fill_dec[s][u] for s in SYSTEMS_DIH]
            + ([opaf[u]] if opaf else [])
        )
with open(DEST / "chm150_flags.local.csv", "w", newline="", encoding="utf8") as f:
    wr = csv.writer(f)
    wr.writerow(["unit_id", "y", "mrcd_V0"] + [f"rule:{s}" for s in SYSTEMS_CHM])
    for u in chm_ids:
        wr.writerow(
            [u, chm_ref[u]["y_rep_literal_ref"], chm_ref[u]["pred_V0_o2026_11"]]
            + [chm_dec[s][u] for s in SYSTEMS_CHM]
        )
print("metrics written")
