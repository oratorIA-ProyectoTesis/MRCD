"""Verbatim profile: adoption gate of the opt-in route ``verbatim_or`` (CACHED CrisperWhisper transcripts; nothing decoded).
Frozen protocol: results/verbatim_profile_20261107/PROTOCOL.md. Frames identical to results/benchmark_empirico_20261107.
usage: python analyze.py [--check]   (--check: load inputs, assert identities, compute no metric)
Writes metrics.json + unit_flags.csv (DIHANA, 0/1 flags only) + chm150_flags.local.csv (local, not committed).
DIHANA fold 5 is never read; no audio is read.
"""

import os

os.environ["OMP_NUM_THREADS"] = "2"
import csv  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

WT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WT))
import numpy as np  # noqa: E402

from mrcd import verbatim as V  # noqa: E402
from mrcd.asr_recovery import stats as S  # noqa: E402
from mrcd.context_eval.metrics import holm  # noqa: E402
from mrcd.context_eval.textfeat import rule_rep  # noqa: E402
from mrcd.optimize.blocks import rule_filler  # noqa: E402

MAIN = Path("/path/to/mrcd")
R = MAIN / "results"
DEST = WT / "results" / "verbatim_profile_20261107"
NBOOT, SEED = 5000, 13
CHECK = "--check" in sys.argv

CW_DIH = R / "asr_survival_models_20261104/runs/asr/cw_large"
CW_CHM = R / "asr_survival_robustness_20261104/runs/chm150/cw_large"
OC = R / "mejora_objetivos_20261101/O_campaign"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read_csv(p):
    return list(csv.DictReader(open(p, encoding="utf-8-sig")))


def cw_text(d, uid):
    return json.loads((Path(d) / f"{uid}.json").read_text("utf8"))["text"]


# ---- frames (same construction as scripts/benchmark_empirico/analyze.py) -------------------------------------------
pr = {r["unit_id"]: r for r in read_csv(OC / "predictions_repetition.csv")}
pf = {r["unit_id"]: r for r in read_csv(OC / "predictions_filler_word.csv")}
s3e1 = {
    r["unit_id"]: r
    for r in read_csv(R / "mejora_objetivos_20261101/S3/e1/predictions.csv")
}
frame_ids = [
    u["unit_id"]
    for u in json.loads((R / "asr_recovery_20261031/subset.json").read_text("utf8"))[
        "frame"
    ]
]
assert len(frame_ids) == 1025 and all(u in pf for u in frame_ids)
assert all(int(pf[u]["fold"]) != 5 for u in frame_ids)
rep_ids = [u for u in frame_ids if u in pr]
assert len(rep_ids) == 1019 and len(pr) == 1019

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


def decisions(ids, cdir, clf_rep=None, clf_fill=None):
    """-> (A classifier, B CW+rule, N route) dicts per class. B is the benchmark decision; N goes through the library route."""
    out = {"A": {}, "B": {}, "N": {}}
    for u in ids:
        raw = cw_text(cdir, u)
        clean = V.strip_tags(raw)
        a = {
            "repetition": clf_rep[u] if clf_rep else 0,
            "filler_word": clf_fill[u] if clf_fill else 0,
        }
        b = {
            "repetition": rule_rep(clean),
            "filler_word": int(bool(rule_filler(clean)) or V.has_filled_pause_tag(raw)),
        }
        n = V.route_decisions(raw, a)
        # identity: the fusion is exactly the OR of the two published decisions
        assert n["repetition"] == int(a["repetition"] or b["repetition"])
        assert n["filler_word"] == int(a["filler_word"] or b["filler_word"])
        out["A"][u], out["B"][u], out["N"][u] = a, b, n
    return out


dih_rep = decisions(rep_ids, CW_DIH, clf_rep={u: int(s3e1[u]["V0"]) for u in rep_ids})
dih_fill = decisions(
    frame_ids, CW_DIH, clf_fill={u: int(pf[u]["decision"]) for u in frame_ids}
)
chm_rep = decisions(
    chm_ids, CW_CHM, clf_rep={u: int(chm_ref[u]["pred_V0_o2026_11"]) for u in chm_ids}
)

if CHECK:
    print("CHECK OK:", len(rep_ids), len(frame_ids), len(chm_ids))
    sys.exit(0)


def col(dec, sysn, cls, ids):
    return np.array([int(dec[sysn][u][cls]) for u in ids], int)


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


def frame(name, ids, dec, cls, y, g, w):
    A, B, N = (col(dec, s, cls, ids) for s in "ABN")
    da = S.paired_wf1_delta(y, N, A, g, w, NBOOT, SEED)  # new route minus MRCD default
    db = S.paired_wf1_delta(
        y, N, B, g, w, NBOOT, SEED
    )  # new route minus CrisperWhisper + rule
    adj = holm([da["p_boot"], db["p_boot"]])  # Holm per frame over {delta_a, delta_b}
    pass_a = bool(da["ci"][0] > 0 and adj[0] < 0.05)
    pass_b = bool(db["ci"][1] >= 0)  # CI of delta_b not entirely below 0
    return {
        "frame": name,
        "class": cls,
        "mrcd_default_A": describe(y, A, g, w),
        "cw_plus_rule_B": describe(y, B, g, w),
        "route_verbatim_or_N": describe(y, N, g, w),
        "delta_a_N_minus_A": {
            "delta": da["delta"],
            "ci95": da["ci"],
            "p_boot": da["p_boot"],
            "p_holm": adj[0],
        },
        "delta_b_N_minus_B": {
            "delta": db["delta"],
            "ci95": db["ci"],
            "p_boot": db["p_boot"],
            "p_holm": adj[1],
        },
        "gate_a_N_above_A": pass_a,
        "gate_b_N_not_below_B": pass_b,
        "gate_frame": bool(pass_a and pass_b),
    }


res = {
    "protocol_sha256": sha(DEST / "PROTOCOL.md"),
    "nboot": NBOOT,
    "seed": SEED,
    "route": V.ROUTE_NAME,
    "route_rule": V.ROUTE_RULE,
    "frames": {},
}
y = np.array([int(pr[u]["y"]) for u in rep_ids])
w = np.array([float(pr[u]["w"]) for u in rep_ids])
g = np.array([pr[u]["speaker"] for u in rep_ids])
res["frames"]["DIH_REP"] = frame(
    "DIHANA folds 1-4, repetition literal, 1019 units, weighted",
    rep_ids,
    dih_rep,
    "repetition",
    y,
    g,
    w,
)
y = np.array([int(pf[u]["y"]) for u in frame_ids])
w = np.ones(len(frame_ids))
g = np.array([pf[u]["speaker"] for u in frame_ids])
res["frames"]["DIH_FILL"] = frame(
    "DIHANA folds 1-4, filler_word, 1025 units, unweighted",
    frame_ids,
    dih_fill,
    "filler_word",
    y,
    g,
    w,
)
y = np.array([int(chm_ref[u]["y_rep_literal_ref"]) for u in chm_ids])
w = np.where(y == 1, 1.0, W_NEG)
g = np.array([chm_ref[u]["speaker"] for u in chm_ids])
res["frames"]["CHM_REP"] = frame(
    "CHM150 sampled frame (76 pos + 424 neg, neg weight 2587/424), repetition literal",
    chm_ids,
    chm_rep,
    "repetition",
    y,
    g,
    w,
)

f = res["frames"]
res["gate_by_class"] = {
    "repetition": bool(f["DIH_REP"]["gate_frame"] and f["CHM_REP"]["gate_frame"]),
    "filler_word": bool(f["DIH_FILL"]["gate_frame"]),
}
res["verdict_by_class"] = {
    c: ("adopted" if ok else "not adopted") for c, ok in res["gate_by_class"].items()
}
res["inputs_sha256"] = {
    "predictions_repetition.csv": sha(OC / "predictions_repetition.csv"),
    "predictions_filler_word.csv": sha(OC / "predictions_filler_word.csv"),
    "S3_e1_predictions.csv": sha(R / "mejora_objetivos_20261101/S3/e1/predictions.csv"),
    "chm_frame.json": sha(R / "asr_survival_robustness_20261104/chm_frame.json"),
    "chm150_predictions.csv": sha(R / "external_chm150_20261102/predictions.csv"),
    "cw_dihana_transcripts_combined": hashlib.sha256(
        b"".join(sha(CW_DIH / f"{u}.json").encode() for u in sorted(frame_ids))
    ).hexdigest(),
    "cw_chm150_transcripts_combined": hashlib.sha256(
        b"".join(sha(CW_CHM / f"{u}.json").encode() for u in sorted(chm_ids))
    ).hexdigest(),
}
(DEST / "metrics.json").write_text(json.dumps(res, indent=1, default=float), "utf8")

with open(DEST / "unit_flags.csv", "w", newline="", encoding="utf8") as fh:
    wr = csv.writer(fh)
    wr.writerow(
        ["frame", "unit_id", "y", "w", "A_mrcd_default", "B_cw_rule", "N_route"]
    )
    for u in rep_ids:
        wr.writerow(
            [
                "DIH_REP",
                u,
                pr[u]["y"],
                pr[u]["w"],
                dih_rep["A"][u]["repetition"],
                dih_rep["B"][u]["repetition"],
                dih_rep["N"][u]["repetition"],
            ]
        )
    for u in frame_ids:
        wr.writerow(
            [
                "DIH_FILL",
                u,
                pf[u]["y"],
                1,
                dih_fill["A"][u]["filler_word"],
                dih_fill["B"][u]["filler_word"],
                dih_fill["N"][u]["filler_word"],
            ]
        )
with open(DEST / "chm150_flags.local.csv", "w", newline="", encoding="utf8") as fh:
    wr = csv.writer(fh)
    wr.writerow(["unit_id", "y", "A", "B", "N"])
    for u in chm_ids:
        wr.writerow(
            [
                u,
                chm_ref[u]["y_rep_literal_ref"],
                chm_rep["A"][u]["repetition"],
                chm_rep["B"][u]["repetition"],
                chm_rep["N"][u]["repetition"],
            ]
        )
print("metrics written")
