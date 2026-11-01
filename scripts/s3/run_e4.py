"""E4: observable lengthening (prolongation) and pause-associated false positives. Protocol: S3/PROTOCOL.md section 5."""

import s3_common as K  # noqa: F401

import json
import time

import numpy as np

from mrcd import fusion_text as F
from mrcd.asr_recovery import signals as SG
from mrcd.context_eval import metrics as M
from mrcd.context_eval.partition import group_folds
from mrcd.decision import best_threshold


def boot_count_delta(p1, p0, mask, g, n_boot=5000, seed=13):
    """Delta of the number of FP units in `mask` (fused minus base), speaker bootstrap, two-sided p."""
    ug, inv = np.unique(g.astype(str), return_inverse=True)
    a = np.zeros(len(ug))
    b = np.zeros(len(ug))
    np.add.at(a, inv, p1 * mask)
    np.add.at(b, inv, p0 * mask)
    d0 = float(a.sum() - b.sum())
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(ug), (n_boot, len(ug)))
    ds = a[idx].sum(1) - b[idx].sum(1)
    B = len(ds)
    p = min(
        1.0, 2 * min((np.sum(ds <= 0) + 1) / (B + 1), (np.sum(ds >= 0) + 1) / (B + 1))
    )
    return {
        "delta": d0,
        "ci": [float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5))],
        "p_boot": float(p),
        "fused": float(a.sum()),
        "base": float(b.sum()),
    }


def sig_matrix(recs, tr):
    st = SG.SignalStats.fit([recs[i] for i in tr])
    return np.array([SG.signal_features(r, st) for r in recs]), st


def stack_cv(score, recs, y, g, fo):
    n = len(y)
    dec, sc = np.zeros(n, int), np.zeros(n)
    thr_dec = np.zeros(n, int)
    info = []
    for f in sorted(set(fo.tolist())):
        te, tr = np.flatnonzero(fo == f), np.flatnonzero(fo != f)
        assert not (set(g[te]) & set(g[tr]))
        S, _ = sig_matrix(recs, tr)  # statistics from TRAIN rows only
        s = np.clip(score, 1e-4, 1 - 1e-4)
        X = np.column_stack([np.log(s / (1 - s)), S])
        sel = F.select_fit_w(X[tr], y[tr], g[tr])
        sc[te] = sel["model"].score(X[te])
        dec[te] = (sc[te] >= sel["threshold"]).astype(int)
        t, _ = best_threshold(score[tr], y[tr])
        thr_dec[te] = (score[te] >= t).astype(int)
        info.append(
            {"fold": int(f), "C": sel["C"], "thr": sel["threshold"], "score_thr": t}
        )
    return dec, sc, thr_dec, info


def describe(y, p, g, pmask):
    m = M.confusion(y, p)
    m["n_groups"] = int(len(set(g)))
    m["n_pos_groups"] = int(len({a for a, b in zip(g, y) if b}))
    m["f1_ci95"] = M.boot_f1_ci(y, p, g, 2000, K.SEED)
    m["fp_with_pause_mark"] = int(((p == 1) & (y == 0) & pmask).sum())
    m["fp_without_pause_mark"] = int(((p == 1) & (y == 0) & ~pmask).sum())
    return m


def main():
    t0 = time.time()
    pool = K.load_pool()
    outer = K.outer_folds(pool)
    base, _ = K.v0_vc(pool)
    T = K.turns_meta()
    ids = [u["unit_id"] for u in pool]
    y = np.array([int(T[u]["lengthening_present"]) for u in ids])
    g = np.array([u["speaker"] for u in pool])
    pmask = np.array(
        [
            "p" in {k.split(":")[0] for k in json.loads(T[u]["raw_tag_counts"])}
            for u in ids
        ]
    )
    recs = []
    for u in ids:
        j = json.loads(
            (
                K.REPO / "results/mejoras_20261030/asr_cache/dihana" / f"{u}.json"
            ).read_text("utf8")
        )
        recs.append({"words": j["words"], "n_samples": j["n_samples"]})
    v0s = np.array([base[u]["v0s_prolongation"] for u in ids])
    v0 = np.array([base[u]["v0_prolongation"] for u in ids])
    dec, sc, thr_dec, info = stack_cv(v0s, recs, y, g, outer)
    res = {
        "n": len(y),
        "pos": int(y.sum()),
        "pos_speakers": int(len({a for a, b in zip(g, y) if b})),
        "units_with_pause_mark": int(pmask.sum()),
        "folds_P_a": info,
    }
    pa = {"V0_argmax": v0, "V0thr": thr_dec, "P_a_stack_V0_signals": dec}
    res["folds_1_4"] = {k: describe(y, p, g, pmask) for k, p in pa.items()}
    fam = []
    d = M.paired_f1_delta(y, dec, thr_dec, g, 5000, K.SEED)
    d["test"] = "E4: P-a minus V0thr, F1"
    fam.append(d)
    d = boot_count_delta(dec * (y == 0), thr_dec * (y == 0), pmask, g)
    d["test"] = "E4: P-a minus V0thr, pause-associated FP count"
    fam.append(d)
    # fold 4 only, Vc-favourable
    idx = np.flatnonzero(np.array([u["fold"] == 4 for u in pool]))
    vcs = np.array([base[ids[i]]["vcs_prolongation"] for i in idx])
    vcd = np.array([base[ids[i]]["vc_prolongation"] for i in idx])
    sub = [recs[i] for i in idx]
    fo4 = group_folds(g[idx], 5, K.SEED)
    dec4, sc4, thr4, info4 = stack_cv(vcs, sub, y[idx], g[idx], fo4)
    pb = {
        "Vc_frozen": vcd,
        "Vcthr": thr4,
        "V0_argmax": v0[idx],
        "P_b_stack_Vc_signals": dec4,
    }
    res["fold4_vc"] = {
        "n": int(len(idx)),
        "pos": int(y[idx].sum()),
        "folds": info4,
        "note": "Vc-favourable, nested CV inside fold 4",
        "models": {k: describe(y[idx], p, g[idx], pmask[idx]) for k, p in pb.items()},
    }
    d = M.paired_f1_delta(y[idx], dec4, vcd, g[idx], 5000, K.SEED)
    d["test"] = "E4: P-b minus Vc, F1"
    fam.append(d)
    d = boot_count_delta(dec4 * (y[idx] == 0), vcd * (y[idx] == 0), pmask[idx], g[idx])
    d["test"] = "E4: P-b minus Vc, pause-associated FP count"
    fam.append(d)
    res["family_E4"] = K.holm_table(fam)
    res["pause_share_of_fp"] = {
        k: (v["fp_with_pause_mark"] / max(v["fp"], 1))
        for k, v in res["folds_1_4"].items()
    }
    res["observable_lengthening_note"] = (
        "label is an observable-lengthening mark; no claim that it is disfluent"
    )
    res["config"] = {
        "protocol_sha256": json.loads((K.S3 / "PROTOCOL.sha256.json").read_text())[
            "sha256"
        ],
        "seconds": round(time.time() - t0, 1),
    }
    out = K.S3 / "e4"
    out.mkdir(exist_ok=True)
    (out / "results.json").write_text(json.dumps(res, indent=1, default=float), "utf8")
    print("done", res["config"]["seconds"])


if __name__ == "__main__":
    main()
