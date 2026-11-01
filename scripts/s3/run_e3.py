"""E3: cross-fitted calibration (Platt / isotonic), abstention band, coverage/risk. Protocol: S3/PROTOCOL.md section 4.

Also writes the frozen development artifact used by the service (calibrators + thresholds + bands fitted on ALL pooled
development scores of folds 1-4)."""

import s3_common as K  # noqa: F401

import json
import pickle
import time

import numpy as np

from mrcd import calibration as C
from mrcd.fusion_text import best_threshold_w

METHODS = ("platt", "isotonic")


def sources():
    pool = K.load_pool()
    outer = K.outer_folds(pool)
    base, _ = K.v0_vc(pool)
    T = K.turns_meta()
    ids = [u["unit_id"] for u in pool]
    g = np.array([u["speaker"] for u in pool])
    one = np.ones(len(ids))
    S = {}
    for c, col in (
        ("filler_word", "filler_present"),
        ("prolongation", "lengthening_present"),
    ):
        y = np.array([int(T[u][col]) for u in ids])
        S[f"V0_{c}"] = dict(
            y=y,
            w=one,
            g=g,
            fo=outer,
            s=np.array([base[u][f"v0s_{c}"] for u in ids]),
            tr=None,
        )
    ok = np.array([u["ref"]["rep_literal"] is not None for u in pool])
    y = np.array([int(u["ref"]["rep_literal"] or 0) for u in pool])
    S["V0_repetition"] = dict(
        y=y[ok],
        w=one[ok],
        g=g[ok],
        fo=outer[ok],
        s=np.array([base[u]["v0s_repetition"] for u in ids])[ok],
        tr=None,
    )
    e1 = pickle.loads((K.S3 / "e1" / "_store.pkl").read_bytes())
    pos = {u: i for i, u in enumerate(e1["ids"])}
    tr = {
        f: {"idx": np.array([pos[u] for u in d["tr_ids"]]), "inner": d["inner_oof"]}
        for f, d in e1["train"].items()
    }
    S["A1_text_P1_repetition"] = dict(
        y=e1["y"], w=e1["w"], g=e1["g"], fo=e1["fo"], s=e1["a1s"], tr=tr
    )
    e2 = pickle.loads((K.S3 / "e2" / "_store.pkl").read_bytes())
    tr = {f: {"idx": d["tr"], "inner": d["inner_oof"]} for f, d in e2["store"].items()}
    S["STACK_filler_word"] = dict(
        y=e2["y"], w=one, g=e2["g"], fo=e2["fo"], s=e2["stack"], tr=tr
    )
    return S


def train_scores(d, f, tr_idx):
    """Scores on the outer-train rows: raw for score sources, inner-OOF for learned models."""
    if d["tr"] is None:
        return d["s"][tr_idx], tr_idx
    t = d["tr"][f]
    return t["inner"], t["idx"]


def run_source(name, d):
    y, w, g, fo, s = d["y"], d["w"], d["g"], d["fo"], d["s"]
    res = {
        "n": int(len(y)),
        "pos": int(y.sum()),
        "pos_speakers": int(len(set(g[y == 1]))),
        "methods": {},
    }
    cal = {m: np.full(len(y), np.nan) for m in METHODS}
    st = {m: np.empty(len(y), object) for m in METHODS}
    st_ref = np.empty(len(y), object)
    bands = {m: [] for m in METHODS}
    for f in sorted(set(fo.tolist())):
        te = np.flatnonzero(fo == f)
        tr_all = np.flatnonzero(fo != f)
        s_tr, idx = train_scores(d, f, tr_all)
        ok = np.isfinite(s_tr)
        s_tr, idx = s_tr[ok], idx[ok]
        for m in METHODS:
            c = C.fit_calibrator(s_tr, y[idx], m, w[idx])
            cal[m][te] = c.apply(s[te])
            cf = C.crossfit(s_tr, y[idx], g[idx], m, 3, K.SEED, w[idx])
            good = np.isfinite(cf)
            thr = best_threshold_w(cf[good], y[idx][good], w[idx][good])
            b = C.choose_band(cf[good], y[idx][good], thr, w[idx][good])
            bands[m].append({"fold": int(f), "thr": thr, **b})
            st[m][te] = C.states(cal[m][te], b["lo"], b["hi"])
            if m == "platt":
                st_ref[te] = C.states(cal[m][te], thr, thr)
    res["no_abstention_reference_platt"] = C.selective_report(y, st_ref, w)
    for m in METHODS:
        r = {
            "brier_raw": C.brier(y, s, w),
            "brier_cal": C.brier(y, cal[m], w),
            "ece_q10_raw": C.ece_quantile(y, s, 10, w),
            "ece_q10_cal": C.ece_quantile(y, cal[m], 10, w),
            "selective": C.selective_report(y, st[m], w),
            "bands": bands[m],
            "n_bands_feasible": int(sum(b["feasible"] for b in bands[m])),
            "reliability_cal": C.reliability_quantile(y, cal[m], 10, w),
            "reliability_raw": C.reliability_quantile(y, s, 10, w),
        }
        r["paired_brier_cal_minus_raw"] = C.paired_metric_delta(
            lambda a, b, ww: C.brier(a, b, ww), y, cal[m], s, g, w
        )
        r["paired_ece_cal_minus_raw"] = C.paired_metric_delta(
            lambda a, b, ww: C.ece_quantile(a, b, 10, ww), y, cal[m], s, g, w
        )
        res["methods"][m] = r
    return res


def final_artifact(S):
    """Calibrator / threshold / band on ALL pooled development scores (the frozen service artifact)."""
    art = {}
    for name, d in S.items():
        y, w, g, s = d["y"], d["w"], d["g"], d["s"]
        if (
            d["tr"] is not None
        ):  # learned model: its pooled OOF scores are the development scores
            pass
        out = {}
        for m in METHODS:
            c = C.fit_calibrator(s, y, m, w)
            cf = C.crossfit(s, y, g, m, 3, K.SEED, w)
            thr = best_threshold_w(cf, y, w)
            b = C.choose_band(cf, y, thr, w)
            out[m] = {
                "calibrator": c.to_dict(),
                "threshold": thr,
                **b,
                "n": int(len(y)),
            }
        art[name] = out
    return art


def main():
    t0 = time.time()
    S = sources()
    res = {"sources": {}}
    fam = []
    for name, d in S.items():
        r = run_source(name, d)
        res["sources"][name] = r
        for m in METHODS:
            t = dict(r["methods"][m]["paired_brier_cal_minus_raw"])
            t["test"] = f"E3: Brier({m}) minus Brier(raw), {name}"
            fam.append(t)
    res["family_E3"] = K.holm_table(fam)
    for name, r in res["sources"].items():
        for m in METHODS:
            e = r["methods"][m]["paired_ece_cal_minus_raw"]
            r["methods"][m]["calibration_wording"] = (
                "calibration improvement shown (ECE CI below 0)"
                if e["ci"][1] < 0
                else "score, not demonstrated calibrated"
            )
    art = final_artifact(S)
    res["config"] = {
        "protocol_sha256": json.loads((K.S3 / "PROTOCOL.sha256.json").read_text())[
            "sha256"
        ],
        "seconds": round(time.time() - t0, 1),
    }
    out = K.S3 / "e3"
    out.mkdir(exist_ok=True)
    (out / "results.json").write_text(json.dumps(res, indent=1, default=float), "utf8")
    (out / "dev_artifact.json").write_text(
        json.dumps(art, indent=1, default=float), "utf8"
    )
    print("done", res["config"]["seconds"])


if __name__ == "__main__":
    main()
