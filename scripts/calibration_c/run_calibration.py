"""Calibration + abstention study (protocol: results/calibracion_20261107/PROTOCOL.md, frozen before this ran).

Run from the worktree with MRCD_REPO_ROOT pointing at the checkout that holds the git-ignored data:
    MRCD_REPO_ROOT=/path/to/mrcd python scripts/calibration_c/run_calibration.py
DIHANA folds 1-4 only (``common.load_pool`` asserts it); fold 5 is never read. Aggregates only.
"""

import os

os.environ["OMP_NUM_THREADS"] = "4"
os.environ.setdefault("MRCD_REPO_ROOT", r"/path/to/mrcd")
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from mrcd.asr_recovery import common as RC
from mrcd.calibration import (
    brier,
    crossfit,
    ece_quantile,
    fit_calibrator,
    reliability_quantile,
)
from mrcd.constants import LABEL2ID
from mrcd.optimize import data as OD

HERE = Path(__file__).resolve().parents[2]
OUT = HERE / "results" / "calibracion_20261107"
SEED = 13
B_FULL, B_INNER = 2000, 500
METHODS = ("none", "platt", "isotonic")
CLASSES = ("filler_word", "prolongation", "repetition")
LABEL_COL = {"filler_word": "filler_present", "prolongation": "lengthening_present"}
GRID = np.round(np.arange(0.02, 0.981, 0.02), 4)
BONF = (1.25, 98.75)  # 2 candidate methods per class
ECE_BINS = 10


# ------------------------------------------------------------------ data
def load_class(cls, feats, pool, turns):
    idx, y, w = [], [], []
    if cls == "repetition":
        fr = json.loads((RC.OUT / "subset.json").read_text("utf8"))["frame"]
        by = {u["unit_id"]: i for i, u in enumerate(pool)}
        for s in fr:
            i = by[s["unit_id"]]
            v = pool[i]["ref"]["rep_literal"]
            if v is not None:
                idx.append(i), y.append(int(v)), w.append(float(s["w"]))
    else:
        for i, u in enumerate(pool):
            (
                idx.append(i),
                y.append(int(turns[u["unit_id"]][LABEL_COL[cls]])),
                w.append(1.0),
            )
    folds = RC.outer_folds(pool)
    c = LABEL2ID[cls]
    s, ev, v0 = [], [], []
    for i in idx:
        p = np.asarray(feats[pool[i]["unit_id"]]["probs"], float)
        s.append(float(p[:, c].max()))
        am = p.argmax(1) == c
        v0.append(int(am.any()))
        ev.append(float(p[am, c].max()) if am.any() else np.nan)
    ix = np.array(idx)
    return dict(
        y=np.array(y),
        w=np.array(w),
        s=np.array(s),
        ev=np.array(ev),
        v0=np.array(v0),
        spk=np.array([pool[i]["speaker"] for i in idx]),
        fold=folds[ix],
    )


RUNS = RC.REPO / "results" / "mejora_objetivos_20261101" / "O_campaign" / "runs"


def attach_o_scores(c, cls):
    """Replace the V0 score by the O-pipeline score of the service default (annex 1): stored outer-test scores
    plus, per outer fold, the inner out-of-fold scores of the stored winning configuration (re-derived)."""
    import pickle

    from mrcd.optimize import search as SE
    from mrcd.optimize import space as SP

    d = OD.load_dihana(cls)
    assert np.array_equal(d.y, c["y"]) and np.array_equal(d.g, c["spk"])
    s = np.full(len(d.y), np.nan)
    dec = np.zeros(len(d.y), int)
    trs, maxdiff = {}, {}
    for k in sorted(set(c["fold"].tolist())):
        r = pickle.loads((RUNS / cls / f"fold{k}.pkl").read_bytes())
        cfg = SP.Config.from_dict(r["config"])
        tr = np.flatnonzero(d.outer != k)
        inner = SE.inner_oof(cfg, d, tr, SP.SEED + 1000 * (k + 1))
        m, X = SE.fit_config(d, tr, cfg)
        re = np.asarray(m.score(X[r["te"]]), float)
        maxdiff[int(k)] = float(np.max(np.abs(re - r["score"])))
        s[r["te"]], dec[r["te"]], trs[int(k)] = r["score"], r["decision"], inner
        print(cls, "fold", k, "reproduction max|diff|", maxdiff[int(k)], flush=True)
    c.update(s=s, trs=trs, svc_dec=dec, o_reproduction_maxdiff=maxdiff)
    return c


# ------------------------------------------------------------------ metrics
def wf1(y, pred, w):
    y = np.asarray(y).astype(bool)
    pred = np.asarray(pred).astype(bool)
    tp = w[y & pred].sum()
    fp = w[~y & pred].sum()
    fn = w[y & ~pred].sum()
    d = 2 * tp + fp + fn
    return float(2 * tp / d) if d else 0.0


def best_t(s, y, w):
    fs = np.array([wf1(y, s >= t, w) for t in GRID])
    top = np.flatnonzero(fs >= fs.max() - 1e-12)
    return float(GRID[top[len(top) // 2]])


def speaker_members(spk):
    ug, inv = np.unique(spk, return_inverse=True)
    return ug, [np.flatnonzero(inv == i) for i in range(len(ug))]


def boot_indices(spk, B, seed=SEED):
    _, mem = speaker_members(spk)
    rng = np.random.default_rng(seed)
    return [
        np.concatenate([mem[k] for k in rng.integers(0, len(mem), len(mem))])
        for _ in range(B)
    ]


def ci(vals, pct=(2.5, 97.5)):
    return [float(np.percentile(vals, pct[0])), float(np.percentile(vals, pct[1]))]


# ------------------------------------------------------------------ calibration (OOF, speaker-disjoint)
def oof_calibrated(s, y, w, spk, fold, method, trs=None):
    """Pooled OOF calibrated scores, OOF decisions at train-chosen thresholds, per-fold thresholds."""
    p = np.full(len(s), np.nan)
    dec = np.zeros(len(s), bool)
    thr = {}
    for k in sorted(set(fold.tolist())):
        tr, te = fold != k, fold == k
        assert not (set(spk[tr]) & set(spk[te]))
        if method == "none":
            ptr, pte = (s[tr] if trs is None else trs[int(k)]), s[te]
        else:
            st = s[tr] if trs is None else trs[int(k)]
            ptr = crossfit(st, y[tr], spk[tr], method, k=3, seed=SEED, w=w[tr])
            pte = fit_calibrator(st, y[tr], method, w[tr]).apply(s[te])
        t = best_t(ptr, y[tr], w[tr])
        p[te], dec[te], thr[int(k)] = pte, pte >= t, t
    return p, dec, thr


def calibration_study(c):
    y, w, s, spk, fold = c["y"], c["w"], c["s"], c["spk"], c["fold"]
    res = {
        m: dict(zip(("p", "dec", "thr"), oof_calibrated(s, y, w, spk, fold, m, c.get('trs'))))
        for m in METHODS
    }
    reps = boot_indices(spk, B_FULL)
    stat = {m: {"ece": [], "brier": [], "f1": []} for m in METHODS}
    for ii in reps:
        for m in METHODS:
            r = res[m]
            stat[m]["ece"].append(ece_quantile(y[ii], r["p"][ii], ECE_BINS, w[ii]))
            stat[m]["brier"].append(brier(y[ii], r["p"][ii], w[ii]))
            stat[m]["f1"].append(wf1(y[ii], r["dec"][ii], w[ii]))
    out = {"methods": {}}
    for m in METHODS:
        r = res[m]
        d = {
            "ece": float(ece_quantile(y, r["p"], ECE_BINS, w)),
            "brier": float(brier(y, r["p"], w)),
            "f1_threshold_rule": wf1(y, r["dec"], w),
            "thresholds_by_outer_fold": r["thr"],
        }
        for k in ("ece", "brier", "f1"):
            d[k + "_ci95"] = ci(stat[m][k])
        if m != "none":
            for k, key in (
                ("ece", "ece"),
                ("brier", "brier"),
                ("f1", "f1_threshold_rule"),
            ):
                dl = np.array(stat[m][k]) - np.array(stat["none"][k])
                d["delta_" + k] = {
                    "delta": d[key] - out["methods"]["none"][key],
                    "ci975": ci(dl, BONF),
                }
        out["methods"][m] = d
    if "svc_dec" in c:
        out["service_pipeline_decision_f1"] = wf1(y, c["svc_dec"], w)
        out["service_pipeline_decision_f1_ci95"] = ci([wf1(y[ii], c["svc_dec"][ii], w[ii]) for ii in reps])
        out["o_reproduction_maxdiff"] = c["o_reproduction_maxdiff"]
    out["v0_argmax_f1"] = wf1(y, c["v0"], w)
    out["v0_argmax_f1_ci95"] = ci([wf1(y[ii], c["v0"][ii], w[ii]) for ii in reps])
    # decision rule (protocol section 5)
    passing = []
    for m in ("platt", "isotonic"):
        d = out["methods"][m]
        a = d["delta_ece"]["ci975"][1] < 0
        b = d["delta_f1"]["ci975"][0] >= -0.01
        d["gate_ece_improves"], d["gate_f1_noninferior"], d["adopt"] = (
            bool(a),
            bool(b),
            bool(a and b),
        )
        if a and b:
            passing.append(m)
    out["adopted"] = (
        min(passing, key=lambda m: out["methods"][m]["brier"]) if passing else None
    )
    out["reliability"] = {
        m: reliability_quantile(y, res[m]["p"], ECE_BINS, w) for m in METHODS
    }
    out["_res"] = res
    return out


def event_study(c, adopted_candidates=("platt", "isotonic")):
    """Secondary: calibrate the event confidence on turns where V0 emits an event (ECE/Brier only)."""
    m_ = ~np.isnan(c["ev"])
    sub = {k: v[m_] for k, v in c.items()}
    y, w, s, spk, fold = sub["y"], sub["w"], sub["ev"], sub["spk"], sub["fold"]
    out = {"n_events_turns": int(m_.sum()), "n_pos": int(y.sum())}
    if len(set(y.tolist())) < 2 or len(set(spk.tolist())) < 5:
        return out
    ps = {}
    for m in METHODS:
        p = np.full(len(s), np.nan)
        for k in sorted(set(fold.tolist())):
            tr, te = fold != k, fold == k
            p[te] = (
                s[te]
                if m == "none"
                else fit_calibrator(s[tr], y[tr], m, w[tr]).apply(s[te])
            )
        ps[m] = p
    reps = boot_indices(spk, B_FULL)
    st = {m: ([], []) for m in METHODS}
    for ii in reps:
        for m in METHODS:
            st[m][0].append(ece_quantile(y[ii], ps[m][ii], ECE_BINS, w[ii]))
            st[m][1].append(brier(y[ii], ps[m][ii], w[ii]))
    for m in METHODS:
        d = {
            "ece": float(ece_quantile(y, ps[m], ECE_BINS, w)),
            "ece_ci95": ci(st[m][0]),
            "brier": float(brier(y, ps[m], w)),
            "brier_ci95": ci(st[m][1]),
        }
        if m != "none":
            d["delta_ece"] = {
                "delta": d["ece"] - out["none"]["ece"],
                "ci975": ci(np.array(st[m][0]) - np.array(st["none"][0]), BONF),
            }
        out[m] = d
    return out


# ------------------------------------------------------------------ abstention
def accept_mask(p, t, d):
    return ((p >= t) & (p - t >= d)) | ((p < t) & (t - p > d))


def margin(p, t):
    return np.where(p >= t, p - t, t - p)


def spk_agg(spk, arrs):
    ug, inv = np.unique(spk, return_inverse=True)
    return [np.bincount(inv, weights=a, minlength=len(ug)) for a in arrs]


def choose_d(p, y, w, spk, t, rng, n_levels=60, target=0.10):
    """Largest coverage whose upper 95% speaker-bootstrap risk <= target (train rows only)."""
    m = margin(p, t)
    ds = np.quantile(m, np.arange(n_levels) / n_levels)
    err = ((p >= t) != y.astype(bool)).astype(float) * w
    ug = np.unique(spk)
    C = rng.multinomial(len(ug), np.ones(len(ug)) / len(ug), size=B_INNER).astype(float)
    best = None
    for d in ds:
        a = accept_mask(p, t, d).astype(float)
        wa, ea = spk_agg(spk, [a * w, a * err])
        wt = spk_agg(spk, [w])[0]
        cov = wa.sum() / wt.sum()
        if wa.sum() == 0:
            continue
        den = C @ wa
        ok = den > 0
        risk_b = (C @ ea)[ok] / den[ok]
        up = np.percentile(risk_b, 97.5)
        if up <= target and (best is None or cov > best[1]):
            best = (float(d), float(cov), float(up))
    return best, ds


def abstention_study(c, method, target=0.10):
    trs = c.get('trs')
    y, w, s, spk, fold = c["y"], c["w"], c["s"], c["spk"], c["fold"]
    p = np.full(len(s), np.nan)
    pte_t, d_by, met = {}, {}, {}
    acc = np.zeros(len(s), bool)
    t_row = np.zeros(len(s))
    rng = np.random.default_rng(SEED + 7)
    levels = {}  # coverage-spaced curve: level j -> accept mask over OOF rows
    NL = 60
    curve_acc = np.zeros((NL, len(s)), bool)
    for k in sorted(set(fold.tolist())):
        tr, te = fold != k, fold == k
        if method == "none":
            ptr, pte = (s[tr] if trs is None else trs[int(k)]), s[te]
        else:
            st = s[tr] if trs is None else trs[int(k)]
            ptr = crossfit(st, y[tr], spk[tr], method, k=3, seed=SEED, w=w[tr])
            pte = fit_calibrator(st, y[tr], method, w[tr]).apply(s[te])
        t = best_t(ptr, y[tr], w[tr])
        best, ds = choose_d(ptr, y[tr], w[tr], spk[tr], t, rng, NL, target)
        d = best[0] if best else None
        p[te], t_row[te] = pte, t
        d_by[int(k)] = {
            "t": t,
            "d": d,
            "train_cov": best[1] if best else None,
            "train_up": best[2] if best else None,
        }
        met[int(k)] = bool(best)
        acc[te] = accept_mask(pte, t, d) if d is not None else True
        # curve: same quantile levels of the train margin applied to the held-out fold
        for j, dj in enumerate(ds):
            curve_acc[j, te] = accept_mask(pte, t, dj)
    pred = p >= t_row
    err = (pred != y.astype(bool)).astype(float)
    reps = boot_indices(spk, B_FULL, SEED + 3)

    def point(mask, ii=None):
        sl = slice(None) if ii is None else ii
        ww, aa, ee, yy, pp = w[sl], mask[sl], err[sl], y[sl], pred[sl]
        wa = ww[aa].sum()
        cov = wa / ww.sum()
        risk = (ww[aa] * ee[aa]).sum() / wa if wa else np.nan
        base = (ww * ee).sum() / ww.sum()
        present = aa & pp
        tp = ww[present & (yy == 1)].sum()
        fp = ww[present & (yy == 0)].sum()
        fn = ww[(yy == 1) & ~present].sum()
        f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
        f1_base = wf1(yy, pp, ww)
        pos_unc = ww[(~aa) & (yy == 1)].sum() / ww[yy == 1].sum()
        return cov, risk, base, f1, f1_base, pos_unc

    pt = point(acc)
    bs = np.array([point(acc, ii) for ii in reps], float)
    names = (
        "coverage",
        "risk",
        "risk_no_abstention",
        "f1_all_cases_uncertain_as_absent",
        "f1_no_abstention",
        "positives_left_uncertain",
    )
    op = {n: {"value": float(pt[i]), "ci95": ci(bs[:, i])} for i, n in enumerate(names)}
    op["risk_upper95"] = float(np.nanpercentile(bs[:, 1], 97.5))
    dr = bs[:, 1] - bs[:, 2]
    op["risk_reduction_vs_no_abstention"] = {
        "delta": float(pt[1] - pt[2]),
        "ci95": ci(dr),
    }
    op["per_outer_fold"] = d_by
    op["target_met_on_train_all_folds"] = all(met.values())
    op["target"] = target
    # risk-coverage curve (held-out), with CI
    curve = []
    for j in range(NL):
        mk = curve_acc[j]
        v = point(mk)
        b = np.array([point(mk, ii)[:2] for ii in reps[:300]], float)
        curve.append(
            {
                "level": j,
                "coverage": float(v[0]),
                "risk": float(v[1]),
                "risk_ci95": ci(b[:, 1])
                if np.isfinite(b[:, 1]).all()
                else [float("nan")] * 2,
            }
        )
    op["curve"] = curve
    # demonstrates value (protocol section 6)
    op["demonstrates_value"] = bool(
        op["risk_upper95"] <= target
        and op["coverage"]["value"] >= 0.50
        and op["risk_reduction_vs_no_abstention"]["ci95"][1] < 0
    )
    return op


# ------------------------------------------------------------------ figures
def figures(results):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 9, "pdf.fonttype": 42, "svg.fonttype": "none"})
    col = {"none": "#555555", "platt": "#1f77b4", "isotonic": "#d95f02"}
    fig, ax = plt.subplots(1, 3, figsize=(10, 3.4), sharey=True)
    for a, cls in zip(ax, CLASSES):
        a.plot([0, 1], [0, 1], color="#bbbbbb", lw=1)
        for m in METHODS:
            rel = results[cls]["calibration"]["reliability"][m]
            a.plot(
                [r[2] for r in rel],
                [r[3] for r in rel],
                "o-",
                ms=3,
                lw=1,
                color=col[m],
                label=f"{m} (ECE {results[cls]['calibration']['methods'][m]['ece']:.3f})",
            )
        a.set_title(cls)
        a.set_xlabel("mean score in bin")
        a.legend(fontsize=7, loc="upper left")
    ax[0].set_ylabel("observed frequency")
    fig.suptitle(
        "Reliability (10 equal-mass bins, pooled held-out-by-speaker; exploratory)",
        fontsize=9,
    )
    fig.tight_layout()
    for ext in ("svg", "pdf"):
        fig.savefig(OUT / f"fig_reliability.{ext}")
    plt.close(fig)
    fig, ax = plt.subplots(1, 3, figsize=(10, 3.4), sharey=False)
    for a, cls in zip(ax, CLASSES):
        ab = results[cls]["abstention"]
        cv = [c["coverage"] for c in ab["curve"]]
        rk = [c["risk"] for c in ab["curve"]]
        lo = [c["risk_ci95"][0] for c in ab["curve"]]
        hi = [c["risk_ci95"][1] for c in ab["curve"]]
        a.plot(cv, rk, color="#1f77b4", lw=1.2)
        a.fill_between(cv, lo, hi, color="#1f77b4", alpha=0.2, lw=0)
        a.axhline(0.10, color="#cc0000", ls="--", lw=1, label="target 0.10")
        a.plot(
            [ab["coverage"]["value"]],
            [ab["risk"]["value"]],
            "k*",
            ms=9,
            label="operating point",
        )
        a.set_title(f"{cls} ({results[cls]['abstention_method']})")
        a.set_xlabel("coverage")
        a.legend(fontsize=7)
    ax[0].set_ylabel("risk (0-1 error among accepted)")
    fig.suptitle(
        "Risk-coverage, held-out-by-speaker (exploratory); band: speaker bootstrap 95%",
        fontsize=9,
    )
    fig.tight_layout()
    for ext in ("svg", "pdf"):
        fig.savefig(OUT / f"fig_risk_coverage.{ext}")
    plt.close(fig)


def main(mode="service"):
    prot = OUT / "PROTOCOL.md"
    h = hashlib.sha256(prot.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    frozen = json.loads((OUT / "PROTOCOL.sha256.json").read_text("utf8"))[
        "sha256_lf_normalised"
    ]
    if h != frozen:
        sys.exit("PROTOCOL.md changed after freezing; refusing to run")
    pool = RC.load_pool()
    feats = OD.load_features()
    turns = {
        r["turn_id"]: r for r in csv.DictReader(RC.TURNS.open(encoding="utf-8-sig"))
    }
    results = {}
    for cls in CLASSES:
        c = load_class(cls, feats, pool, turns)
        o_score = mode == "service" and cls != "repetition"
        if o_score:
            attach_o_scores(c, cls)
        print(cls, len(c["y"]), int(c["y"].sum()), flush=True)
        cal = calibration_study(c)
        res = cal.pop("_res")
        ev = {} if o_score else event_study(c)
        # abstention: adopted calibrator if any, else raw score
        am = cal["adopted"] or "none"
        ab = abstention_study(c, am)
        ab_raw = abstention_study(c, "none") if am != "none" else None
        results[cls] = {
            "n": int(len(c["y"])),
            "n_pos": int(c["y"].sum()),
            "n_speakers": int(len(set(c["spk"].tolist()))),
            "score_object": "o_pipeline_unit_score" if o_score else "v0_unit_max_softmax",
            "calibration": cal,
            "event_confidence_secondary": ev,
            "abstention_method": am,
            "abstention": ab,
            **(
                {
                    "abstention_raw_score": {
                        k: v for k, v in ab_raw.items() if k != "curve"
                    }
                }
                if ab_raw
                else {}
            ),
        }
        print(cls, "adopted:", cal["adopted"], flush=True)
    out = {
        "protocol_sha256": h,
        "status": "exploratory, pre-registered; development data",
        "bootstrap": [B_FULL, B_INNER],
        "results": results,
    }
    (OUT / ("results.json" if mode == "service" else "results_v0_profile.json")).write_text(json.dumps(out, indent=1), "utf8")
    figures(results)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "service")
