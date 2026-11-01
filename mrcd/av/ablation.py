"""Speaker-grouped paired ablation (A+T vs A+T+V, V alone as control), speaker bootstrap and Holm.

All variants see exactly the same units and the same speaker-grouped folds. Threshold is fixed at 0.5 with
class-balanced logistic regression (no threshold tuning, so nothing is selected on the test fold).
"""

from __future__ import annotations

import hashlib

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

VARIANTS = ("AT", "V", "F1_concat", "F2_gated", "F3_resid")
FUSION = ("F1_concat", "F2_gated", "F3_resid")


def subsample_negatives(unit_ids, y, cap_ratio: float) -> np.ndarray:
    """Boolean keep-mask. Keeps all positives and, if negatives exceed cap_ratio*positives, the negatives with the
    smallest sha1(unit_id): deterministic and independent of any feature or model."""
    y = np.asarray(y).astype(int)
    keep = np.ones(len(y), bool)
    npos, neg = int(y.sum()), np.where(y == 0)[0]
    cap = int(np.floor(cap_ratio * npos))
    if len(neg) > cap:
        order = sorted(
            neg, key=lambda i: hashlib.sha1(str(unit_ids[i]).encode()).hexdigest()
        )
        keep[order[cap:]] = False
    return keep


def grouped_folds(groups, y, k: int, seed: int) -> np.ndarray:
    """Fold index per unit. Whole speakers go to one fold; greedy balancing of positives (labels only balance sizes)."""
    groups = np.asarray(groups)
    y = np.asarray(y).astype(int)
    rng = np.random.default_rng(seed)
    sp = list(dict.fromkeys(groups.tolist()))
    pos = {s: int(y[groups == s].sum()) for s in sp}
    tot = {s: int((groups == s).sum()) for s in sp}
    order = sorted(sp, key=lambda s: (-pos[s], rng.random()))
    load_p, load_n = np.zeros(k), np.zeros(k)
    assign = {}
    for s in order:
        f = int(np.lexsort((load_n, load_p))[0])
        assign[s] = f
        load_p[f] += pos[s]
        load_n[f] += tot[s]
    return np.array([assign[g] for g in groups])


def _design(variant, tr, te, A, T, V, conf, imputers):
    def prep(M):
        return M

    AT = np.hstack([A, T])
    if variant == "AT":
        Xtr, Xte = AT[tr], AT[te]
    elif variant == "V":
        Xtr, Xte = V[tr], V[te]
    else:
        Xtr, Xte = np.hstack([AT[tr], V[tr]]), np.hstack([AT[te], V[te]])
    med = np.nanmedian(Xtr, 0)
    med = np.where(np.isfinite(med), med, 0.0)
    fill = lambda M: np.where(np.isfinite(M), M, med)  # noqa: E731
    Xtr, Xte = fill(Xtr), fill(Xte)
    sc = StandardScaler().fit(Xtr)
    Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
    nv = V.shape[1]
    if variant == "F2_gated":
        Xtr[:, -nv:] *= conf[tr][:, None]
        Xte[:, -nv:] *= conf[te][:, None]
    if variant == "F3_resid":
        na = Xtr.shape[1] - nv
        rg = Ridge(alpha=10.0).fit(Xtr[:, :na], Xtr[:, na:])
        Xtr = np.hstack([Xtr[:, :na], Xtr[:, na:] - rg.predict(Xtr[:, :na])])
        Xte = np.hstack([Xte[:, :na], Xte[:, na:] - rg.predict(Xte[:, :na])])
    return Xtr, Xte


def fit_predict(variant, tr, te, A, T, V, conf, y, C=1.0):
    Xtr, Xte = _design(variant, tr, te, A, T, V, conf, None)
    if len(set(y[tr].tolist())) < 2:
        return np.full(len(te), float(y[tr].mean()))
    clf = LogisticRegression(C=C, class_weight="balanced", max_iter=2000)
    clf.fit(Xtr, y[tr])
    return clf.predict_proba(Xte)[:, 1]


def oof_predictions(
    A, T, V, conf, y, groups, k, repeats, seed, C=1.0, variants=VARIANTS
):
    """(R, n, len(variants)) out-of-fold probabilities; V is the speaker-normalised visual block."""
    n = len(y)
    out = np.full((repeats, n, len(variants)), np.nan)
    for r in range(repeats):
        folds = grouped_folds(groups, y, k, seed + r)
        for f in range(k):
            te = np.where(folds == f)[0]
            tr = np.where(folds != f)[0]
            if len(te) == 0:
                continue
            for j, v in enumerate(variants):
                out[r, te, j] = fit_predict(v, tr, te, A, T, V, conf, y, C)
    return out


def counts_by_speaker(probs, y, groups, thr=0.5):
    """(S, R, Vn, 3) TP/FP/FN per speaker; speakers list."""
    sp = list(dict.fromkeys(np.asarray(groups).tolist()))
    R, n, Vn = probs.shape
    out = np.zeros((len(sp), R, Vn, 3))
    pred = probs >= thr
    for i, s in enumerate(sp):
        m = np.asarray(groups) == s
        yy = y[m].astype(bool)
        pp = pred[:, m, :]
        out[i, :, :, 0] = (pp & yy[None, :, None]).sum(1)
        out[i, :, :, 1] = (pp & ~yy[None, :, None]).sum(1)
        out[i, :, :, 2] = (~pp & yy[None, :, None]).sum(1)
    return sp, out


def prf(c):
    tp, fp, fn = c[..., 0], c[..., 1], c[..., 2]
    P = np.divide(
        tp, tp + fp, out=np.full_like(tp, np.nan, dtype=float), where=(tp + fp) > 0
    )
    R = np.divide(
        tp, tp + fn, out=np.full_like(tp, np.nan, dtype=float), where=(tp + fn) > 0
    )
    F = np.divide(
        2 * tp,
        2 * tp + fp + fn,
        out=np.zeros_like(tp, dtype=float),
        where=(2 * tp + fp + fn) > 0,
    )
    return P, R, F


def speaker_bootstrap(cnt, B, seed):
    """F1 per variant for B speaker-resamples. cnt (S,R,Vn,3). Returns (B, Vn) mean-over-repeats pooled F1."""
    S = cnt.shape[0]
    rng = np.random.default_rng(seed)
    w = rng.multinomial(S, np.full(S, 1.0 / S), size=B)  # (B,S) speaker multiplicities
    tot = np.einsum("bs,srvc->brvc", w, cnt)
    _, _, F = prf(tot)
    return F.mean(1)


def holm(pvals: dict) -> dict:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m, out, run = len(items), {}, 0.0
    for i, (k, p) in enumerate(items):
        run = max(run, min(1.0, (m - i) * p))
        out[k] = run
    return out


def boot_p(delta_b: np.ndarray) -> float:
    B = len(delta_b)
    return float(
        min(
            1.0,
            2
            * min(
                (np.sum(delta_b <= 0) + 1) / (B + 1),
                (np.sum(delta_b >= 0) + 1) / (B + 1),
            ),
        )
    )


def summarize(probs, y, groups, B, seed, margin, variants=VARIANTS):
    sp, cnt = counts_by_speaker(probs, y, groups)
    tot = cnt.sum(0)  # (R,Vn,3)
    P, R_, F = prf(tot)
    boot = speaker_bootstrap(cnt, B, seed)
    vi = {v: i for i, v in enumerate(variants)}
    res = {
        "speakers": len(sp),
        "pos_speakers": int(len({g for g, yy in zip(np.asarray(groups).tolist(), y) if yy})),
        "n_units": int(len(y)),
        "n_pos": int(np.sum(y)),
        "variants": {},
        "paired": {},
    }
    ap = {
        v: float(
            np.mean(
                [
                    average_precision_score(y, probs[r, :, vi[v]])
                    for r in range(probs.shape[0])
                ]
            )
        )
        for v in variants
    }
    try:
        auc = {
            v: float(
                np.mean(
                    [
                        roc_auc_score(y, probs[r, :, vi[v]])
                        for r in range(probs.shape[0])
                    ]
                )
            )
            for v in variants
        }
    except ValueError:
        auc = {v: float("nan") for v in variants}
    for v in variants:
        j = vi[v]
        lo, hi = np.percentile(boot[:, j], [2.5, 97.5])
        res["variants"][v] = {
            "tp": float(tot[:, j, 0].mean()),
            "fp": float(tot[:, j, 1].mean()),
            "fn": float(tot[:, j, 2].mean()),
            "precision": float(np.nanmean(P[:, j])),
            "recall": float(np.nanmean(R_[:, j])),
            "f1": float(F[:, j].mean()),
            "f1_ci95": [float(lo), float(hi)],
            "ap": ap[v],
            "auc": auc[v],
        }
    fam = {}
    for v in FUSION:
        if v not in vi or "AT" not in vi:
            continue
        d = boot[:, vi[v]] - boot[:, vi["AT"]]
        point = float(F[:, vi[v]].mean() - F[:, vi["AT"]].mean())
        lo, hi = np.percentile(d, [2.5, 97.5])
        fam[v] = boot_p(d)
        res["paired"][v] = {
            "delta_f1": point,
            "ci95": [float(lo), float(hi)],
            "p_boot": fam[v],
            "delta_ap": ap[v] - ap["AT"],
            "within_equivalence_margin": bool(lo > -margin and hi < margin),
        }
    for v, p in holm(fam).items():
        res["paired"][v]["p_holm"] = p
    prev = float(np.mean(y))
    res["chance_f1_all_positive"] = 2 * prev / (1 + prev)
    return res


def verdict(full: dict, no_dom: dict, cfg: dict) -> dict:
    """Frozen decision rule (see PROTOCOL.md section 8)."""
    adequate = (
        full["n_pos"] >= cfg["min_admitted_positives"]
        and full.get("pos_speakers", 0) >= cfg["min_positive_speakers"]
    )
    out = {"adequate_sample": bool(adequate), "per_variant": {}}
    for v, r in full["paired"].items():
        nd = no_dom["paired"].get(v) if no_dom else None
        gain = (
            r["p_holm"] < 0.05
            and r["ci95"][0] > 0
            and r["delta_f1"] > 0
            and (nd is not None and nd["delta_f1"] > 0)
        )
        none = r["within_equivalence_margin"] and r["p_holm"] >= 0.05
        if not adequate:
            label = "no concluyente"  # frozen rule: an inadequate sample never supports a verdict either way
        else:
            label = "aporte medido" if gain else ("sin aporte" if none else "no concluyente")
        out["per_variant"][v] = label
    labels = set(out["per_variant"].values())
    out["overall"] = (
        "aporte medido"
        if "aporte medido" in labels
        else ("sin aporte" if labels == {"sin aporte"} else "no concluyente")
    )
    return out


def identity_probe(X, groups, min_units=8, seed=0, folds=4, C=1.0):
    """Linear probe: balanced accuracy of identifying the speaker from features (unit-level stratified CV).

    Only speakers with >= min_units rows are used. Returns (balanced_acc, chance, n_speakers, n_rows).
    """
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.model_selection import StratifiedKFold

    g = np.asarray(groups)
    vals, cnts = np.unique(g, return_counts=True)
    keep_sp = vals[cnts >= max(min_units, folds)]
    m = np.isin(g, keep_sp)
    Xm, gm = np.asarray(X, float)[m], g[m]
    med = np.nanmedian(Xm, 0)
    Xm = np.where(np.isfinite(Xm), Xm, np.where(np.isfinite(med), med, 0.0))
    pred = np.empty(len(gm), object)
    for tr, te in StratifiedKFold(folds, shuffle=True, random_state=seed).split(Xm, gm):
        sc = StandardScaler().fit(Xm[tr])
        clf = LogisticRegression(C=C, max_iter=3000).fit(sc.transform(Xm[tr]), gm[tr])
        pred[te] = clf.predict(sc.transform(Xm[te]))
    return (
        float(balanced_accuracy_score(gm, pred)),
        float(1.0 / len(keep_sp)),
        int(len(keep_sp)),
        int(m.sum()),
    )
