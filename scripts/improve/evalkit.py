"""Evaluation kit for the improvement study: loads frozen-cache features, defines versions and comparators,
computes confusion / P / R / F1, bootstrap CIs by group and PAIRED bootstrap on identical units. No label is created."""

from __future__ import annotations
import pickle
import sys
from pathlib import Path
import numpy as np
from scipy.stats import binomtest

sys.path.insert(0, str(Path(__file__).parent))
from common import FEAT, load_asr, references, CLASSES4, sha256_file, confusion, SEED  # noqa
from mrcd import decision as D  # noqa

LABELS = [
    "fluent",
    "filler_word",
    "prolongation",
    "repetition",
    "block",
    "revision",
    "rhetorical_pause",
    "neutral_pause",
]

_asr_rules = None


def asr_rules(text):
    global _asr_rules
    if _asr_rules is None:
        from evaluate_audio import (
            asr_rules as f,
        )  # baseline module, imported (not edited)

        _asr_rules = f
    return _asr_rules(text)


def load_features(corpus: str, mode: str) -> dict:
    out = {}
    for p in sorted(FEAT.glob(f"{corpus}_{mode}_shard*.pkl")):
        out.update(pickle.loads(p.read_bytes()))
    return out


def unit_table(corpus: str, mode: str = "zero", splits=None):
    """List of dict rows: unit_id, group, split, labels, rec (feature record), asr_text, asr rules."""
    feats = load_features(corpus, mode)
    refs = references(corpus)
    zero = feats if mode == "zero" else load_features(corpus, "zero")
    rows = []
    for uid, rec in feats.items():
        r = refs[uid]
        if splits is not None and r["split"] not in splits:
            continue
        words = load_asr(corpus, uid)["words"]
        text = " ".join(w["text"] for w in words)
        rows.append(
            {
                "unit_id": uid,
                "group": r["group"],
                "split": r["split"],
                "fold": r.get("fold"),
                "ref": r,
                "rec": rec,
                "rule_existing": zero[uid]["rule_pred"] if uid in zero else None,
                "asr_text": text,
                "asr": asr_rules(text),
                "dur_s": rec["dur_s"],
                "padded": rec["padded"],
            }
        )
    rows.sort(key=lambda r: r["unit_id"])
    return rows


# ------------------------------------------------------------------ predictions
def pred_argmax(row, cls):
    return D.any_window_argmax(row["rec"]["probs"], LABELS, cls)


def make_threshold_pred(agg, thr):
    return lambda row, cls: int(
        D.unit_score(row["rec"]["probs"], LABELS, cls, agg) >= thr
    )


def make_head_pred(head_scores_by_uid, class_index, thr):
    """head_scores_by_uid[uid] -> (W,C) per-window sigmoid scores; unit score = max over windows."""
    return lambda row, cls: int(
        head_scores_by_uid[row["unit_id"]][:, class_index[cls]].max() >= thr
    )


def with_abstention(pred_fn, tau):
    if tau is None:
        return pred_fn
    return lambda row, cls: 0 if row["dur_s"] < tau else pred_fn(row, cls)


COMPARATORS = {
    "always_present": lambda r, c: 1,
    "always_absent": lambda r, c: 0,
    "rule_existing_autolabeler": lambda r, c: r["rule_existing"][c],
    "rule_asr_lexical": lambda r, c: r["asr"][c],
}


def preds(rows, fn, cls):
    out = [fn(r, cls) for r in rows]
    return None if any(v is None for v in out) else np.array(out, int)


# ------------------------------------------------------------------ statistics
def _counts(y, p, g):
    ug, inv = np.unique(g, return_inverse=True)
    C = np.zeros((len(ug), 3))
    np.add.at(C[:, 0], inv, (y == 1) & (p == 1))
    np.add.at(C[:, 1], inv, (y == 0) & (p == 1))
    np.add.at(C[:, 2], inv, (y == 1) & (p == 0))
    return C


def _f1(c):  # c (...,3) tp,fp,fn
    den = 2 * c[..., 0] + c[..., 1] + c[..., 2]
    return np.where(den > 0, 2 * c[..., 0] / np.maximum(den, 1), np.nan)


def boot_ci(y, p, g, n_boot=2000, seed=SEED):
    C = _counts(np.asarray(y), np.asarray(p), np.asarray(g))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(C), (n_boot, len(C)))
    f = _f1(C[idx].sum(1))
    return (
        [float(np.nanpercentile(f, 2.5)), float(np.nanpercentile(f, 97.5))]
        if np.isfinite(f).any()
        else [float("nan")] * 2
    )


def paired_delta(y, p1, p0, g, n_boot=2000, seed=SEED):
    """F1(p1) - F1(p0) on the SAME units, bootstrap over groups. Returns dict(delta, ci, n_groups, p_boot_le0)."""
    y, p1, p0, g = map(np.asarray, (y, p1, p0, g))
    C1, C0 = _counts(y, p1, g), _counts(y, p0, g)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(C1), (n_boot, len(C1)))
    d = _f1(C1[idx].sum(1)) - _f1(C0[idx].sum(1))
    d0 = float(_f1(C1.sum(0)) - _f1(C0.sum(0)))
    return {
        "delta": d0,
        "ci": [float(np.nanpercentile(d, 2.5)), float(np.nanpercentile(d, 97.5))],
        "n_groups": int(len(C1)),
    }


def mcnemar(y, p1, p0):
    y, p1, p0 = map(np.asarray, (y, p1, p0))
    c1, c0 = p1 == y, p0 == y
    b, c = int((c1 & ~c0).sum()), int((~c1 & c0).sum())
    pv = binomtest(b, b + c, 0.5).pvalue if b + c else 1.0
    return {"v_right_ref_wrong": b, "ref_right_v_wrong": c, "p_exact": float(pv)}


def verdict(delta_ci):
    lo, hi = delta_ci
    if lo > 0:
        return "improvement demonstrated (CI > 0)"
    if hi < 0:
        return "WORSENING (CI < 0)"
    return "not demonstrated (CI includes 0)"


def macro_f1(rows, fn, classes, groups=None):
    f = []
    for c in classes:
        y = np.array([r["ref"][c] for r in rows])
        p = preds(rows, fn, c)
        f.append(confusion(y, p)["f1"])
    return float(np.nanmean(f))


def paired_macro_delta(Y, P1, P0, groups, classes, n_boot=2000, seed=SEED):
    """Paired by-speaker bootstrap of the macro-F1 difference (mean over classes). Y/P1/P0: dict class -> int array (same units)."""
    g = np.asarray(groups); ug = np.unique(g)
    C1 = [_counts(np.asarray(Y[c]), np.asarray(P1[c]), g) for c in classes]
    C0 = [_counts(np.asarray(Y[c]), np.asarray(P0[c]), g) for c in classes]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(ug), (n_boot, len(ug)))
    d = np.nanmean([_f1(C[idx].sum(1)) for C in C1], 0) - np.nanmean([_f1(C[idx].sum(1)) for C in C0], 0)
    d0 = float(np.nanmean([_f1(C.sum(0)) for C in C1]) - np.nanmean([_f1(C.sum(0)) for C in C0]))
    return {"delta": d0, "ci": [float(np.nanpercentile(d, 2.5)), float(np.nanpercentile(d, 97.5))]}


def cooccurrence(rows, p_fn, cls):
    """Rows: reference class present in the unit (or none) x columns: unit predicted/not predicted as `cls`. Counts of units."""
    names = ["f", "a", "r", "rev"]
    cmap = dict(zip(names, ["filler_word", "prolongation", "repetition", "revision"]))
    out = {}
    for n, c in cmap.items():
        sel = [r for r in rows if r["ref"].get(c)]
        out[f"ref_has_{n}"] = {
            "n": len(sel),
            f"pred_{cls}": int(sum(p_fn(r, cls) for r in sel)),
        }
    sel = [r for r in rows if not any(r["ref"].get(c) for c in cmap.values())]
    out["ref_has_none"] = {
        "n": len(sel),
        f"pred_{cls}": int(sum(p_fn(r, cls) for r in sel)),
    }
    return out


def eval_table(rows, comps, classes, corpus, partition, n_boot=2000):
    """One row per (class, comparator): confusion + P/R/F1 + by-group bootstrap CI of F1."""
    out = []
    g = np.array([r["group"] for r in rows])
    for c in classes:
        y = np.array([r["ref"][c] for r in rows])
        for name, fn in comps.items():
            p = preds(rows, fn, c)
            if p is None: continue
            m = confusion(y, p)
            m.update({"corpus": corpus, "partition": partition, "class": c, "comparator": name, "n_groups": int(len(set(g)))})
            if m["support_pos"] > 0 and len(set(g)) > 3:
                m["f1_ci95"] = boot_ci(y, p, g, n_boot)
            out.append(m)
    return out


def fmt_row(m):
    ci = m.get("f1_ci95")
    cis = f" [{ci[0]:.2f}-{ci[1]:.2f}]" if ci else ""
    unst = " UNSTABLE(<30 pos)" if m["support_pos"] < 30 else ""
    pf = lambda x: "n/a" if x != x else f"{x:.2f}"
    return f"| {m['comparator']} | {m['partition']} | {m['n']} | {m['support_pos']} | {m['tp']}/{m['fp']}/{m['fn']}/{m['tn']} | {pf(m['precision'])} | {pf(m['recall'])} | {pf(m['f1'])}{cis}{unst} |"
