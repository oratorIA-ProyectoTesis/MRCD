"""Plan A dev evaluation of decision profile o2026_12 on DIHANA folds 1-4 (EXPLORATORY). Protocol:
results/plan_a_20261108/PROTOCOL_EXPLORATORY_A.md (its sha256 is checked and written into every result JSON).

    python scripts/plan_a/evaluate.py cv        # restricted nested CV, filler_word / prolongation x {p0, p1} (resumable)
    python scripts/plan_a/evaluate.py final     # final fit on all rows (p1) -> JSON artifacts of profile o2026_12
    python scripts/plan_a/evaluate.py report    # every metric -> results_a.json (aggregates only)

Same environment as build_cache.py (PYTHONPATH=<worktree>, MRCD_REPO_ROOT=<main checkout>, OMP_NUM_THREADS=4).
Per-unit predictions stay in the git-ignored cache; results_a.json holds aggregates only.
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "4")

import csv  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import pickle  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_cache as BC  # noqa: E402

from mrcd.asr_recovery import common as RC  # noqa: E402
from mrcd.constants import LABELS  # noqa: E402
from mrcd.context_eval.metrics import holm  # noqa: E402
from mrcd.decision import any_window_argmax  # noqa: E402
from mrcd.optimize import search as SE  # noqa: E402
from mrcd.optimize import space as SP  # noqa: E402
from mrcd.optimize.blocks import FeatureStore  # noqa: E402
from mrcd.service import contextual as CX  # noqa: E402

WT = BC.WT
OUT = WT / "results" / "plan_a_20261108"
CACHE = BC.CACHE
PROTOCOL = OUT / "PROTOCOL_EXPLORATORY_A.md"
O11_DIR = WT / "mrcd" / "service" / "profiles" / "o2026_11"
O12_DIR = WT / "mrcd" / "service" / "profiles" / "o2026_12"
O_CAMPAIGN = RC.REPO / "results" / "mejora_objetivos_20261101" / "O_campaign"
PIPE_CLASSES = ("filler_word", "prolongation")
LABEL_COL = {"filler_word": "filler_present", "prolongation": "lengthening_present"}
LR_C = tuple(
    dict(hp)["C"] for m, hp in SP.MODELS if m == "lr"
)  # (0.03, 0.3, 3.0), frozen grid
N_BOOT, SEED = 5000, 13
LABEL = "exploratory (DIHANA folds 1-4)"


# ------------------------------------------------------------------------------------------- provenance
def protocol_sha() -> str:
    h = hashlib.sha256(PROTOCOL.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    frozen = json.loads((OUT / "PROTOCOL_EXPLORATORY_A.sha256.json").read_text("utf8"))[
        "sha256_lf"
    ]
    if h != frozen:
        raise SystemExit(
            "PROTOCOL_EXPLORATORY_A.md changed after freezing: write an annex instead"
        )
    return h


def git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=WT, capture_output=True, text=True
    ).stdout.strip()


# ------------------------------------------------------------------------------------------- data
def load_all():
    pool = RC.load_pool()  # folds 1-4 only (asserted), audited counts
    outer = RC.outer_folds(pool)
    turns = {
        r["turn_id"]: r
        for r in csv.DictReader(RC.TURNS.open(encoding="utf-8-sig"))
        if r["fold"] in ("1", "2", "3", "4")  # same filter as build_pool
    }
    rows = []
    for i, u in enumerate(pool):
        uid = u["unit_id"]
        t = turns[uid]
        f0 = BC.load_features(uid, BC.FEAT_P0_DIR)
        f1 = BC.load_features(uid)
        tags = json.loads(t["raw_tag_counts"])
        rows.append(
            {
                "uid": uid,
                "speaker": u["speaker"],
                "dihana_fold": int(u["fold"]),
                "outer": int(outer[i]),
                "duration_s": f1["n_samples"] / 16000.0,
                "rep_ref": u["ref"]["rep_literal"],
                "filler_ref": int(t["filler_present"]),
                "prolong_ref": int(t["lengthening_present"]),
                "lexfill_ref": int(tags.get("d:rellenar", 0) > 0),
                "words": {"p0": BC.p0_words(uid), "p1": BC.p1_words(uid)},
                "feat": {"p0": f0, "p1": f1},
            }
        )
    return rows


def make_data(rows, cls: str, prof: str) -> SE.Data:
    recs = []
    for r in rows:
        f = r["feat"][prof]
        w = r["words"][prof]
        recs.append(
            {
                "probs": f["probs"],
                "pooled": f["pooled"],
                "pause": f["pause"],
                "n_samples": f["n_samples"],
                "words": {prof: w},
                "text": {prof: " ".join(x["text"] for x in w)},
                "acoustic": None,
            }
        )
    y = np.array(
        [r["filler_ref" if cls == "filler_word" else "prolong_ref"] for r in rows]
    )
    return SE.Data(
        cls,
        [r["uid"] for r in rows],
        y,
        np.ones(len(rows)),
        np.array([r["speaker"] for r in rows]),
        np.array([r["outer"] for r in rows]),
        FeatureStore(recs, cls),
    )


def o11_config(cls: str) -> dict:
    return json.loads((O11_DIR / f"{cls}.json").read_text("utf8"))["config"]


def restricted_order(prof: str):
    """The o2026_11 final blocks and fusion with the frozen LR grid; nothing else is searched (protocol 1.3)."""

    def order(cls, seed, budget=SP.BUDGET):
        base = o11_config(cls)
        return [
            SP.Config(
                cls, prof, tuple(base["blocks"]), "lr", (("C", c),), base["fusion"]
            )
            for c in LR_C
        ]

    return order


# ------------------------------------------------------------------------------------------- stage cv / final
def stage_cv(rows) -> None:
    d_cv = CACHE / "cv"
    d_cv.mkdir(parents=True, exist_ok=True)
    for cls in PIPE_CLASSES:
        for prof in ("p1", "p0"):
            SP.sample_order = restricted_order(prof)
            d = make_data(rows, cls, prof)
            for f in sorted(set(d.outer.tolist())):
                p = d_cv / f"{cls}_{prof}_fold{f}.pkl"
                if p.exists():
                    continue
                t0 = time.time()
                r = SE.outer_fold(d, f)
                r["ids"] = [d.ids[i] for i in r["te"]]
                p.write_bytes(pickle.dumps(r))
                print(
                    cls,
                    prof,
                    "fold",
                    f,
                    r["config"]["hp"],
                    f"inner={r['inner_f1']:.3f}",
                    f"{time.time() - t0:.0f}s",
                    flush=True,
                )


def cv_predictions(
    cls: str, prof: str, ids: list[str]
) -> tuple[np.ndarray, list[dict]]:
    pos = {u: i for i, u in enumerate(ids)}
    dec = np.full(len(ids), -1)
    per = []
    for p in sorted((CACHE / "cv").glob(f"{cls}_{prof}_fold*.pkl")):
        r = pickle.loads(p.read_bytes())
        for u, v in zip(r["ids"], r["decision"]):
            dec[pos[u]] = int(v)
        per.append(
            {
                k: r[k]
                for k in (
                    "fold",
                    "config",
                    "inner_f1",
                    "thr",
                    "t_or",
                    "calibration",
                    "band",
                    "n_train",
                    "n_test",
                )
            }
        )
    if (dec < 0).any():
        raise SystemExit(f"cv incomplete for {cls}/{prof}: run the cv stage")
    return dec, per


def stage_final(rows) -> None:
    from mrcd.optimize import product_pipeline as PP
    from mrcd.optimize import safe_pipeline as SAFE

    sha = protocol_sha()
    O12_DIR.mkdir(parents=True, exist_ok=True)
    SP.sample_order = restricted_order("p1")
    files = {}
    for cls in PIPE_CLASSES:
        d = make_data(rows, cls, "p1")
        tr = np.arange(len(d.y))
        sel = SE.search(d, tr, SP.SEED + 99)
        pipe = PP.fit_class_pipeline(
            d,
            tr,
            sel["best"],
            f"fit on DIHANA folds 1-4 ({len(tr)} rows), P1 ASR text and P1 engine unit features; restricted search "
            f"(o2026_11 blocks and fusion, LR C grid {list(LR_C)}); plan A protocol sha256 {sha[:12]}",
        )
        blob = SAFE.dumps(SAFE.export_pipeline(pipe))
        (O12_DIR / f"{cls}.json").write_bytes(blob)
        files[f"{cls}.json"] = {
            "sha256": hashlib.sha256(blob).hexdigest(),
            "config": pipe.cfg.to_dict(),
            "inner_f1": sel["best"]["inner_f1"],
            "n_evaluated": sel["n_evaluated"],
        }
        print(cls, pipe.cfg.label(), files[f"{cls}.json"]["sha256"], flush=True)
    src = {
        "profile": "o2026_12",
        "status": "exploratory",
        "calibration_claim": "none",
        "asr_profile_trained_on": "p1_disfluent_prompt",
        "training_data": "DIHANA folds 1-4, 5056 turns, 180 speakers (fold 5 never indexed)",
        "features": "DisfluencyEngine.unit_features recomputed with P1 words (service path)",
        "selection": "restricted: o2026_11 final blocks and fusion per class, LR C in the frozen grid, inner K=3 by speaker",
        "protocol": "results/plan_a_20261108/PROTOCOL_EXPLORATORY_A.md",
        "protocol_sha256": sha,
        "git_head_at_fit": git_head(),
        "files": files,
    }
    (O12_DIR / "SOURCE.json").write_text(
        json.dumps(src, indent=1, sort_keys=True) + "\n", "utf8"
    )


# ------------------------------------------------------------------------------------------- statistics
class Boot:
    """Speaker-cluster bootstrap shared by every system on the same unit set (paired deltas use the same draws)."""

    def __init__(self, groups, n_boot: int = N_BOOT, seed: int = SEED):
        self.ug, self.inv = np.unique(
            np.asarray(groups).astype(str), return_inverse=True
        )
        rng = np.random.default_rng(seed)
        self.idx = rng.integers(0, len(self.ug), (n_boot, len(self.ug)))
        self.n_boot = n_boot

    def counts(self, y, p):
        y, p = np.asarray(y).astype(int), np.asarray(p).astype(int)
        C = np.zeros((len(self.ug), 3))
        np.add.at(C[:, 0], self.inv, (y == 1) & (p == 1))
        np.add.at(C[:, 1], self.inv, (y == 0) & (p == 1))
        np.add.at(C[:, 2], self.inv, (y == 1) & (p == 0))
        return C

    @staticmethod
    def prf(T):
        tp, fp, fn = T[..., 0], T[..., 1], T[..., 2]
        with np.errstate(invalid="ignore", divide="ignore"):
            P = np.where(tp + fp > 0, tp / (tp + fp), np.nan)
            R = np.where(tp + fn > 0, tp / (tp + fn), np.nan)
            F = np.where(2 * tp + fp + fn > 0, 2 * tp / (2 * tp + fp + fn), np.nan)
        return {"precision": P, "recall": R, "f1": F}

    def boot_prf(self, y, p):
        return self.prf(self.counts(y, p)[self.idx].sum(1))


def _num(x):
    return None if x is None or not np.isfinite(x) else round(float(x), 4)


def _ci(a):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    return (
        [None, None]
        if not len(a)
        else [_num(np.percentile(a, 2.5)), _num(np.percentile(a, 97.5))]
    )


def describe(B: Boot, y, p) -> dict:
    y, p = np.asarray(y).astype(int), np.asarray(p).astype(int)
    tp, fp = int(((y == 1) & (p == 1)).sum()), int(((y == 0) & (p == 1)).sum())
    fn, tn = int(((y == 1) & (p == 0)).sum()), int(((y == 0) & (p == 0)).sum())
    point = B.prf(np.array([tp, fp, fn], float))
    bs = B.boot_prf(y, p)
    out = {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "n": int(len(y)),
        "positives": int(y.sum()),
    }
    for m in ("precision", "recall", "f1"):
        v = float(point[m])
        if np.isfinite(v):
            out[m] = {"value": _num(v), "ci95": _ci(bs[m])}
        else:
            out[m] = {
                "value": None,
                "ci95": [None, None],
                "not_available": "undefined: no predicted or no reference positives",
            }
    return out


def paired(B: Boot, y, a, b) -> dict:
    """a - b, same units and bootstrap draws; two-sided bootstrap p with floor 1/(B+1) (as asr_recovery.stats)."""
    pa, pb = B.prf(B.counts(y, a).sum(0)), B.prf(B.counts(y, b).sum(0))
    ba, bb = B.boot_prf(y, a), B.boot_prf(y, b)
    out = {}
    for m in ("precision", "recall", "f1"):
        d = ba[m] - bb[m]
        d = d[np.isfinite(d)]
        n = len(d)
        p = (
            min(
                1.0,
                2 * min((np.sum(d <= 0) + 1) / (n + 1), (np.sum(d >= 0) + 1) / (n + 1)),
            )
            if n
            else float("nan")
        )
        out[m] = {
            "delta": _num(float(pa[m] - pb[m])),
            "ci95": _ci(d),
            "p_boot": _num(p),
        }
    return out


def per_fold(rows, key, y, systems: dict) -> list[dict]:
    out = []
    keys = sorted({r[key] for r in rows})
    for k in keys:
        ix = np.array([r[key] == k for r in rows])
        row = {
            key: int(k),
            "n": int(ix.sum()),
            "positives": int(np.asarray(y)[ix].sum()),
        }
        for name, p in systems.items():
            P = Boot.prf(
                np.array(
                    [
                        ((y == 1) & (p == 1))[ix].sum(),
                        ((y == 0) & (p == 1))[ix].sum(),
                        ((y == 1) & (p == 0))[ix].sum(),
                    ],
                    float,
                )
            )
            row[name] = {m: _num(float(P[m])) for m in ("precision", "recall", "f1")}
        out.append(row)
    return out


# ------------------------------------------------------------------------------------------- report
def repetition_block(rows) -> tuple[dict, dict]:
    R = [r for r in rows if r["rep_ref"] is not None]
    y = np.array([int(r["rep_ref"]) for r in R])
    g = [r["speaker"] for r in R]
    sysd = {
        k: []
        for k in (
            "V0",
            "p0_rule",
            "p1_rule",
            "o2026_12",
            "o2026_12_u",
            "V0_p1_features",
        )
    }
    agg = None
    flips = {
        "to_negative_on_reference_positive": 0,
        "to_negative_on_reference_negative": 0,
    }
    flip_rules = {
        "positive": {r: 0 for r in CX.REPETITION_RULES},
        "negative": {r: 0 for r in CX.REPETITION_RULES},
    }
    for r, yy in zip(R, y):
        dur = r["duration_s"]
        c0 = CX.classify_repetitions(r["words"]["p0"], dur, rules=())
        c1 = CX.classify_repetitions(r["words"]["p1"], dur)
        cu = CX.classify_repetitions(
            r["words"]["p1"], dur, rules=("loop", "gap", "seam")
        )
        sysd["V0"].append(
            any_window_argmax(r["feat"]["p0"]["probs"], list(LABELS), "repetition")
        )
        sysd["V0_p1_features"].append(
            any_window_argmax(r["feat"]["p1"]["probs"], list(LABELS), "repetition")
        )
        sysd["p0_rule"].append(int(bool(c0)))
        sysd["p1_rule"].append(int(bool(c1)))
        o12 = int(any(c["state"] == CX.DISFLUENT for c in c1))
        sysd["o2026_12"].append(o12)
        sysd["o2026_12_u"].append(int(any(c["state"] == CX.DISFLUENT for c in cu)))
        st = CX.demotion_stats(c1)
        if agg is None:
            agg = st
        else:
            agg["n_candidates"] += st["n_candidates"]
            for part in ("by_state", "demoted_by_rule", "matched_by_rule"):
                for k, v in st[part].items():
                    agg[part][k] += v
        if c1 and not o12:
            side = "positive" if yy else "negative"
            flips[f"to_negative_on_reference_{side}"] += 1
            for rule in {c["rule"] for c in c1 if c["rule"]}:
                flip_rules[side][rule] += 1
    sysd = {k: np.array(v, int) for k, v in sysd.items()}
    B = Boot(g)
    out = {
        "unit": "DIHANA turn (rep_literal defined)",
        "n_turns": int(len(y)),
        "n_speakers": int(len(set(g))),
        "systems": {k: describe(B, y, v) for k, v in sysd.items()},
    }
    fam = {
        "HA1_o2026_12_minus_V0": paired(B, y, sysd["o2026_12"], sysd["V0"]),
        "HA2_o2026_12_minus_p0_rule": paired(B, y, sysd["o2026_12"], sysd["p0_rule"]),
        "HA3_o2026_12_minus_p1_rule": paired(B, y, sysd["o2026_12"], sysd["p1_rule"]),
    }
    adj = holm([fam[k]["f1"]["p_boot"] for k in fam])
    for k, a in zip(fam, adj):
        f = fam[k]["f1"]
        f["p_holm"] = _num(a)
        f["verdict"] = (
            "supported"
            if (a < 0.05 and f["ci95"][0] is not None and f["ci95"][0] > 0)
            else "not supported"
        )
    out["primary_family_holm"] = fam
    out["descriptive_deltas"] = {
        "o2026_12_u_minus_p1_rule": paired(B, y, sysd["o2026_12_u"], sysd["p1_rule"]),
        "p1_rule_minus_p0_rule": paired(B, y, sysd["p1_rule"], sysd["p0_rule"]),
        "o2026_12_minus_o2026_12_u": paired(B, y, sysd["o2026_12"], sysd["o2026_12_u"]),
    }
    d3 = fam["HA3_o2026_12_minus_p1_rule"]
    n_c = agg["n_candidates"]
    out["demotion"] = {
        **agg,
        "share_demoted": _num((n_c - agg["by_state"]["disfluent"]) / n_c)
        if n_c
        else None,
        "share_demoted_by_rule": {
            k: _num(v / n_c) if n_c else None for k, v in agg["demoted_by_rule"].items()
        },
        "turn_flips_1_to_0": flips,
        "turn_flips_by_rule_present": flip_rules,
        "raises_precision_without_losing_recall": bool(
            d3["precision"]["ci95"][0] is not None
            and d3["precision"]["ci95"][0] > 0
            and d3["recall"]["delta"] >= -0.02
        ),
        "rule": "true iff delta precision (o2026_12 - p1_rule) CI lower > 0 and delta recall >= -0.02 (protocol section 4)",
    }
    out["per_outer_fold"] = per_fold(
        R, "outer", y, {k: sysd[k] for k in ("V0", "p0_rule", "p1_rule", "o2026_12")}
    )
    out["per_dihana_fold"] = per_fold(
        R,
        "dihana_fold",
        y,
        {k: sysd[k] for k in ("V0", "p0_rule", "p1_rule", "o2026_12")},
    )
    return out, sysd


def o11_predictions(cls: str, ids: list[str], y: np.ndarray) -> np.ndarray:
    rows = {
        r["unit_id"]: r
        for r in csv.DictReader(
            open(O_CAMPAIGN / f"predictions_{cls}.csv", encoding="utf8")
        )
    }
    assert all(int(rows[u]["y"]) == int(v) for u, v in zip(ids, y)), (
        "label mismatch with the o2026_11 predictions"
    )
    return np.array([int(rows[u]["decision"]) for u in ids])


def pipelines_block(rows) -> tuple[dict, dict]:
    ids = [r["uid"] for r in rows]
    g = [r["speaker"] for r in rows]
    B = Boot(g)
    out, preds = {}, {}
    for cls in PIPE_CLASSES:
        key = "filler_ref" if cls == "filler_word" else "prolong_ref"
        y = np.array([r[key] for r in rows])
        p11 = o11_predictions(cls, ids, y)
        p1, per1 = cv_predictions(cls, "p1", ids)
        p0, per0 = cv_predictions(cls, "p0", ids)
        name = "filled_pause" if cls == "filler_word" else "prolongation"
        sysd = {"o2026_11": p11, "p0_restricted": p0, f"o2026_12_{name}": p1}
        preds[cls] = (y, sysd)
        gate = paired(B, y, p1, p11)
        f = gate["f1"]
        out[cls] = {
            "reference": LABEL_COL[cls]
            + (
                " (all DIHANA f marks: non-lexical filled pauses)"
                if cls == "filler_word"
                else ""
            ),
            "systems": {k: describe(B, y, v) for k, v in sysd.items()},
            "gate_vs_o2026_11": {
                **gate,
                "pass": bool(
                    f["delta"] >= -0.02
                    and f["ci95"][1] is not None
                    and f["ci95"][1] >= 0
                ),
                "rule": "delta F1 >= -0.02 and CI upper >= 0 (protocol section 4, GF / GP)",
            },
            "p1_minus_p0_restricted": paired(B, y, p1, p0),
            "p0_restricted_minus_o2026_11": paired(B, y, p0, p11),
            "per_outer_fold": per_fold(rows, "outer", y, sysd),
            "cv_per_fold_selection": {"p1": per1, "p0_restricted": per0},
        }
    return out, preds


def lexical_block(rows, filled_p1) -> dict:
    g = [r["speaker"] for r in rows]
    B = Boot(g)
    y = np.array([r["lexfill_ref"] for r in rows])
    yf = np.array([r["filler_ref"] for r in rows])
    dis, len_, dis0 = [], [], []
    st = {"n_candidates": 0, "by_state": {s: 0 for s in CX.CONTEXT_STATES}}
    tok = {}
    for r in rows:
        c = CX.classify_lexical_fillers(r["words"]["p1"])
        c0 = CX.classify_lexical_fillers(r["words"]["p0"])
        dis.append(int(any(x["state"] == CX.DISFLUENT for x in c)))
        len_.append(int(bool(c)))
        dis0.append(int(any(x["state"] == CX.DISFLUENT for x in c0)))
        s = CX.lexical_stats(c)
        st["n_candidates"] += s["n_candidates"]
        for k, v in s["by_state"].items():
            st["by_state"][k] += v
        for x in c:
            tok.setdefault(
                x["token"], {"disfluent": 0, "uncertain": 0, "functional": 0}
            )[x["state"]] += 1
    dis, len_, dis0 = (np.array(a, int) for a in (dis, len_, dis0))
    union = ((filled_p1 == 1) | (dis == 1)).astype(int)
    return {
        "reference": "turn has >= 1 DIHANA d tipo=rellenar mark (discourse marker with a filling function); negative "
        "coverage of d marks not verified",
        "n_turns": int(len(y)),
        "systems": {
            "o2026_12_lexical_disfluent": describe(B, y, dis),
            "o2026_12_lexical_lenient_disfluent_or_uncertain": describe(B, y, len_),
            "p0_lexical_disfluent": describe(B, y, dis0),
        },
        "candidate_states": st,
        "candidates_by_lexicon_item": tok,
        "union_filler_word_vs_filler_present": {
            "note": "filled_pause OR lexical disfluent against the non-lexical f-mark reference (descriptive)",
            "o2026_12_union": describe(B, yf, union),
            "o2026_12_filled_pause_only": describe(B, yf, filled_p1),
        },
        "per_outer_fold": per_fold(
            rows, "outer", y, {"o2026_12_lexical_disfluent": dis}
        ),
    }


def stage_report(rows) -> None:
    sha = protocol_sha()
    rep, _ = repetition_block(rows)
    pipes, preds = pipelines_block(rows)
    lex = lexical_block(rows, preds["filler_word"][1]["o2026_12_filled_pause"])
    src = (
        json.loads((O12_DIR / "SOURCE.json").read_text("utf8"))
        if (O12_DIR / "SOURCE.json").exists()
        else None
    )
    res = {
        "label": LABEL,
        "protocol": "results/plan_a_20261108/PROTOCOL_EXPLORATORY_A.md",
        "protocol_sha256": sha,
        "git_head": git_head(),
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "data": {
            "corpus": "DIHANA folds 1-4 (fold 5 never indexed)",
            "turns": len(rows),
            "speakers": len({r["speaker"] for r in rows}),
            "outer_partition": "frozen 5-fold speaker partition (asr_recovery.common.outer_folds, seed 13)",
            "p1_words": "1025 frozen ASR-recovery decodes + 4031 decoded by scripts/plan_a/build_cache.py",
            "features": "DisfluencyEngine.unit_features recomputed from audio with P0 and P1 words",
        },
        "bootstrap": {
            "cluster": "speaker",
            "n_boot": N_BOOT,
            "seed": SEED,
            "ci": "percentile 95 %",
        },
        "repetition": rep,
        "filler_word_filled_pause_and_prolongation": pipes,
        "lexical_filler": lex,
        "not_available": {
            "functional_state_accuracy": "DIHANA carries no function labels; functional cannot be validated",
            "lexical_filler_outside_dihana": "no other corpus with a lexical-filler reference was evaluated",
            "timing_accuracy": "scoring is turn-level; event boundaries are not compared",
        },
        "profile_source": src,
    }
    (OUT / "results_a.json").write_text(
        json.dumps(res, indent=1, ensure_ascii=False) + "\n", "utf8"
    )
    print(
        json.dumps(
            {"repetition_f1": {k: v["f1"]["value"] for k, v in rep["systems"].items()}},
            indent=1,
        )
    )


def main() -> None:
    stage = sys.argv[1]
    protocol_sha()
    rows = load_all()
    {"cv": stage_cv, "final": stage_final, "report": stage_report}[stage](rows)


if __name__ == "__main__":
    main()
