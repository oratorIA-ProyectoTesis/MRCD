"""Plan F evaluation (EXPLORATORY, 36 DIHANA folds 1-4 validation speakers). Protocol:
results/plan_f_20261108/PROTOCOL_EXPLORATORY_F.md (its sha256 is checked and written into the result JSON).

    python scripts/plan_f/evaluate_f.py            # -> results/plan_f_20261108/results_f.json (aggregates only)

Run after ``decode_ft.py`` and ``ft_features.py``. No threshold, token set, rule or margin is tuned here: every constant below is
the protocol's. Per-turn data stay in the git-ignored cache; the JSON holds aggregates only (no ids, no text).
"""

from __future__ import annotations

import csv
import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

LABEL = "exploratory (DIHANA folds 1-4, 36 validation speakers)"
N_BOOT, SEED = 5000, 13
HF3_MARGIN = 0.05
NI_F1_MARGIN = 0.02


def _load_plan_a_evaluate():
    p = C.WT / "scripts" / "plan_a" / "evaluate.py"
    sys.path.insert(0, str(p.parent))
    spec = importlib.util.spec_from_file_location("plan_a_evaluate", p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["plan_a_evaluate"] = mod
    spec.loader.exec_module(mod)
    return mod


def git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=C.WT, capture_output=True, text=True
    ).stdout.strip()


# ------------------------------------------------------------------------------------------- data
def load_rows():
    from mrcd.asr_recovery import common as RC

    turns = {
        r["turn_id"]: r
        for r in csv.DictReader(RC.TURNS.open(encoding="utf-8-sig"))
        if r["fold"] in ("1", "2", "3", "4")
    }
    rows = []
    for u in C.val_pool():
        uid = u["unit_id"]
        t = turns[uid]
        rows.append(
            {
                "uid": uid,
                "speaker": u["speaker"],
                "rep_ref": u["ref"]["rep_literal"],
                "filler_ref": int(t["filler_present"]),
                "prolong_ref": int(t["lengthening_present"]),
                "n_ref_words": len(u["text_A"].split()),
                "words": {
                    "p0": C.p0_words(uid),
                    "p1": C.p1_words(uid),
                    "p1_reused": C.p1_words(uid, "reused"),
                    "ft": C.ft_words(uid),
                },
                "feat_p0": C.load_features(uid, C.PLAN_A_CACHE / "feat_p0"),
                "feat_ft": C.load_features(uid, C.CACHE / "feat_ft"),
            }
        )
    return rows


def records(rows, which: str) -> list[dict]:
    """Turn-level unit records as the o2026_11 pipelines were fitted (key ``p0`` is their internal convention)."""
    recs = []
    for r in rows:
        f = r["feat_p0"] if which == "p0" else r["feat_ft"]
        w = r["words"]["p0" if which == "p0" else "ft"]
        recs.append(
            {
                "probs": f["probs"],
                "pooled": f["pooled"],
                "pause": f["pause"],
                "n_samples": f["n_samples"],
                "words": {"p0": w},
                "text": {"p0": " ".join(x["text"] for x in w)},
                "acoustic": None,
            }
        )
    return recs


# ------------------------------------------------------------------------------------------- stats helpers
def _num(x):
    return None if x is None or not np.isfinite(x) else round(float(x), 4)


def ratio_block(B, num, den, scale: float = 1.0) -> dict:
    """Pooled ratio sum(num)/sum(den)*scale with a speaker-cluster bootstrap CI (same draws as ``B``)."""
    S = np.zeros((len(B.ug), 2))
    np.add.at(S[:, 0], B.inv, np.asarray(num, float))
    np.add.at(S[:, 1], B.inv, np.asarray(den, float))
    tot = S.sum(0)
    if tot[1] <= 0:
        return {
            "value": None,
            "ci95": [None, None],
            "not_available": "empty denominator",
        }
    bs = S[B.idx].sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        v = np.where(bs[:, 1] > 0, bs[:, 0] / bs[:, 1] * scale, np.nan)
    v = v[np.isfinite(v)]
    return {
        "value": _num(tot[0] / tot[1] * scale),
        "ci95": [_num(np.percentile(v, 2.5)), _num(np.percentile(v, 97.5))],
        "numerator": float(tot[0]),
        "denominator": float(tot[1]),
    }


def ratio_delta(B, na, da, nb, db, scale: float = 1.0) -> dict:
    def per(num, den):
        S = np.zeros((len(B.ug), 2))
        np.add.at(S[:, 0], B.inv, np.asarray(num, float))
        np.add.at(S[:, 1], B.inv, np.asarray(den, float))
        return S

    A, Bm = per(na, da), per(nb, db)

    def val(S):
        t = S.sum(0)
        return t[0] / t[1] * scale

    ba, bb = A[B.idx].sum(1), Bm[B.idx].sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        d = ba[:, 0] / ba[:, 1] * scale - bb[:, 0] / bb[:, 1] * scale
    d = d[np.isfinite(d)]
    return {
        "delta": _num(val(A) - val(Bm)),
        "ci95": [_num(np.percentile(d, 2.5)), _num(np.percentile(d, 97.5))],
    }


def two_sided_p(d: np.ndarray) -> float:
    d = d[np.isfinite(d)]
    n = len(d)
    return min(
        1.0, 2 * min((np.sum(d <= 0) + 1) / (n + 1), (np.sum(d >= 0) + 1) / (n + 1))
    )


# ------------------------------------------------------------------------------------------- blocks
def repetition_block(EV, rows) -> tuple[dict, dict]:
    from mrcd.constants import LABELS
    from mrcd.decision import any_window_argmax
    from mrcd.service import contextual as CX

    R = [r for r in rows if r["rep_ref"] is not None]
    y = np.array([int(r["rep_ref"]) for r in R])
    B = EV.Boot([r["speaker"] for r in R], N_BOOT, SEED)
    sysd = {k: [] for k in ("V0", "p0_rule", "p1_rule", "ft_rule", "p1_rule_reused")}
    n_cand = {k: 0 for k in ("p0_rule", "p1_rule", "ft_rule", "p1_rule_reused")}
    for r in R:
        sysd["V0"].append(
            any_window_argmax(r["feat_p0"]["probs"], list(LABELS), "repetition")
        )
        for k, w in (
            ("p0_rule", "p0"),
            ("p1_rule", "p1"),
            ("ft_rule", "ft"),
            ("p1_rule_reused", "p1_reused"),
        ):
            c = CX.primary_repetition_candidates(r["words"][w])
            sysd[k].append(int(bool(c)))
            n_cand[k] += len(c)
    sysd = {k: np.array(v, int) for k, v in sysd.items()}
    out = {
        "unit": "DIHANA validation turn with rep_literal defined",
        "n_turns": int(len(y)),
        "n_speakers": int(len(B.ug)),
        "n_positive_turns": int(y.sum()),
        "systems": {k: EV.describe(B, y, v) for k, v in sysd.items()},
        "n_candidates_total": n_cand,
    }
    fam = {
        "HF1_ft_rule_minus_p1_rule": EV.paired(B, y, sysd["ft_rule"], sysd["p1_rule"]),
        "HF2_ft_rule_minus_p0_rule": EV.paired(B, y, sysd["ft_rule"], sysd["p0_rule"]),
    }
    out["descriptive_deltas"] = {
        "p1_rule_minus_p0_rule": EV.paired(B, y, sysd["p1_rule"], sysd["p0_rule"]),
        "ft_rule_minus_V0": EV.paired(B, y, sysd["ft_rule"], sysd["V0"]),
        "sensitivity_ft_rule_minus_p1_rule_reused": EV.paired(
            B, y, sysd["ft_rule"], sysd["p1_rule_reused"]
        ),
    }
    return {"block": out, "family_raw": fam, "Bsize": len(B.ug)}, sysd


def filler_block(EV, rows, o11_shipped, o11_oof) -> tuple[dict, dict]:
    from mrcd.service import contextual as CX

    y = np.array([r["filler_ref"] for r in rows])
    B = EV.Boot([r["speaker"] for r in rows], N_BOOT, SEED)
    sysd = {
        "ft_filler": np.array(
            [int(bool(CX.ft_filler_words(r["words"]["ft"]))) for r in rows]
        ),
        "o2026_11": o11_shipped,
        "o2026_11_oof": o11_oof,
        "p0_text": np.array(
            [int(bool(CX.ft_filler_words(r["words"]["p0"]))) for r in rows]
        ),
        "p1_text": np.array(
            [int(bool(CX.ft_filler_words(r["words"]["p1"]))) for r in rows]
        ),
    }
    out = {
        "unit": "DIHANA validation turn (reference: >= 1 f mark)",
        "n_turns": int(len(y)),
        "n_positive_turns": int(y.sum()),
        "systems": {k: EV.describe(B, y, v) for k, v in sysd.items()},
    }
    # turn-level false-positive rate on reference-negative turns
    neg = (y == 0).astype(float)
    out["false_positive_turn_rate"] = {
        k: ratio_block(B, ((v == 1) & (y == 0)).astype(float), neg)
        for k, v in sysd.items()
    }
    # filled-pause tokens per 100 reference words on turns without any f mark (Plan B definition)
    den = np.array(
        [r["n_ref_words"] if r["filler_ref"] == 0 else 0 for r in rows], float
    )
    tok = {
        k: np.array(
            [
                len(CX.ft_filler_words(r["words"][w])) if r["filler_ref"] == 0 else 0
                for r in rows
            ],
            float,
        )
        for k, w in (("ft_text", "ft"), ("p0_text", "p0"), ("p1_text", "p1"))
    }
    ff = {k: ratio_block(B, v, den, 100.0) for k, v in tok.items()}
    ff["o2026_11"] = {
        "value": None,
        "ci95": [None, None],
        "not_available": "o2026_11 emits turn-level decisions, not filled-pause tokens; see false_positive_turn_rate",
    }
    out["false_filler_per_100_words"] = ff
    out["false_filler_deltas"] = {
        "ft_text_minus_p0_text": ratio_delta(
            B, tok["ft_text"], den, tok["p0_text"], den, 100.0
        ),
        "ft_text_minus_p1_text": ratio_delta(
            B, tok["ft_text"], den, tok["p1_text"], den, 100.0
        ),
    }
    out["secondary_filler_f1_ni"] = {}
    for ref in ("o2026_11", "o2026_11_oof"):
        d = EV.paired(B, y, sysd["ft_filler"], sysd[ref])
        f = d["f1"]
        out["secondary_filler_f1_ni"][ref] = {
            **d,
            "non_inferior_f1": bool(
                f["delta"] is not None
                and f["delta"] >= -NI_F1_MARGIN
                and f["ci95"][1] is not None
                and f["ci95"][1] >= 0
            ),
            "rule": "delta F1 >= -0.02 and CI upper >= 0 (protocol section 4)",
        }
    # HF3: recall non-inferiority vs the shipped o2026_11 head
    ba = B.boot_prf(y, sysd["ft_filler"])["recall"]
    bb = B.boot_prf(y, sysd["o2026_11"])["recall"]
    d = (ba - bb)[np.isfinite(ba - bb)]
    pa = B.prf(B.counts(y, sysd["ft_filler"]).sum(0))["recall"]
    pb = B.prf(B.counts(y, sysd["o2026_11"]).sum(0))["recall"]
    hf3 = {
        "delta": _num(pa - pb),
        "ci95": [_num(np.percentile(d, 2.5)), _num(np.percentile(d, 97.5))],
        "p_one_sided_boot": _num((np.sum(d <= -HF3_MARGIN) + 1) / (len(d) + 1)),
        "margin": HF3_MARGIN,
    }
    return {"block": out, "hf3_raw": hf3}, sysd


def prolongation_block(EV, rows, pipe, recs_p0, recs_ft) -> dict:
    y = np.array([r["prolong_ref"] for r in rows])
    B = EV.Boot([r["speaker"] for r in rows], N_BOOT, SEED)
    p11 = np.array([int(v) for v in pipe.predict(recs_p0)["decision"]])
    p13 = np.array([int(v) for v in pipe.predict(recs_ft)["decision"]])
    d = EV.paired(B, y, p13, p11)
    return {
        "unit": "DIHANA validation turn (reference: lengthening_present)",
        "n_turns": int(len(y)),
        "n_positive_turns": int(y.sum()),
        "status": "descriptive; the same shipped o2026_11 head, not refitted, in-sample for these speakers",
        "systems": {
            "o2026_11_on_p0_words": EV.describe(B, y, p11),
            "o2026_13_on_ft_words": EV.describe(B, y, p13),
        },
        "o2026_13_minus_o2026_11": d,
        "degraded": bool(
            d["f1"]["delta"] is not None and d["f1"]["delta"] < -NI_F1_MARGIN
        ),
        "rule": "degraded iff point delta F1 < -0.02 (protocol section 4)",
    }


def identity_block(rows) -> dict:
    out: dict = {}
    pb = C.PLAN_B_CACHE / "asr"
    if not pb.is_dir():
        reason = "Plan B cache not found on this machine"
        return {
            k: {"not_available": reason}
            for k in (
                "p1_fresh_vs_plan_b_p1",
                "p1_reused_vs_plan_b_p1",
                "ft_vs_plan_b_ft_best",
            )
        }

    def txt(ws):
        return " ".join(w["text"] for w in ws)

    def key(ws):
        return [
            (w["text"], round(float(w["start"]), 3), round(float(w["end"]), 3))
            for w in ws
        ]

    for name, arm, mine in (
        ("p1_fresh_vs_plan_b_p1", "P1", "p1"),
        ("p1_reused_vs_plan_b_p1", "P1", "p1_reused"),
        ("ft_vs_plan_b_ft_best", "FT_best", "ft"),
    ):
        common = same_text = same_all = 0
        for r in rows:
            f = pb / arm / f"{r['uid']}.json"
            if not f.exists():
                continue
            other = json.loads(f.read_text("utf8"))["words"]
            common += 1
            same_text += int(txt(other) == txt(r["words"][mine]))
            same_all += int(key(other) == key(r["words"][mine]))
        out[name] = {
            "turns_in_both": common,
            "identical_text": same_text,
            "identical_text_and_times": same_all,
        }
    same = sum(txt(r["words"]["p1"]) == txt(r["words"]["p1_reused"]) for r in rows)
    out["p1_fresh_vs_p1_reused"] = {
        "turns_in_both": len(rows),
        "identical_text": int(same),
    }
    return out


def rtf_block() -> dict:
    sec = aud = 0.0
    n = 0
    for p in (C.CACHE / "asr_ft").glob("*.json"):
        d = json.loads(p.read_text("utf8"))
        sec += d["sec"]
        aud += d["audio_s"]
        n += 1
    return {
        "ft_small_v1_cpu_int8_4_threads": {
            "turns": n,
            "decode_s": round(sec, 1),
            "audio_s": round(aud, 1),
            "rtf": _num(sec / aud) if aud else None,
        },
        "caveat": "wall-clock over the whole decode, measured while the machine was shared with the session; the "
        "latency gate is G4 (service), not this figure",
    }


def env_block() -> dict:
    from importlib import metadata

    def ver(n):
        try:
            return metadata.version(n)
        except Exception:
            return None

    info = json.loads((C.CACHE / "ft_model_info.json").read_text("utf8"))
    return {
        "ft_model_bin_sha256": info["model_bin_sha256"],
        "ft_compute_type": info["compute_type"],
        "ft_device": info["device"],
        "ft_profile_describe": info["profile"],
        "versions": {n: ver(n) for n in ("faster-whisper", "ctranslate2", "numpy")},
        "omp_num_threads": __import__("os").environ.get("OMP_NUM_THREADS"),
    }


def main() -> None:
    sha = C.protocol_sha()
    EV = _load_plan_a_evaluate()
    from mrcd.context_eval.metrics import holm
    from mrcd.service import decision_profile as DP

    rows = load_rows()
    pipes = DP.o_pipelines()  # the verified, shipped, unmodified o2026_11 heads
    recs_p0, recs_ft = records(rows, "p0"), records(rows, "ft")
    o11_fill = np.array(
        [int(v) for v in pipes["filler_word"].predict(recs_p0)["decision"]]
    )
    oof_rows = {
        r["unit_id"]: r
        for r in csv.DictReader(
            open(EV.O_CAMPAIGN / "predictions_filler_word.csv", encoding="utf8")
        )
    }
    assert all(int(oof_rows[r["uid"]]["y"]) == r["filler_ref"] for r in rows), (
        "label mismatch with the O-campaign file"
    )
    o11_oof = np.array([int(oof_rows[r["uid"]]["decision"]) for r in rows])

    rep, _ = repetition_block(EV, rows)
    fill, _ = filler_block(EV, rows, o11_fill, o11_oof)
    # Holm family: HF1, HF2 (two-sided p, superiority) and HF3 (one-sided p, non-inferiority)
    R = [r for r in rows if r["rep_ref"] is not None]
    fam = {
        "HF1_repetition_f1_ft_rule_minus_p1_rule": {
            **rep["family_raw"]["HF1_ft_rule_minus_p1_rule"]["f1"],
            "criterion": "CI lower > 0 and p_Holm < 0.05 (two-sided bootstrap p)",
        },
        "HF2_repetition_f1_ft_rule_minus_p0_rule": {
            **rep["family_raw"]["HF2_ft_rule_minus_p0_rule"]["f1"],
            "criterion": "CI lower > 0 and p_Holm < 0.05 (two-sided bootstrap p)",
        },
        "HF3_filler_recall_ft_filler_minus_o2026_11_NI": {
            **fill["hf3_raw"],
            "p_boot": fill["hf3_raw"]["p_one_sided_boot"],
            "criterion": "CI lower > -0.05 and p_Holm < 0.05 (one-sided bootstrap p of H0 delta <= -0.05)",
        },
    }
    names = list(fam)
    adj = holm([fam[k]["p_boot"] for k in names])
    for k, a in zip(names, adj):
        f = fam[k]
        f["p_holm"] = _num(a)
        lo = f["ci95"][0]
        bound = HF3_MARGIN * -1 if k.startswith("HF3") else 0.0
        f["verdict"] = (
            "supported"
            if (a < 0.05 and lo is not None and lo > bound)
            else "not supported"
        )
    res = {
        "label": LABEL,
        "protocol": "results/plan_f_20261108/PROTOCOL_EXPLORATORY_F.md",
        "protocol_sha256": sha,
        "git_head": git_head(),
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "data": {
            "corpus": "DIHANA folds 1-4, the 36 Plan B validation speakers (fold 5 never indexed)",
            "turns": len(rows),
            "speakers": len({r["speaker"] for r in rows}),
            "word_sources": {
                "p0": "frozen campaign cache",
                "p1": "decoded fresh by this plan (decode_p1.py, transcribe_chunked P1); reused frozen decodes only as a sensitivity (identity_checks)",
                "ft": "decoded by this plan through VerbatimASR(profile='ft_small_v1')",
            },
        },
        "bootstrap": {
            "cluster": "speaker",
            "n_boot": N_BOOT,
            "seed": SEED,
            "ci": "percentile 95 %",
        },
        "holm_family": fam,
        "repetition": rep["block"],
        "filler": fill["block"],
        "prolongation": prolongation_block(
            EV, rows, pipes["prolongation"], recs_p0, recs_ft
        ),
        "identity_checks": identity_block(rows),
        "asr_speed": rtf_block(),
        "environment": env_block(),
        "o2026_11_bias_note": "the shipped o2026_11 filler and prolongation heads were fitted on all DIHANA folds 1-4 "
        "turns including these speakers: in-sample, biased in its favour; o2026_11_oof is the out-of-fold reading",
        "not_available": {
            "functional_state": "o2026_13 has no contextual layer; nothing to report",
            "lexical_filler": "o2026_13 does not emit lexical fillers (este, pues, ...)",
            "timing_accuracy": "scoring is turn-level; event boundaries are not compared",
            "outside_dihana": "no other corpus was evaluated (data rules of the plan)",
            "o2026_11_false_filler_tokens": "o2026_11 emits turn decisions, not tokens",
        },
    }
    (C.OUT / "results_f.json").write_text(
        json.dumps(res, indent=1, ensure_ascii=False) + "\n", "utf8"
    )
    print(
        json.dumps(
            {
                "rep_f1": {
                    k: v["f1"]["value"] for k, v in res["repetition"]["systems"].items()
                },
                "holm": {k: v["verdict"] for k, v in fam.items()},
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
