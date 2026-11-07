"""Survival on larger ASR models (results/asr_survival_models_20261104/PROTOCOL.md). Writes metrics.json and unit_flags.csv
(aggregates and 0/1 flags only, no corpus text).

Reuses the exact survival / rule-FP / loop / bootstrap / Holm code of S2 (scripts/s2_noloop/analyze.py) and of
scripts/control_noprompt/analyze_control.py. Cached small-model transcripts (P0, P1, C1, p_none) are read, never re-decoded.
usage: python analyze_models.py            (CPU, mrcd venv)
       ASM_DRYRUN=<dir> python analyze_models.py   (code check only: small-model dirs stand in for the missing systems)
"""

import os
import sys

NUMPY_FIRST = "numpy" in sys.modules
os.environ["OMP_NUM_THREADS"] = "4"
os.environ.setdefault("HF_HUB_OFFLINE", "1")
if NUMPY_FIRST:
    raise SystemExit("numpy imported before the OMP pin")
from pathlib import Path  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [
    str(REPO),
    str(REPO / "scripts" / "cierre3b"),
    str(REPO / "scripts" / "s2_noloop"),
]
import importlib  # noqa: E402

_S2 = importlib.import_module(
    "analyze"
)  # must be imported BEFORE numpy (OMP pin check)
NBOOT, SEED, boot_ratio, holm = _S2.NBOOT, _S2.SEED, _S2.boot_ratio, _S2.holm

import csv  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

from asr_analyze import align, norm_text  # noqa: E402
from mrcd.asr_profiles import detect_asr_anomalies  # noqa: E402
from mrcd.asr_recovery.common import LITERAL_SUBTYPES, OUT, load_pool  # noqa: E402
from mrcd.asr_recovery.noloop import is_loop_clip, loop_excess  # noqa: E402
from mrcd.asr_recovery.stats import corpus_wer  # noqa: E402
from mrcd.context_eval.reference import _norm_tokens  # noqa: E402
from mrcd.context_eval.survival import turn_survives  # noqa: E402
from mrcd.context_eval.textfeat import rule_rep  # noqa: E402
from mrcd.context_eval.transcript_reference import is_nonlexical_filler  # noqa: E402

DEST = REPO / "results" / "asr_survival_models_20261104"
RUNS = DEST / "runs" / "asr"
NONLEX_TAGS = ("f:a", "f:e", "f:m", "f:sin_identificar")

# --- CrisperWhisper verbatim tags (PROTOCOL section 4.3, frozen before decoding) -----------------------------------
TAG_RE = re.compile(r"\[([^\[\]]*)\]")
FILLED_PAUSE_TAGS = frozenset(
    {"uh", "um", "eh", "em", "er", "ah", "ahh", "hm", "hmm", "mm", "mmm"}
)


def cw_score_text(raw: str) -> str:
    """Primary scoring text: every bracketed tag removed (tags are markup, not words of the reference text A)."""
    return " ".join(TAG_RE.sub(" ", raw).split())


def cw_tagtext(raw: str) -> str:
    """Sensitivity text: tags kept as ordinary words ("[UH]" -> "uh")."""
    return " ".join(TAG_RE.sub(lambda m: " " + m.group(1) + " ", raw).split())


def cw_filled_tags(raw: str) -> int:
    return sum(
        m.group(1).strip().lower() in FILLED_PAUSE_TAGS for m in TAG_RE.finditer(raw)
    )


def cw_other_tags(raw: str) -> int:
    return sum(
        m.group(1).strip().lower() not in FILLED_PAUSE_TAGS
        for m in TAG_RE.finditer(raw)
    )


# --- systems ---------------------------------------------------------------------------------------------------------
C = REPO / "results" / "control_noprompt_20261103" / "runs" / "asr" / "p_none"
SYSTEMS = {  # name -> (directory, kind)
    "small_P0": (OUT / "asr" / "P0", "words"),
    "small_P1": (OUT / "asr" / "P1", "words"),
    "small_C1": (OUT / "asr" / "C1", "words"),
    "small_pnone": (C, "words"),
    "lv3_P0": (RUNS / "lv3_P0", "words"),
    "lv3_pnone": (RUNS / "lv3_p_none", "words"),
    "lv3_P1": (RUNS / "lv3_P1", "words"),
    "lv3_C1": (RUNS / "lv3_C1", "words"),
    "cw_large": (RUNS / "cw_large", "cw"),
    "cw_large_tagtext": (
        RUNS / "cw_large",
        "cw_tagtext",
    ),  # sensitivity: same decodes, tags kept as words
}
DRYRUN = os.environ.get("ASM_DRYRUN")
if DRYRUN:  # code check only (never a result)
    for k in ("lv3_P0", "lv3_pnone", "lv3_P1", "lv3_C1"):
        SYSTEMS[k] = (SYSTEMS["small_" + k.split("_")[1]][0], "words")
    SYSTEMS["cw_large"] = (OUT / "asr" / "P0", "words")
    SYSTEMS["cw_large_tagtext"] = (OUT / "asr" / "P0", "words")

# Frozen contrasts (PROTOCOL section 6): Holm family = FAMILY_CONTRASTS x FAMILY_METRICS = 10 x 2 = 20.
FAMILY_CONTRASTS = (
    ("lv3_P0", "small_P0"),
    ("lv3_P1", "small_P1"),
    ("lv3_C1", "small_C1"),
    ("lv3_pnone", "small_pnone"),
    ("cw_large", "small_P0"),
    ("cw_large", "small_pnone"),
    ("lv3_P1", "lv3_P0"),
    ("lv3_P0", "lv3_pnone"),
    ("lv3_P1", "lv3_pnone"),
    ("lv3_P1", "lv3_C1"),
)
FAMILY_METRICS = ("survival_rep", "survival_filler")
DESCRIPTIVE_CONTRASTS = (("cw_large_tagtext", "cw_large"),)
ALL_METRICS = (
    "survival_rep",
    "survival_filler",
    "survival_rev",
    "wer",
    "wer_neg",
    "ins_neg",
    "loop_rate",
    "rule_fp",
    "filler_emission_absent",
    "adj_pairs_neg",
)
THRESH_PERSISTS, THRESH_MAJORITY = 0.75, 0.50


def sha_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def n_fillers(text):
    t = _norm_tokens(text)
    return sum(
        is_nonlexical_filler(x, t[i + 1] if i + 1 < len(t) else None)
        for i, x in enumerate(t)
    )


def adjacent_pairs(text):
    t = _norm_tokens(text)
    return sum(a == b for a, b in zip(t, t[1:]))


def tier(ci):
    """Frozen reading of a survival interval (PROTOCOL 6.1)."""
    lo, hi = ci
    if hi < THRESH_MAJORITY:
        return "majority erased (CI upper < 0.50)"
    if hi < THRESH_PERSISTS:
        return "erasure persists (CI upper < 0.75)"
    if lo >= THRESH_PERSISTS:
        return "substantially attenuated (CI lower >= 0.75)"
    return "inconclusive"


def main():
    t0 = time.time()
    frame = json.loads((OUT / "subset.json").read_text("utf8"))["frame"]
    ids = [s["unit_id"] for s in frame]
    kind = {s["unit_id"]: s["kind"] for s in frame}
    assert len(ids) == 1025 and all(int(s["fold"]) in (1, 2, 3, 4) for s in frame)
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
                if k in NONLEX_TAGS
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
    refA = {u: norm_text(pool[u]["text_A"]) for u in ids}
    g = {u: pool[u]["speaker"] for u in ids}

    per, extra = {}, {}
    for v, (d, k) in SYSTEMS.items():
        files = [d / f"{u}.json" for u in ids]
        missing = [p.name for p in files if not p.exists()]
        if missing:
            raise SystemExit(f"{v}: {len(missing)} units not decoded")
        recs = {u: json.loads(p.read_text("utf8")) for u, p in zip(ids, files)}
        r, ex = {}, {"filled_tags": 0, "other_tags": 0, "units_with_tag": 0}
        for u in ids:
            rec = recs[u]
            if k == "words":
                t, tag_fill, empty = (
                    " ".join(w["text"] for w in rec["words"]),
                    0,
                    not rec["words"],
                )
                n_words = len(rec["words"])
            else:
                raw = rec["text"]
                t = cw_score_text(raw) if k == "cw" else cw_tagtext(raw)
                tag_fill = cw_filled_tags(raw)
                ex["filled_tags"] += tag_fill
                ex["other_tags"] += cw_other_tags(raw)
                ex["units_with_tag"] += int(bool(TAG_RE.search(raw)))
                empty = not t.strip()
                n_words = len(_norm_tokens(t))
            h = norm_text(t)
            s_, dl, ins, _ = align(refA[u], h)
            r[u] = {
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
                "n_fill": n_fillers(t) + tag_fill,
                "rule_fp": int(rule_rep(t)),
                "err": s_ + dl + ins,
                "ins": ins,
                "n_ref": len(refA[u]),
                "loop": int(is_loop_clip(h, refA[u])),
                "excess": loop_excess(h, refA[u]),
                "lenient": int(
                    any(
                        x["kind"] == "repeated_ngram_loop"
                        for x in detect_asr_anomalies(t, "")
                    )
                ),
                "adj": adjacent_pairs(t),
                "empty": int(empty),
                "n_words": n_words,
                "sec": rec.get("sec"),
                "dur": rec["n_samples"] / 16000,
                "text_neg": t if kind[u] == "neg" else None,
            }
        per[v], extra[v] = r, ex

    def arrays(v, metric):
        R = per[v]
        spec = {
            "survival_rep": (rep_pos, lambda u: R[u]["survival_rep"], lambda u: 1),
            "survival_rev": (rev_pos, lambda u: R[u]["survival_rev"], lambda u: 1),
            "survival_filler": (f_pos, lambda u: int(R[u]["n_fill"] > 0), lambda u: 1),
            "filler_emission_absent": (
                f_abs,
                lambda u: int(R[u]["n_fill"] > 0),
                lambda u: 1,
            ),
            "wer": (ids, lambda u: R[u]["err"], lambda u: R[u]["n_ref"]),
            "wer_neg": (neg, lambda u: R[u]["err"], lambda u: R[u]["n_ref"]),
            "ins_neg": (neg, lambda u: R[u]["ins"], lambda u: R[u]["n_ref"]),
            "loop_rate": (ids, lambda u: R[u]["loop"], lambda u: 1),
            "rule_fp": (neg, lambda u: R[u]["rule_fp"], lambda u: 1),
            "adj_pairs_neg": (neg, lambda u: R[u]["adj"], lambda u: 1),
        }[metric]
        us = spec[0]
        return (
            us,
            [g[u] for u in us],
            np.array([spec[1](u) for u in us], float),
            np.array([spec[2](u) for u in us], float),
        )

    summary = {}
    for v in SYSTEMS:
        R, s = per[v], {}
        for m in ALL_METRICS:
            us, gg, n, d = arrays(v, m)
            r = boot_ratio(gg, n, np.zeros(len(n)), d)
            s[m] = {
                "k": float(n.sum()),
                "n": float(d.sum()),
                "n_units": len(us),
                "n_groups": r["n_groups"],
                "value": r["delta"],
                "ci": r["ci"],
            }
        for m in ("survival_rep", "survival_rev", "survival_filler"):
            s[m]["reading"] = tier(s[m]["ci"])
        s["loop_clips"] = int(sum(R[u]["loop"] for u in ids))
        s["loop_excess_tokens"] = int(sum(R[u]["excess"] for u in ids))
        s["lenient_loop_flag_units"] = int(sum(R[u]["lenient"] for u in ids))
        s["wer_vs_ref_A_on_negatives_c1_function"] = corpus_wer(
            [(pool[u]["text_A"], R[u]["text_neg"]) for u in neg]
        )
        s["units_with_adjacent_pair_on_negatives"] = int(
            sum(R[u]["adj"] > 0 for u in neg)
        )
        s["filler_tokens_total"] = int(sum(R[u]["n_fill"] for u in ids))
        s["empty_outputs"] = int(sum(R[u]["empty"] for u in ids))
        s["mean_words_per_unit"] = float(np.mean([R[u]["n_words"] for u in ids]))
        secs = [R[u]["sec"] for u in ids if R[u]["sec"] is not None]
        s["decode_seconds_total"] = float(sum(secs)) if secs else None
        s["decode_rtf_per_unit_sum"] = (
            float(sum(secs) / sum(R[u]["dur"] for u in ids)) if secs else None
        )
        s["verbatim_tags"] = extra[v] if v.startswith("cw") and not DRYRUN else None
        summary[v] = s

    def paired(a, b, metrics):
        out = {}
        for m in metrics:
            us1, g1, n1, d1 = arrays(a, m)
            us0, _, n0, d0 = arrays(b, m)
            assert us1 == us0 and np.array_equal(d1, d0)
            r = boot_ratio(g1, n1, n0, d1)
            if (
                m.startswith("survival")
                or m.endswith("_absent")
                or m in ("rule_fp", "loop_rate")
            ):
                r["gained"] = int(((n1 > 0) & (n0 == 0)).sum())
                r["lost"] = int(((n1 == 0) & (n0 > 0)).sum())
            out[m] = r
        return out

    contrasts = {}
    for a, b in FAMILY_CONTRASTS:
        contrasts[f"{a}-{b}"] = paired(a, b, ALL_METRICS)
    for a, b in DESCRIPTIVE_CONTRASTS:
        contrasts[f"{a}-{b}"] = paired(a, b, ALL_METRICS)
    fam = {
        f"{a}-{b}|{m}": contrasts[f"{a}-{b}"][m]["p"]
        for a, b in FAMILY_CONTRASTS
        for m in FAMILY_METRICS
    }
    assert len(fam) == 20
    adj = holm(fam, len(fam))
    for k, p in adj.items():
        c, m = k.split("|")
        contrasts[c][m]["p_holm_family20"] = p

    # Decision rules (PROTOCOL section 6.2-6.3)
    def sig(c, m, sign):
        r = contrasts[c][m]
        h = r.get("p_holm_family20")
        if h is None or r["ci"][0] is None:
            return False
        return h < 0.05 and (r["ci"][0] > 0 if sign > 0 else r["ci"][1] < 0)

    erasure = {
        v: {
            m: summary[v][m]["reading"]
            for m in ("survival_rep", "survival_rev", "survival_filler")
        }
        for v in (
            "small_P0",
            "small_pnone",
            "lv3_P0",
            "lv3_pnone",
            "lv3_P1",
            "lv3_C1",
            "cw_large",
        )
    }
    rep_persist = {
        v: summary[v]["survival_rep"]["ci"][1] is not None
        and summary[v]["survival_rep"]["ci"][1] < THRESH_PERSISTS
        for v in ("lv3_P0", "cw_large")
    }
    decisions = {
        "D1_erasure_reading_per_system": erasure,
        "D2_literal_repetition_erasure_persists_in_lv3_P0_and_cw_large": rep_persist,
        "D2_generalizes": "yes"
        if all(rep_persist.values())
        else ("partly" if any(rep_persist.values()) else "no"),
        "D3_P1_trade_off_replicates_on_lv3": {
            "rep_P1_gt_P0": sig("lv3_P1-lv3_P0", "survival_rep", +1),
            "filler_P1_lt_P0": sig("lv3_P1-lv3_P0", "survival_filler", -1),
        },
    }
    d3 = decisions["D3_P1_trade_off_replicates_on_lv3"]
    d3["replicates"] = (
        "yes" if all(d3.values()) else ("partly" if any(d3.values()) else "no")
    )

    chk = {}
    for name in ("env_check_lv3_P0.json", "env_check_cw_large.json"):
        p = DEST / "runs" / name
        chk[name] = (
            json.loads(p.read_text("utf8")) if p.exists() and not DRYRUN else None
        )
    res = {
        "protocol_sha256": None
        if DRYRUN
        else json.loads((DEST / "PROTOCOL.sha256.json").read_text("utf8"))["sha256"],
        "dryrun": bool(DRYRUN),
        "frame": {
            "units": len(ids),
            "rep_literal_pos": len(rep_pos),
            "revision_pos": len(rev_pos),
            "sampled_negatives": len(neg),
            "filler_pos": len(f_pos),
            "filler_absent": len(f_abs),
            "speakers": len(set(g.values())),
            "dihana_folds": [1, 2, 3, 4],
        },
        "determinism_checks": {
            k: (None if v is None else {"n": v["n"], "identical": v["identical"]})
            for k, v in chk.items()
        },
        "summary": summary,
        "contrasts": contrasts,
        "holm_family": sorted(fam),
        "decisions": decisions,
        "bootstrap": {"seed": SEED, "n_boot": NBOOT, "groups": "speaker"},
        "thresholds": {"persists": THRESH_PERSISTS, "majority": THRESH_MAJORITY},
        "inputs_sha256": {
            v: hashlib.sha256(
                "".join(
                    sorted(sha_file(SYSTEMS[v][0] / f"{u}.json") for u in ids)
                ).encode()
            ).hexdigest()
            for v in SYSTEMS
        },
        "code_sha256": {
            str(p.relative_to(REPO)).replace("\\", "/"): sha_file(p)
            for p in (
                Path(__file__),
                REPO / "scripts/asr_survival_models/decode_largev3.py",
                REPO / "scripts/asr_survival_models/decode_crisper.py",
                REPO / "scripts/s2_noloop/analyze.py",
                REPO / "mrcd/asr_recovery/decode.py",
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
        names = list(SYSTEMS)
        w.writerow(
            ["unit_id", "kind", "rep_pos", "rev_pos", "filler_pos"]
            + [f"{n}:{m}" for n in names for m in ("rep", "rev", "fill")]
        )
        for u in ids:
            w.writerow(
                [
                    u,
                    kind[u],
                    int(u in set(rep_pos)),
                    int(u in set(rev_pos)),
                    fill_pos[u],
                ]
                + [
                    x
                    for n in names
                    for x in (
                        per[n][u]["survival_rep"],
                        per[n][u]["survival_rev"],
                        int(per[n][u]["n_fill"] > 0),
                    )
                ]
            )
    for v in SYSTEMS:
        s = summary[v]
        print(
            f"{v:18s} rep {s['survival_rep']['k']:.0f}/{s['survival_rep']['n']:.0f}={s['survival_rep']['value']:.3f} {s['survival_rep']['ci']} "
            f"rev {s['survival_rev']['k']:.0f}/{s['survival_rev']['n']:.0f} fill {s['survival_filler']['k']:.0f}/{s['survival_filler']['n']:.0f} wer {s['wer']['value']:.3f}"
        )
    for a, b in FAMILY_CONTRASTS:
        for m in FAMILY_METRICS:
            r = contrasts[f"{a}-{b}"][m]
            print(
                f"{a}-{b:12s} {m:16s} d={r['delta']:+.4f} ci=[{r['ci'][0]:+.4f},{r['ci'][1]:+.4f}] holm={r.get('p_holm_family20', float('nan')):.4f}"
            )
    print(json.dumps(decisions, indent=1))


if __name__ == "__main__":
    main()
