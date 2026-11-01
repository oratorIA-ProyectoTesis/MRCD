"""Score audio-unit runs (DIHANA turns, AMERESCO segments, EPIC sentences) against ORIGINAL labels. Writes per-unit predictions + metrics.
Comparators on identical units/partitions: always_present, always_absent, mrcd_checkpoint (any window argmax == class),
rule_existing_autolabeler (mrcd/annotation/auto_labeler detectors, default params), rule_asr_lexical (fixed simple rules on MRCD's own ASR; never the marked transcript).
Separate TEXT-TASK rows (T): rule on the human transcript with marks stripped (DIHANA repetition) -- different task, never pooled with the above.
No parameter is fit here. Tuning fold (4) is reported separately and was NOT used to change any rule/threshold. Test (fold 5) untouched.
"""

import csv
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from bl_common import OUT, confusion, cluster_bootstrap, per_group, write_config
from text_baselines import rule_rep, FILL_A, toks

EDIT_TERMS = re.compile(
    r"\b(digo|perd[oó]n|perdona|quiero decir|mejor dicho)\b"
)  # fixed a-priori lexicon, naive
CLASSES = ["filler_word", "prolongation", "repetition", "revision"]


def asr_rules(text):
    t = toks(text)
    return {
        "filler_word": int(any(FILL_A.match(w) for w in t)),
        "repetition": rule_rep(text),
        "revision": int(bool(EDIT_TERMS.search(text.lower()))),
        "prolongation": None,
    }


def load(run_dir, name):
    d = {}
    for fp in sorted(Path(run_dir).glob(f"{name}_shard*.jsonl")):
        for l in fp.read_text("utf8").splitlines():
            if l.strip():
                try:
                    r = json.loads(l)
                except json.JSONDecodeError:  # truncated last line of a killed shard
                    continue
                d[r["unit_id"]] = r
    return d


def score(rows, classes, comps, partition_key, corpus, n_boot=2000):
    out = []
    for cls in classes:
        for part in sorted({r[partition_key] for r in rows}) + ["all"]:
            sel = [r for r in rows if part == "all" or r[partition_key] == part]
            y = [r[cls] for r in sel]
            g = [r["group"] for r in sel]
            for cname, fn in comps.items():
                p = [fn(r, cls) for r in sel]
                if any(v is None for v in p):
                    continue
                m = confusion(y, p)
                m.update(
                    {
                        "corpus": corpus,
                        "class": cls,
                        "comparator": cname,
                        "partition": part,
                        "n_groups": len(set(g)),
                        "n_groups_pos": len({gg for gg, yy in zip(g, y) if yy}),
                        "stable_ge30_pos": m["support_pos"] >= 30,
                    }
                )
                if m["support_pos"] > 0 and len(set(g)) > 1:
                    m["ci95_by_group"] = cluster_bootstrap(y, p, g, n_boot)
                out.append(m)
    return out


COMPS = {
    "always_present": lambda r, c: 1,
    "always_absent": lambda r, c: 0,
    "mrcd_checkpoint": lambda r, c: r["model"][c],
    "rule_existing_autolabeler": lambda r, c: r["rule"][c],
    "rule_asr_lexical": lambda r, c: r["asr"][c],
}


def main():
    allm = []
    extra = {}
    # ------------------------------------------------------------- DIHANA
    D = OUT / "dihana"
    T = {
        r["turn_id"]: r
        for r in csv.DictReader((D / "turns.csv").open(encoding="utf-8-sig"))
    }
    run = load(D / "mrcd_run", "dihana")
    rows = []
    for uid, r in run.items():
        t = T[uid]
        rows.append(
            {
                "unit_id": uid,
                "group": t["speaker"],
                "fold": int(t["fold"]),
                "split": t["split"],
                "padded": r["padded"],
                "dur_s": r["target"][1] - r["target"][0],
                "filler_word": int(t["filler_present"]),
                "prolongation": int(t["lengthening_present"]),
                "repetition": int(t["repetition_present"]),
                "revision": int(t["reformulation_present"]),
                "model": r["model_pred"],
                "rule": r["rule_pred"],
                "asr": asr_rules(r["asr_text"]),
                "asr_text": r["asr_text"],
                "ref_text": t["ref_text_marks_stripped"],
                "txt_rep": rule_rep(t["ref_text_marks_stripped"]),
            }
        )
    allm += score(rows, CLASSES, COMPS, "split", "DIHANA(turn,audio+ASR)")
    allm += score(
        [r for r in rows if not r["padded"]],
        CLASSES,
        COMPS,
        "split",
        "DIHANA(turn,audio+ASR) turns>=3s",
    )
    allm += score(
        rows,
        ["repetition"],
        {"rule_adjacent_repetition_on_transcript": lambda r, c: r["txt_rep"]},
        "split",
        "DIHANA T-task: human transcript marks stripped",
    )
    with (D / "predictions_per_turn.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as f:
        cols = (
            ["unit_id", "speaker", "fold", "split", "padded", "dur_s"]
            + [f"y_{c}" for c in CLASSES]
            + [f"{k}_{c}" for k in ["mrcd", "rule", "asr"] for c in CLASSES]
            + ["asr_text", "ref_text"]
        )
        w = csv.writer(f)
        w.writerow(cols)
        for r in sorted(rows, key=lambda r: r["unit_id"]):
            w.writerow(
                [
                    r["unit_id"],
                    r["group"],
                    r["fold"],
                    r["split"],
                    int(r["padded"]),
                    round(r["dur_s"], 2),
                ]
                + [r[c] for c in CLASSES]
                + [r["model"][c] for c in CLASSES]
                + [r["rule"][c] for c in CLASSES]
                + [r["asr"][c] for c in CLASSES]
                + [r["asr_text"], r["ref_text"]]
            )
    extra["dihana_units_scored"] = {
        s: sum(r["split"] == s for r in rows) for s in ["dev", "tune", "test"]
    }
    extra["dihana_speakers_scored"] = {
        s: len({r["group"] for r in rows if r["split"] == s})
        for s in ["dev", "tune", "test"]
    }
    tt = [r for r in rows if r["split"] == "test"]
    extra["dihana_per_speaker_test_mrcd"] = {
        c: per_group(
            [r[c] for r in tt], [r["model"][c] for r in tt], [r["group"] for r in tt]
        )
        for c in CLASSES
    }
    # ------------------------------------------------------------- AMERESCO
    A = OUT / "ameresco"
    arun = load(A / "mrcd_run", "ameresco")
    refs = {
        r["unit_id"]: r
        for r in csv.DictReader((A / "references.csv").open(encoding="utf-8-sig"))
    }
    arows = []
    for uid, r in arun.items():
        ref = refs[uid]
        arows.append(
            {
                "unit_id": uid,
                "group": ref["group"],
                "split": "pilot32" if ref["in_pilot32"] == "True" else "new66",
                "prolongation": int(ref["prolongation_present"]),
                "model": r["model_pred"],
                "rule": r["rule_pred"],
                "asr": asr_rules(r["asr_text"]),
                "asr_text": r["asr_text"],
                "max_prob": r["max_prob"],
            }
        )
    acomps = {k: v for k, v in COMPS.items() if k != "rule_asr_lexical"}
    allm += score(
        arows, ["prolongation"], acomps, "split", "AMERESCO(segment 3-8 s, audio+ASR)"
    )
    perconv = {}
    for g in sorted({r["group"] for r in arows}):
        sel = [r for r in arows if r["group"] == g]
        y = [r["prolongation"] for r in sel]
        perconv[g] = {
            c: confusion(y, [fn(r, "prolongation") for r in sel])
            for c, fn in acomps.items()
        }
    extra["ameresco_per_conversation"] = perconv
    with (A / "predictions_per_segment.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as f:
        w = csv.writer(f)
        w.writerow(
            [
                "unit_id",
                "conversation",
                "subset",
                "y_prolongation",
                "mrcd_checkpoint",
                "rule_existing",
                "max_prob_prolongation",
                "asr_text",
            ]
        )
        for r in sorted(arows, key=lambda r: r["unit_id"]):
            w.writerow(
                [
                    r["unit_id"],
                    r["group"],
                    r["split"],
                    r["prolongation"],
                    r["model"]["prolongation"],
                    r["rule"]["prolongation"],
                    round(r["max_prob"]["prolongation"], 4),
                    r["asr_text"],
                ]
            )
    pil = Path(
        r"C:\zArnes-tesis\mrcd\results\reproduccion_piloto_20261030\audio_predictions.jsonl"
    )
    pm = {
        x["recording_id"] + "_" + x["source_annotation_id"]: x
        for x in map(json.loads, pil.read_text("utf8").splitlines())
    }
    same = [
        (
            pm[r["unit_id"]]["model_pred"] == r["model"]["prolongation"],
            pm[r["unit_id"]]["heuristic_pred"] == r["rule"]["prolongation"],
        )
        for r in arows
        if r["unit_id"] in pm
    ]
    extra["ameresco_repro_vs_pilot"] = {
        "n": len(same),
        "model_pred_same": sum(a for a, b in same),
        "heuristic_same": sum(b for a, b in same),
    }
    # ------------------------------------------------------------- EPIC
    E = OUT / "epic"
    if (E / "mrcd_run").exists():
        erun = load(E / "mrcd_run", "epic")
        eref = {
            r["unit_id"]: r
            for r in csv.DictReader((E / "references.csv").open(encoding="utf-8-sig"))
        }
        erows = [
            {
                "unit_id": u,
                "group": eref[u]["group"],
                "split": "dev(all 136)",
                "filler_word": int(eref[u]["filler_present"]),
                "model": r["model_pred"],
                "rule": r["rule_pred"],
                "asr": asr_rules(r["asr_text"]),
                "asr_text": r["asr_text"],
                "src": eref[u]["source_text"],
                "codex_split": eref[u]["split"],
            }
            for u, r in erun.items()
        ]
        allm += score(
            erows,
            ["filler_word"],
            COMPS,
            "split",
            "EPIC v2 (sentence; 136 Codex set = DEVELOPMENT, not test)",
        )
        with (E / "predictions_per_sentence.csv").open(
            "w", encoding="utf-8-sig", newline=""
        ) as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "unit_id",
                    "actor",
                    "codex_split",
                    "y_filler",
                    "mrcd",
                    "rule_existing",
                    "rule_asr_lexical",
                    "asr_text",
                    "source_text",
                ]
            )
            for r in sorted(erows, key=lambda r: r["unit_id"]):
                w.writerow(
                    [
                        r["unit_id"],
                        r["group"],
                        r["codex_split"],
                        r["filler_word"],
                        r["model"]["filler_word"],
                        r["rule"]["filler_word"],
                        r["asr"]["filler_word"],
                        r["asr_text"],
                        r["src"],
                    ]
                )
    (OUT / "metrics_audio.json").write_text(
        json.dumps({"metrics": allm, "extra": extra}, indent=1, ensure_ascii=True)
    )
    write_config(
        OUT / "eval_audio_run",
        "evaluate_audio",
        __file__,
        {
            "n_boot": 2000,
            "bootstrap_unit": "speaker (DIHANA/EPIC), conversation (AMERESCO)",
        },
        {},
    )
    print(
        json.dumps(extra["ameresco_repro_vs_pilot"]),
        json.dumps(extra["dihana_units_scored"]),
    )
    for m in allm:
        if (
            m["corpus"].startswith(("DIHANA(turn,audio+ASR)", "AMERESCO(", "EPIC"))
            and not m["corpus"].endswith("3s")
            and m["partition"]
            in ("test", "tune", "all", "dev(all 136)", "pilot32", "new66")
        ):
            ci = m.get("ci95_by_group", {}).get("f1")
            print(
                f"{m['corpus'][:8]:8s} {m['class'][:10]:10s} {m['comparator'][:25]:25s} {m['partition'][:6]:6s} n={m['n']:5d} pos={m['support_pos']:4d} P={m['precision']:.2f} R={m['recall']:.2f} F1={m['f1']:.2f} CI={[round(x, 2) for x in ci] if ci else None}"
            )


if __name__ == "__main__":
    main()
