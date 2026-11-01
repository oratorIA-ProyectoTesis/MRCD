"""FP/FN analysis for the audio-unit runs. Creates NO labels: candidate causes are computed from existing fields only
(ASR text vs the marks-stripped human transcript, turn duration/padding, co-occurring original tags). Examples are seeded samples.
"""

import csv
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from bl_common import OUT, SEED, write_config
from evaluate_audio import load, asr_rules, CLASSES
from text_baselines import rule_rep

rng = random.Random(SEED)


def pick(lst, k=5):
    return rng.sample(lst, min(k, len(lst)))


def main():
    D = OUT / "dihana"
    T = {
        r["turn_id"]: r
        for r in csv.DictReader((D / "turns.csv").open(encoding="utf-8-sig"))
    }
    run = load(D / "mrcd_run", "dihana")
    ycol = {
        "filler_word": "filler_present",
        "prolongation": "lengthening_present",
        "repetition": "repetition_present",
        "revision": "reformulation_present",
    }
    res = {
        "note": "computed on all scored DIHANA turns; no labels created",
        "classes": {},
    }
    for c in CLASSES:
        pos = [u for u in run if int(T[u][ycol[c]])]
        neg = [u for u in run if not int(T[u][ycol[c]])]
        fn = [u for u in pos if not run[u]["model_pred"][c]]
        fp = [u for u in neg if run[u]["model_pred"][c]]
        d = {
            "n_pos": len(pos),
            "n_neg": len(neg),
            "mrcd_FN": len(fn),
            "mrcd_FP": len(fp),
            "FN_padded_lt3s_frac": sum(run[u]["padded"] for u in fn) / max(1, len(fn)),
            "pos_padded_lt3s_frac": sum(run[u]["padded"] for u in pos)
            / max(1, len(pos)),
            "TN_padded_frac": sum(run[u]["padded"] for u in neg) / max(1, len(neg)),
            "FP_padded_frac": sum(run[u]["padded"] for u in fp) / max(1, len(fp)),
        }

        # co-occurring original tags among FP vs all negatives
        def tagrate(us, tag):
            return sum(tag in json.loads(T[u]["raw_tag_counts"]) for u in us) / max(
                1, len(us)
            )

        d["FP_tag_rates_vs_neg"] = {
            t: [round(tagrate(fp, t), 3), round(tagrate(neg, t), 3)]
            for t in [
                "p",
                "b",
                "d:rellenar",
                "l:cortada",
                "l:mal_pronunciada",
                "f:e",
                "a",
                "r:repeticion",
            ]
            if t != {"filler_word": "f:e"}.get(c, "")
        }
        if c == "repetition":
            has_txt = [u for u in pos if rule_rep(T[u]["ref_text_marks_stripped"])]
            d["pos_with_adjacent_repetition_in_human_transcript"] = len(has_txt) / max(
                1, len(pos)
            )
            d["of_those_ASR_text_also_has_adjacent_repetition"] = sum(
                asr_rules(run[u]["asr_text"])["repetition"] for u in has_txt
            ) / max(1, len(has_txt))
            d["ASR_deleted_repetition_FN_count"] = sum(
                1
                for u in fn
                if rule_rep(T[u]["ref_text_marks_stripped"])
                and not rule_rep(run[u]["asr_text"])
            )
        if c == "filler_word":
            d["pos_where_ASR_text_has_vocalic_filler_token"] = sum(
                asr_rules(run[u]["asr_text"])["filler_word"] for u in pos
            ) / max(1, len(pos))
            d["neg_where_ASR_text_has_vocalic_filler_token"] = sum(
                asr_rules(run[u]["asr_text"])["filler_word"] for u in neg
            ) / max(1, len(neg))
        ex = lambda us: [
            {
                "turn": u,
                "ref_marks_stripped": T[u]["ref_text_marks_stripped"],
                "asr": run[u]["asr_text"],
                "dur_s": round(run[u]["target"][1] - run[u]["target"][0], 1),
                "window_labels": run[u]["window_labels"][:6],
                "orig_tags": T[u]["raw_tag_counts"],
            }
            for u in pick(us)
        ]
        d["examples_FN"] = ex(fn)
        d["examples_FP"] = ex(fp)
        res["classes"][c] = d
    # AMERESCO
    A = OUT / "ameresco"
    arun = load(A / "mrcd_run", "ameresco")
    refs = {
        r["unit_id"]: int(r["prolongation_present"])
        for r in csv.DictReader((A / "references.csv").open(encoding="utf-8-sig"))
    }
    fn = [u for u in arun if refs[u] and not arun[u]["model_pred"]["prolongation"]]
    fp = [u for u in arun if not refs[u] and arun[u]["model_pred"]["prolongation"]]
    res["ameresco"] = {
        "mrcd_FN": len(fn),
        "mrcd_FP": len(fp),
        "FN_max_prob_mean": sum(arun[u]["max_prob"]["prolongation"] for u in fn)
        / max(1, len(fn)),
        "examples_FN": [
            {
                "unit": u,
                "asr": arun[u]["asr_text"],
                "max_p": round(arun[u]["max_prob"]["prolongation"], 3),
                "rule_events": arun[u]["rule_events"]["prolongation"],
            }
            for u in pick(fn)
        ],
        "examples_FP": [
            {
                "unit": u,
                "asr": arun[u]["asr_text"],
                "max_p": round(arun[u]["max_prob"]["prolongation"], 3),
            }
            for u in pick(fp)
        ],
    }
    # EPIC
    E = OUT / "epic"
    if (E / "mrcd_run").exists():
        erun = load(E / "mrcd_run", "epic")
        eref = {
            r["unit_id"]: r
            for r in csv.DictReader((E / "references.csv").open(encoding="utf-8-sig"))
        }
        pos = [u for u in erun if int(eref[u]["filler_present"])]
        neg = [u for u in erun if not int(eref[u]["filler_present"])]
        fn = [u for u in pos if not erun[u]["model_pred"]["filler_word"]]
        fp = [u for u in neg if erun[u]["model_pred"]["filler_word"]]
        res["epic"] = {
            "mrcd_FN": len(fn),
            "mrcd_FP": len(fp),
            "pos_ASR_has_vocalic_filler_token": sum(
                asr_rules(erun[u]["asr_text"])["filler_word"] for u in pos
            )
            / max(1, len(pos)),
            "examples_FN": [
                {
                    "unit": u,
                    "src": eref[u]["source_text"][:200],
                    "asr": erun[u]["asr_text"][:200],
                }
                for u in pick(fn)
            ],
            "examples_FP": [
                {
                    "unit": u,
                    "src": eref[u]["source_text"][:200],
                    "asr": erun[u]["asr_text"][:200],
                    "windows": erun[u]["window_labels"][:6],
                }
                for u in pick(fp)
            ],
        }
    (OUT / "error_analysis.json").write_text(
        json.dumps(res, indent=1, ensure_ascii=True)
    )
    write_config(
        OUT / "error_analysis_run", "error_analysis", __file__, {"seed": SEED}, {}
    )
    for c, d in res["classes"].items():
        print(
            c,
            {
                k: (round(v, 3) if isinstance(v, float) else v)
                for k, v in d.items()
                if not k.startswith("examples") and k != "FP_tag_rates_vs_neg"
            },
        )


if __name__ == "__main__":
    main()
