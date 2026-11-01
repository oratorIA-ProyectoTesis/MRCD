"""TEXT-ONLY task (T): fixed lexical rules on text with label notation removed. disfluentES (sentence, partition=speaker) and ES-Port (utterance, partition=dialogue).
Partitions are fixed BEFORE scoring by sha256(group id) % 5: buckets 0-2 = dev, 3 = tune, 4 = test. Rules are fixed (nothing is fit), so test is simply untouched.
Not comparable with audio tasks (different units, population, text source = human transcript, not ASR).
Same comparators on the same units: always-present, always-absent, fixed rule(s).
Labels: disfluentES REP* / FILL* (annotated records only, POS_* ambiguous sentences excluded as in the pilot); ES-Port '< >' = repetition U false start (joint, NOT split).
ES-Port filler is NOT evaluated: removing the '<% >' mark removes the token, and keeping the word would make the rule a lookup of the label itself.
"""

import csv
import hashlib
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from bl_common import (
    CORPORA,
    OUT,
    confusion,
    cluster_bootstrap,
    write_config,
    sha256_file,
)

D = OUT / "text"
D.mkdir(parents=True, exist_ok=True)


def bucket(g):
    b = int(hashlib.sha256(("MRCD-bl-13|" + str(g)).encode()).hexdigest(), 16) % 5
    return "dev" if b <= 2 else "tune" if b == 3 else "test"


def toks(t):
    return re.findall(r"[^\W\d_]+", t.lower(), flags=re.UNICODE)


def rule_rep(
    text,
):  # immediate exact repetition of 1-4 tokens (same rule as the pilot's run_text_baseline.py)
    t = toks(text)
    return int(
        any(
            t[i : i + n] == t[i + n : i + 2 * n]
            for n in range(1, 5)
            for i in range(len(t) - 2 * n + 1)
        )
    )


FILL_A = re.compile(
    r"^(e{2,}h*|eh+|ehm+|em+|mm+|hm+|ah+|aah+)$"
)  # a priori vocalic lexicon


def rule_fill_a(text):
    return int(any(FILL_A.match(w) for w in toks(text)))


def rule_fill_b(
    text,
):  # optimistic: informed by inspecting FILL tokens in this same corpus ('e' written for a filled pause)
    t = toks(text)
    return int(
        any(
            FILL_A.match(w)
            or (w == "e" and not (i + 1 < len(t) and re.match(r"h?i", t[i + 1])))
            for i, w in enumerate(t)
        )
    )


def evaluate(rows, ycol, preds, name, task):
    out = []
    for part in ["dev", "tune", "test", "all"]:
        sel = [r for r in rows if part == "all" or r["split"] == part]
        y = [r[ycol] for r in sel]
        g = [r["group"] for r in sel]
        for comp, fn in preds.items():
            p = [fn(r) for r in sel]
            m = confusion(y, p)
            m.update(
                {
                    "corpus": name,
                    "task": task,
                    "comparator": comp,
                    "partition": part,
                    "n_groups": len(set(g)),
                }
            )
            if m["support_pos"] > 0 and len(set(g)) > 1:
                m["ci95_by_group"] = cluster_bootstrap(y, p, g, 1000)
            m["stable"] = m["support_pos"] >= 30
            out.append(m)
    return out


def main():
    res = []
    errs = {}
    # ---------------- disfluentES (sentence; group = speaker)
    src = CORPORA / "10_disfluentES" / "disfluentES_v1.json"
    data = json.loads(src.read_bytes())
    rows = []
    excl = 0
    for rec in data:
        if not rec["disfluencies"]:
            continue
        for i, s in enumerate(rec["disfluencies"]):
            L = [x.strip() for x in s["lab"].split("|")]
            W = [x.strip() for x in s["man"].split("|")]
            if len(L) != len(W) or any(l.startswith("POS") for l in L):
                excl += 1
                continue
            text = " ".join(w for w in W if w not in ["-", "", "ε"])
            rows.append(
                {
                    "unit_id": f"{rec['speaker']}#{i}",
                    "group": rec["speaker"],
                    "grade": rec["nivel"],
                    "split": bucket(rec["speaker"]),
                    "text": text,
                    "rep": int(any(re.fullmatch(r"REP\d*", l) for l in L)),
                    "fill": int(any(re.fullmatch(r"FILL\d*", l) for l in L)),
                    "labels": "|".join(L),
                }
            )
    ones = lambda r: 1
    zeros = lambda r: 0
    res += evaluate(
        rows,
        "rep",
        {
            "always_present": ones,
            "always_absent": zeros,
            "rule_adjacent_repetition": lambda r: rule_rep(r["text"]),
        },
        "disfluentES",
        "T:repetition(sentence)",
    )
    res += evaluate(
        rows,
        "fill",
        {
            "always_present": ones,
            "always_absent": zeros,
            "rule_vocalic_lexicon_apriori": lambda r: rule_fill_a(r["text"]),
            "rule_lexicon_plus_bare_e_OPTIMISTIC": lambda r: rule_fill_b(r["text"]),
        },
        "disfluentES",
        "T:filler(sentence)",
    )
    with (D / "disfluentES_predictions.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "unit_id",
                "group",
                "grade",
                "split",
                "text",
                "rep",
                "fill",
                "labels",
                "pred_rep",
                "pred_fill_a",
                "pred_fill_b",
            ],
        )
        w.writeheader()
        for r in rows:
            w.writerow(
                {
                    **r,
                    "pred_rep": rule_rep(r["text"]),
                    "pred_fill_a": rule_fill_a(r["text"]),
                    "pred_fill_b": rule_fill_b(r["text"]),
                }
            )
    errs["disfluentES_rep"] = {
        "FP": [r["text"] for r in rows if r["rep"] == 0 and rule_rep(r["text"])][:12],
        "FN": [
            (r["text"], r["labels"])
            for r in rows
            if r["rep"] == 1 and not rule_rep(r["text"])
        ][:12],
    }
    errs["disfluentES_fill_b"] = {
        "FP": [r["text"] for r in rows if r["fill"] == 0 and rule_fill_b(r["text"])][
            :8
        ],
        "FN": [
            (r["text"], r["labels"])
            for r in rows
            if r["fill"] == 1 and not rule_fill_b(r["text"])
        ][:8],
    }
    # ---------------- ES-Port (utterance; group = dialogue)
    rows2 = []
    for p in sorted((CORPORA / "09_ES_Port" / "extracted" / "esport").glob("*.json")):
        d = json.loads(p.read_text("utf8"))
        g = d["file-id"]
        for t in d["turns"]:
            for spk, us in t["speakers"].items():
                for k, u in enumerate(us):
                    u = u["utterance"]
                    y = int(
                        bool(
                            re.search(
                                r"<(?!%)[^<>]*[^<>\-\s][^<>]*>",
                                re.sub(r"<[^<>]*->", " ", u),
                            )
                        )
                    )  # '< >' mark, excluding '<% >' and '<x->'
                    txt = re.sub(r"<%[^>]*>", " ", u)
                    txt = re.sub(
                        r"<([^<>]*)->", " ", txt
                    )  # remove fillers and unfinished-word marks
                    txt = re.sub(r"[<>=+]", " ", txt)
                    rows2.append(
                        {
                            "unit_id": f"{g}#{t['turn']}#{spk}#{k}",
                            "group": g,
                            "split": bucket(g),
                            "text": " ".join(txt.split()),
                            "raw": u,
                            "repvs": y,
                        }
                    )
    res += evaluate(
        rows2,
        "repvs",
        {
            "always_present": ones,
            "always_absent": zeros,
            "rule_adjacent_repetition": lambda r: rule_rep(r["text"]),
        },
        "ES-Port",
        "T:repetition U false-start (joint, utterance)",
    )
    errs["esport_repvs"] = {
        "FP": [(r["raw"]) for r in rows2 if r["repvs"] == 0 and rule_rep(r["text"])][
            :12
        ],
        "FN": [
            (r["raw"]) for r in rows2 if r["repvs"] == 1 and not rule_rep(r["text"])
        ][:12],
    }
    (D / "text_metrics.json").write_text(json.dumps(res, indent=1, ensure_ascii=True))
    (D / "text_error_examples.json").write_text(
        json.dumps(errs, indent=1, ensure_ascii=True)
    )
    write_config(
        D / "run",
        "text_baselines",
        __file__,
        {
            "n_disfluentES_sentences": len(rows),
            "disfluentES_excluded_POS_or_misaligned": excl,
            "n_esport_utterances": len(rows2),
            "partition": "sha256('MRCD-bl-13|'+group)%5: 0-2 dev, 3 tune, 4 test",
            "rules_fixed": True,
        },
        {"disfluentES_v1.json": sha256_file(src)},
    )
    for m in res:
        if m["partition"] in ("all", "test"):
            ci = m.get("ci95_by_group", {}).get("f1")
            print(
                f"{m['corpus']:12s} {m['task'][:28]:28s} {m['comparator'][:34]:34s} {m['partition']:4s} n={m['n']:6d} pos={m['support_pos']:5d} P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f} CI={ci}"
            )


if __name__ == "__main__":
    main()
