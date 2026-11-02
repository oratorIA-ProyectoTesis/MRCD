"""Build the occurrence table (reference + frozen ASR alignment + acoustics + V0 projection)."""

from __future__ import annotations

import argparse
import csv
import glob
import json
import pickle
import time
from pathlib import Path

import numpy as np

from mrcd.context_ambiguity import acoustic as ac
from mrcd.context_ambiguity.align import map_equal, map_span
from mrcd.context_ambiguity.pool import load_turns

REPO = Path(__file__).resolve().parents[2]
DIHANA = Path(
    r"<DATASET_ROOT>"
)
TURNS = REPO / "results/baselines_20261030/dihana/turns.csv"
ASR0 = REPO / "results/mejoras_20261030/asr_cache/dihana"
ASR1 = REPO / "results/asr_recovery_20261031/asr/P1"
FEAT = REPO / "results/mejoras_20261030/features"
OUT = REPO / "results/fundamentacion_20261101/f1_context"
FILLER_IDX = 1  # LABELS = ("fluent", "filler_word", ...)


def punct_ctx(toks, s, e):
    def edge(t, lead):
        txt = t.text
        i = 0
        if lead:
            while i < len(txt) and not txt[i].isalnum():
                i += 1
            return txt[:i]
        j = len(txt)
        while j > 0 and not txt[j - 1].isalnum():
            j -= 1
        return txt[j:]

    before = edge(toks[s], True)
    if not before and s > 0 and not toks[s - 1].norm:
        before = toks[s - 1].text
    after = edge(toks[e - 1], False)
    if not after and e < len(toks) and not toks[e].norm:
        after = toks[e].text
    return before[:3], after[:3]


def load_asr(path: Path):
    if not path.exists():
        return None
    d = json.loads(path.read_text("utf8"))
    return [w for w in d["words"] if w["norm"]]


def v0_flag(rec, a, b):
    st = np.asarray(rec["starts"], float)
    pr = np.asarray(rec["probs"], float)
    cover = (st < b) & (st + 3.0 > a)
    if not cover.any():
        return 0, 0.0
    am = pr[cover].argmax(1) == FILLER_IDX
    return int(am.any()), float(pr[cover][:, FILLER_IDX].max())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args()
    t0 = time.time()
    turns = load_turns(DIHANA, TURNS)
    ul = {r["turn_id"]: r["ul_rel"] for r in csv.DictReader(TURNS.open(encoding="utf-8-sig"))}
    v0 = {}
    for f in sorted(glob.glob(str(FEAT / "dihana_zero_shard*.pkl"))):
        v0.update(pickle.load(open(f, "rb")))
    rows, tdata = [], {}
    for k, u in enumerate(turns):
        toks = u["tokens"]
        widx, words = [], []
        for t in toks:
            widx.append(len(words))
            if t.norm:
                words.append(t.norm)
        ent = {"words": words, "occ": [], "asr0": None, "asr1": None}
        if not u["occs"]:
            continue
        for tag, root in (("asr0", ASR0), ("asr1", ASR1)):
            w = load_asr(root / f"{u['turn_id']}.json")
            if w is not None:
                ent[tag] = {
                    "norm": [x["norm"] for x in w],
                    "start": [x["start"] for x in w],
                    "end": [x["end"] for x in w],
                    "prob": [x["prob"] for x in w],
                    "text": [x["text"] for x in w],
                }
        tr = None
        for o in u["occs"]:
            s = widx[o.start]
            e = s + (o.end - o.start)
            row = {
                "turn_id": u["turn_id"],
                "speaker": u["speaker"],
                "fold": u["fold"],
                "form": o.form,
                "y": int(o.d_type == "rellenar"),
                "d_type": o.d_type or "none",
                "s": s,
                "e": e,
                "punct": punct_ctx(toks, o.start, o.end),
            }
            for tag in ("asr0", "asr1"):
                A = ent[tag]
                row[tag] = None
                if A is None:
                    continue
                m = map_equal(words, A["norm"])
                sp = map_span(m, s, e)
                if sp is None:
                    continue
                s2, e2 = sp
                if tag == "asr0":
                    if tr is None:
                        tr = ac.turn_tracks(ac.load_ulaw(DIHANA / ul[u["turn_id"]]))
                    nchar = sum(len(x) for x in A["norm"][s2:e2])
                    feats = ac.word_acoustics(
                        tr,
                        A["start"][s2],
                        A["end"][e2 - 1],
                        nchar,
                        float(np.mean(A["prob"][s2:e2])),
                        A["end"][s2 - 1] if s2 > 0 else None,
                        A["start"][e2] if e2 < len(A["norm"]) else None,
                    )
                    flag, sc = v0_flag(v0[u["turn_id"]], A["start"][s2], A["end"][e2 - 1])
                    row[tag] = {"s": s2, "e": e2, "ac": feats, "v0": flag, "v0_score": sc}
                else:
                    row[tag] = {"s": s2, "e": e2}
            rows.append(row)
        tdata[u["turn_id"]] = ent
        if k % 500 == 0:
            print(k, len(rows), round(time.time() - t0), flush=True)
    a.out.mkdir(parents=True, exist_ok=True)
    pickle.dump({"rows": rows, "turns": tdata}, open(a.out / "_table.pkl", "wb"))
    print("rows", len(rows), "sec", round(time.time() - t0))


if __name__ == "__main__":
    main()
