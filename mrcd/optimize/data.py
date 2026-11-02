"""Real-data loaders (run stage only). DIHANA folds 1-4 are the experimental pool; fold 5 is loaded ONLY by
``load_fold5_historical`` (single final report). Labels are the existing ones; nothing is annotated or created."""

from __future__ import annotations

import csv
import json
import pickle
import sys

import numpy as np

from mrcd.asr_recovery import common as RC
from mrcd.optimize.blocks import FeatureStore
from mrcd.optimize.search import Data

REPO = RC.REPO
FEAT = REPO / "results" / "mejoras_20261030" / "features"
ASR_DIRS = {
    "p0": RC.ASR,
    "p1": RC.OUT / "asr" / "P1",
    "p0_noloop": REPO / "results/mejora_objetivos_20261101/S2/asr/cpu/B",
    "p1_noloop": REPO / "results/mejora_objetivos_20261101/S2/asr/cpu/D_B",
}
ACOUSTIC = RC.OUT / "features" / "acoustic.pkl"
LABEL_COL = {"filler_word": "filler_present", "prolongation": "lengthening_present"}


def _words(profile: str, uid: str):
    p = ASR_DIRS[profile] / f"{uid}.json"
    d = json.loads(p.read_text("utf8"))
    return d["words"], d.get("n_samples")


def _rec(uid, feats, profiles, acoustic):
    f = feats[uid]
    words, text, n_samples = {}, {}, None
    for pr in profiles:
        w, n = _words(pr, uid)
        words[pr], text[pr] = w, " ".join(x["text"] for x in w)
        n_samples = n_samples or n
    ac = None
    if acoustic is not None and uid in acoustic:
        a, v = acoustic[uid]
        ac = list(a) + list(v)
    return {
        "probs": f["probs"],
        "pooled": f["pooled"],
        "pause": f["pause"],
        "n_samples": n_samples or int(f["dur_s"] * 16000),
        "words": words,
        "text": text,
        "acoustic": ac,
    }


def load_features() -> dict:
    feats = {}
    for p in sorted(FEAT.glob("dihana_zero_shard*.pkl")):
        feats.update(pickle.loads(p.read_bytes()))
    return feats


def load_dihana(cls: str) -> Data:
    pool = RC.load_pool()  # asserts folds 1-4 only and the audited counts
    outer = RC.outer_folds(pool)
    feats = load_features()
    acoustic = pickle.loads(ACOUSTIC.read_bytes())["feats"]
    turns = {
        r["turn_id"]: r for r in csv.DictReader(RC.TURNS.open(encoding="utf-8-sig"))
    }
    profiles = ("p0",)
    keep, y, w = [], [], []
    if cls == "repetition":
        profiles = ("p0", "p1", "p0_noloop", "p1_noloop")
        fr = json.loads((RC.OUT / "subset.json").read_text("utf8"))["frame"]
        by = {u["unit_id"]: i for i, u in enumerate(pool)}
        for s in fr:
            i = by[s["unit_id"]]
            v = pool[i]["ref"]["rep_literal"]
            if v is not None:  # indeterminate turns are excluded, never negatives
                keep.append(i), y.append(int(v)), w.append(float(s["w"]))
    else:
        for i, u in enumerate(pool):
            (
                keep.append(i),
                y.append(int(turns[u["unit_id"]][LABEL_COL[cls]])),
                w.append(1.0),
            )
    recs = [_rec(pool[i]["unit_id"], feats, profiles, acoustic) for i in keep]
    return Data(
        cls,
        [pool[i]["unit_id"] for i in keep],
        np.array(y),
        np.array(w),
        np.array([pool[i]["speaker"] for i in keep]),
        outer[np.array(keep)],
        FeatureStore(recs, cls),
    )


def _evalkit():
    sys.modules.pop("common", None)  # scripts/o_campaign/common.py would shadow scripts/improve/common.py
    sys.path.insert(0, str(REPO / "scripts" / "improve"))
    sys.path.insert(0, str(REPO / "scripts" / "baselines"))
    import evalkit

    return evalkit


def load_external(corpus: str, cls: str):
    """EPIC (filler_word) / AMERESCO (prolongation): already inspected in earlier phases (declared). Returns (recs, y, groups)."""
    ek = _evalkit()
    rows = ek.unit_table(corpus, "zero")
    from mrcd.asr_recovery.common import REPO as _  # noqa: F401

    recs, y, g = [], [], []
    for r in rows:
        words = ek.load_asr(corpus, r["unit_id"])["words"]
        recs.append(
            {
                "probs": r["rec"]["probs"],
                "pooled": r["rec"]["pooled"],
                "pause": r["rec"]["pause"],
                "n_samples": int(r["rec"]["dur_s"] * 16000),
                "words": {"p0": words},
                "text": {"p0": r["asr_text"]},
                "acoustic": None,
            }
        )
        y.append(int(r["ref"][cls]))
        g.append(r["group"])
    return recs, np.array(y), np.array(g)


def load_fold5_historical(cls: str):
    """HISTORICAL, CONTAMINATED: used only by the single final report. filler_word / prolongation only."""
    ek = _evalkit()
    rows = ek.unit_table("dihana", "zero", {"test"})
    recs, y, g = [], [], []
    for r in rows:
        words = ek.load_asr("dihana", r["unit_id"])["words"]
        recs.append(
            {
                "probs": r["rec"]["probs"],
                "pooled": r["rec"]["pooled"],
                "pause": r["rec"]["pause"],
                "n_samples": int(r["rec"]["dur_s"] * 16000),
                "words": {"p0": words},
                "text": {"p0": r["asr_text"]},
                "acoustic": None,
            }
        )
        y.append(int(r["ref"][cls]))
        g.append(r["group"])
    return recs, np.array(y), np.array(g)
