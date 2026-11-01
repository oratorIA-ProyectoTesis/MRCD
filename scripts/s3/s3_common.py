"""Shared loaders for the S3 experiments (DIHANA folds 1-4 only; fold 5 is never loaded)."""

import os

os.environ["OMP_NUM_THREADS"] = "4"  # before numpy (project rule)
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import csv  # noqa: E402
import json  # noqa: E402
import pickle  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

REPO = Path(os.environ.get("MRCD_REPO") or Path(__file__).resolve().parents[2])
sys.path.insert(0, str(REPO))
import numpy as np  # noqa: E402

from mrcd.asr_recovery import common as _rc  # noqa: E402

REC, TURNS, sha = _rc.OUT, _rc.TURNS, _rc.sha
load_pool, outer_folds, task_units = _rc.load_pool, _rc.outer_folds, _rc.task_units
from mrcd.constants import LABELS  # noqa: E402
from mrcd.decision import any_window_argmax, unit_score  # noqa: E402

S3 = REPO / "results" / "mejora_objetivos_20261101" / "S3"
FEAT = REPO / "results" / "mejoras_20261030" / "features"
HEAD = REPO / "results" / "mejoras_20261030" / "frozen_head_zero.json"
SEED = 13
CLASSES = ("filler_word", "prolongation", "repetition")


def sha_file(p) -> str:
    return sha(p)


def words_text(words) -> str:
    return " ".join(w["text"] for w in words)


def load_features():
    feats = {}
    for p in sorted(FEAT.glob("dihana_zero_shard*.pkl")):
        feats.update(pickle.loads(p.read_bytes()))
    return feats


def v0_vc(pool):
    """Per unit: V0 argmax decisions, V0 unit scores (max window prob) and Vc head scores (max over windows)."""
    from mrcd import adaptation as A

    feats = load_features()
    info = json.loads(HEAD.read_text("utf8"))
    ck, _ = A.load_adapted(
        info["checkpoint"], expected_sha256=info["checkpoint_sha256"]
    )
    classes = list(ck["classes"])
    out = {}
    for u in pool:
        rec = feats[u["unit_id"]]
        hs = A.head_scores(
            ck["state"], rec["pooled"], len(classes), ck["head_kind"], ck["hidden"]
        )
        d = {}
        for c in (*CLASSES, "revision"):
            d[f"v0_{c}"] = any_window_argmax(rec["probs"], list(LABELS), c)
            d[f"v0s_{c}"] = unit_score(rec["probs"], list(LABELS), c, "max")
        for c in CLASSES:
            if c in classes:
                sc = float(hs[:, classes.index(c)].max())
                d[f"vcs_{c}"] = sc
                d[f"vc_{c}"] = int(sc >= info["thresholds"][c]["threshold"])
        out[u["unit_id"]] = d
    return out, info


def turns_meta():
    return {r["turn_id"]: r for r in csv.DictReader(TURNS.open(encoding="utf-8-sig"))}


def frame_units(pool):
    """Frame of the P1 subset with label-defined rep_literal: (unit ids, y, w, speakers, pool index)."""
    fr = json.loads((REC / "subset.json").read_text("utf8"))["frame"]
    by = {u["unit_id"]: i for i, u in enumerate(pool)}
    ids, y, w, g, ix = [], [], [], [], []
    for s in fr:
        i = by[s["unit_id"]]
        v = pool[i]["ref"]["rep_literal"]
        if v is None:
            continue
        ids.append(s["unit_id"])
        y.append(int(v))
        w.append(float(s["w"]))
        g.append(pool[i]["speaker"])
        ix.append(i)
    return ids, np.array(y), np.array(w), np.array(g), np.array(ix)


def asr_words(variant, uid):
    return json.loads((REC / "asr" / variant / f"{uid}.json").read_text("utf8"))[
        "words"
    ]


def file_hashes(paths):
    return {str(Path(p).relative_to(REPO)): sha_file(p) for p in paths}


def holm_table(rows, key="p_boot"):
    from mrcd.context_eval.metrics import holm

    adj = holm([r[key] for r in rows])
    for r, a in zip(rows, adj):
        r["p_holm"] = a
        ci = r["ci"]
        r["verdict"] = (
            "supported after Holm"
            if (a < 0.05 and ci[0] * ci[1] > 0)
            else "not demonstrated"
        )
    return rows
