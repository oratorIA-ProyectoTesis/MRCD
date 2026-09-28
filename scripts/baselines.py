"""Simple comparators on the same windows (B1 text, B2 acoustic+prosody, B3 concatenation).

Pooled per-window features and a class-balanced logistic regression. Without --test-data,
speaker-grouped cross-validation; scaling is fit inside each training fold only.

    python scripts/baselines.py --data data/windows_human_train.npz --test-data data/windows_human_dev.npz
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.constants import LABELS, TAXONOMY  # noqa: E402


def features(d: dict) -> dict[str, np.ndarray]:
    ac = d["ac"]
    acoustic = np.concatenate([ac.mean(1), ac.std(1), d["pros"]], 1)
    m = d["lmask"][..., None]
    text = (d["ling"] * m).sum(1) / np.maximum(m.sum(1), 1)
    return {"B1_text": text, "B2_acoustic_prosody": acoustic, "B3_concat": np.concatenate([acoustic, text], 1)}


def model():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--test-data")
    ap.add_argument("--allow-auto-labels", action="store_true")
    ap.add_argument("--out", default="results/baselines.json")
    a = ap.parse_args()
    d = dict(np.load(a.data, allow_pickle=False))
    t = dict(np.load(a.test_data, allow_pickle=False)) if a.test_data else None
    sources = {str(x["label_source"]) for x in (d, t) if x is not None}
    if "auto" in sources and not a.allow_auto_labels:
        raise SystemExit(
            "Etiquetas auto: comparar contra heurísticas es circular (usa --allow-auto-labels solo como humo)."
        )
    y_true = (t or d)["y"]
    report = {
        "train": a.data,
        "test": a.test_data or "GroupKFold(5) por hablante",
        "label_sources": sorted(sources),
        "support": {c: int((y_true == LABELS.index(c)).sum()) for c in LABELS},
        "systems": {},
    }
    feats, test_feats = features(d), features(t) if t else None
    for name, x in feats.items():
        if t is not None:
            pred = model().fit(x, d["y"]).predict(test_feats[name])
        else:
            pred = np.zeros_like(d["y"])
            folds = GroupKFold(min(5, len(set(d["group"])))).split(x, d["y"], d["group"])
            for tr, te in folds:
                pred[te] = model().fit(x[tr], d["y"][tr]).predict(x[te])
        per_class = f1_score(y_true, pred, labels=range(len(LABELS)), average=None, zero_division=0)
        present = [c for c in TAXONOMY if report["support"][c]]
        report["systems"][name] = {
            "f1": dict(zip(LABELS, map(float, per_class))),
            "macro_f1_present_classes": float(np.mean([per_class[LABELS.index(c)] for c in present]))
            if present
            else None,
            "macro_classes": present,
        }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps(report["systems"], indent=1))


if __name__ == "__main__":
    main()
