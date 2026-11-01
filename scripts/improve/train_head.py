"""Ec: supervised adaptation of a final head with EXISTING human DIHANA labels (f, a, r).
Train = dev units (folds 1-3), selection of (head, wd, epoch) by mean AP on fold 4, per-class threshold = argmax F1 on fold 4.
Fold 5, EPIC, AMERESCO are never loaded here. Writes a NEW checkpoint (refuses to overwrite) + frozen_head_<tag>.json + config.json.
Usage: python train_head.py PAD_MODE TAG"""

import json
import sys
import time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from common import OUT, CKPT, REPO, write_cfg, CLASSES3, references  # noqa
from evalkit import load_features  # noqa
from mrcd import adaptation as A, decision as D  # noqa

HEADS, WDS, LR, EPOCHS, SNAP = ["linear", "mlp"], [1e-4, 1e-3, 1e-2], 1e-2, 200, 10
HIDDEN = 32


def build(feats, ids, refs):
    X, ui, Y = [], [], []
    for k, u in enumerate(ids):
        p = feats[u]["pooled"]
        X.append(p)
        ui += [k] * len(p)
        Y.append([refs[u][c] for c in CLASSES3])
    return np.concatenate(X), np.array(ui), np.array(Y, np.float32)


def unit_scores(state, kind, X, ui, U):
    s = A.head_scores(state, X, len(CLASSES3), kind, HIDDEN)
    out = np.full((U, s.shape[1]), -1.0)
    for k in range(U):
        out[k] = s[ui == k].max(0)
    return out


def main():
    mode, tag = sys.argv[1], sys.argv[2]
    fr = json.loads((OUT / "dev_units_frozen.json").read_text())
    refs = references("dihana")
    dev, tune = fr["dev_units"], fr["tune_units"]
    spk = {
        f: {refs[u]["group"] for u in refs if int(refs[u]["fold"]) == f}
        for f in (1, 2, 3, 4, 5)
    }
    for a in spk:
        for b in spk:
            if a < b:
                assert not (spk[a] & spk[b]), (
                    f"speaker leakage between folds {a} and {b}"
                )
    feats = load_features("dihana", mode)
    Xd, ud, Yd = build(feats, dev, refs)
    Xt, ut, Yt = build(feats, tune, refs)
    print(
        "train windows",
        len(Xd),
        "units",
        len(dev),
        "pos",
        Yd.sum(0),
        "| tune units",
        len(tune),
        "pos",
        Yt.sum(0),
        flush=True,
    )
    grid, best = [], None
    for kind in HEADS:
        for wd in WDS:
            t0 = time.time()
            snaps = A.fit_head(
                Xd,
                ud,
                Yd,
                kind=kind,
                wd=wd,
                lr=LR,
                epochs=EPOCHS,
                snapshot_every=SNAP,
                hidden=HIDDEN,
                seed=13,
            )
            for ep, st in snaps.items():
                st_t = unit_scores(st, kind, Xt, ut, len(tune))
                st_d = unit_scores(st, kind, Xd, ud, len(dev))
                ap_t = [A.average_precision(Yt[:, j], st_t[:, j]) for j in range(3)]
                ap_d = [A.average_precision(Yd[:, j], st_d[:, j]) for j in range(3)]
                rec = {
                    "kind": kind,
                    "wd": wd,
                    "epoch": ep,
                    "tune_ap": ap_t,
                    "tune_map": float(np.mean(ap_t)),
                    "train_ap": ap_d,
                }
                grid.append(rec)
                if best is None or rec["tune_map"] > best[0]["tune_map"]:
                    best = (rec, st)
            print(
                kind,
                wd,
                f"{time.time() - t0:.0f}s best so far {best[0]['tune_map']:.3f}",
                flush=True,
            )
    cfg, state = best
    st_t = unit_scores(state, cfg["kind"], Xt, ut, len(tune))
    thr = {}
    for j, c in enumerate(CLASSES3):
        t, f = D.best_threshold(st_t[:, j], Yt[:, j])
        thr[c] = {"threshold": t, "tune_f1_at_threshold": f}
    ck_path = OUT / "checkpoints" / f"ckpt_head_dihana_f123_{tag}.pt"
    meta = {
        "tag": tag,
        "pad_mode_features": mode,
        "selected": cfg,
        "thresholds": thr,
        "classes": CLASSES3,
        "train_folds": [1, 2, 3],
        "selection_fold": 4,
        "n_train_units": len(dev),
        "dev_units_sha256": fr["sha256_of_json_dumps"],
        "label_source": "human DIHANA turn-level presence (f, a, r:repeticion)",
        "seed": 13,
        "hidden": HIDDEN,
        "lr": LR,
        "epochs_max": EPOCHS,
        "pooling": "max over windows",
        "scores_are_calibrated_probabilities": False,
    }
    h = A.save_adapted(
        ck_path,
        base_ckpt=CKPT,
        classes=CLASSES3,
        state=state,
        kind=cfg["kind"],
        hidden=HIDDEN,
        in_dim=Xd.shape[1],
        meta=meta,
    )
    (OUT / f"frozen_head_{tag}.json").write_text(
        json.dumps(
            {**meta, "checkpoint": str(ck_path), "checkpoint_sha256": h, "grid": grid},
            indent=1,
            default=float,
        )
    )
    write_cfg(
        OUT / "head_" / tag if False else OUT / f"head_{tag}",
        f"train_head_{tag}",
        __file__,
        {
            "selected": cfg,
            "checkpoint": str(ck_path),
            "checkpoint_sha256": h,
            "thresholds": thr,
            "dev_units_sha256": fr["sha256_of_json_dumps"],
            "grid_size": len(grid),
        },
    )
    print("SELECTED", cfg)
    print("THRESHOLDS", thr)
    print("ckpt", ck_path, h)


if __name__ == "__main__":
    main()
