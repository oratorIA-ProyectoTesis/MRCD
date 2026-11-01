"""S6 step 5: paired speaker-grouped ablation + identity probe + verdict (frozen protocol sections 5-8)."""

import argparse
import json

import numpy as np
from s6_common import OUT, WORK

from mrcd.av.ablation import identity_probe, oof_predictions, summarize, verdict
from mrcd.av.config import ANALYSIS, config_hash

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", required=True)  # e.g. a2_face_only
args = ap.parse_args()
z = np.load(WORK / f"dataset_{args.dataset}.npz", allow_pickle=True)
A, T, V, Vraw, conf, y, g = z["A"], z["T"], z["V"], z["Vraw"], z["conf"], z["y"], z["g"]
cfg = ANALYSIS


def run(mask):
    P = oof_predictions(
        A[mask],
        T[mask],
        V[mask],
        conf[mask],
        y[mask],
        g[mask],
        cfg["cv_folds"],
        cfg["cv_repeats"],
        cfg["seed"],
        cfg["logreg_C"],
    )
    return summarize(
        P,
        y[mask],
        g[mask],
        cfg["bootstrap_B"],
        cfg["seed"],
        cfg["equivalence_margin_f1"],
    )


full = run(np.ones(len(y), bool))
nodom = run(~np.isin(g, cfg["dominant_speakers"]))
ver = verdict(full, nodom, cfg)
eg, eV, eR = z["elig_g"], z["elig_V"], z["elig_Vraw"]  # all eligible units (before the negative cap)
bn, br = identity_probe(eV, eg, seed=cfg["seed"]), identity_probe(eR, eg, seed=cfg["seed"])
perm = identity_probe(eV, np.random.default_rng(cfg["seed"]).permutation(eg), seed=cfg["seed"])
ident = {
    "normalised": {
        "balanced_acc": bn[0],
        "chance": bn[1],
        "speakers": bn[2],
        "rows": bn[3],
    },
    "raw": {"balanced_acc": br[0], "chance": br[1], "speakers": br[2], "rows": br[3]},
    "normalised_labels_permuted": {"balanced_acc": perm[0], "chance": perm[1]},
}
out = {
    "protocol_config_hash": config_hash(),
    "full": full,
    "without_dominant_speakers": nodom,
    "verdict": ver,
    "identity_probe": ident,
}
(OUT / f"ablation_results_{args.dataset}.json").write_text(json.dumps(out, indent=1))
print(
    json.dumps(
        {"verdict": ver, "paired_full": full["paired"], "ident": ident}, indent=1
    )
)
