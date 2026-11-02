"""Tiny synthetic datasets for the O-campaign tests (no real data, no ASR)."""

import numpy as np

from mrcd.context_eval.partition import group_folds
from mrcd.optimize.blocks import FeatureStore
from mrcd.optimize.search import Data


PROFS = ("p0", "p1", "p0_noloop", "p1_noloop")


def _words(rng, filler, n=6):
    w, t = [], 0.0
    for i in range(n):
        d = float(rng.uniform(0.15, 0.4))
        norm = "eh" if (filler and i == 2) else str(rng.choice(["hola", "quiero", "ir", "a", "madrid", "el", "tren"]))
        w.append({"text": norm, "norm": norm, "start": t, "end": t + d, "prob": float(rng.uniform(0.3, 1)), "punct_after": ""})
        t += d + float(rng.uniform(0.0, 0.2))
    return w


def make_data(cls="filler_word", n_spk=40, per=6, seed=0, signal=1.5):
    rng = np.random.default_rng(seed)
    recs, y, g, ids = [], [], [], []
    for s in range(n_spk):
        for k in range(per):
            yy = int(rng.random() < 0.3)
            probs = rng.dirichlet(np.ones(8), size=2)
            probs[:, 1] += signal * yy * 0.3
            probs = probs / probs.sum(1, keepdims=True)
            words = _words(rng, filler=bool(yy and cls == "filler_word" and rng.random() < 0.6))
            if cls == "repetition" and yy:
                words = words[:3] + [dict(words[2])] + words[3:]
            text = " ".join(x["text"] for x in words)
            recs.append({
                "probs": probs, "pooled": rng.normal(size=(2, 128)) + yy * 0.2,
                "pause": {"max_internal_pause_s": float(rng.random()), "edge_silence_s": 0.3, "speech_frac": 0.8, "n_frames": 200},
                "n_samples": 48000, "words": {k: words for k in PROFS}, "text": {k: text for k in PROFS},
                "acoustic": list(rng.normal(size=22))})
            y.append(yy), g.append(f"S{s:03d}"), ids.append(f"S{s:03d}_{k}")
    g = np.array(g)
    return Data(cls, ids, np.array(y), np.ones(len(y)), g, group_folds(g, 5, 13), FeatureStore(recs, cls))
