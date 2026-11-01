"""O2: timing / confidence signals already present in the cached ASR words (zero extra ASR cost). Protocol section 4."""

from __future__ import annotations

import numpy as np

SR = 16000
NAMES = (
    "log1p_n_words",
    "log_T",
    "chars_per_sec",
    "covered_frac",
    "max_z",
    "n_z_gt2",
    "max_dur_short",
    "max_gap",
    "n_gap_gt_025",
    "gap_frac",
    "min_prob",
    "mean_prob",
    "n_prob_lt_03",
    "min_prob_short",
    "n_overlap_pairs",
    "n_same_pairs",
    "n_same_close_or_overlap",
    "min_prob_same_pair",
    "n_z15_next_to_gap",
    "trailing_silence",
    "empty",
)


def _r(words) -> np.ndarray:
    d = np.array([max(w["end"] - w["start"], 0.02) for w in words])
    L = np.array([max(len(w["norm"]), 1) for w in words])
    return np.log(d / L)


class SignalStats:
    """Median / 1.4826*MAD of the per-word log(duration / length), learned on TRAIN turns only."""

    def __init__(self, med: float = 0.0, scale: float = 1.0):
        self.med, self.scale = med, scale

    @classmethod
    def fit(cls, records: list[dict]) -> "SignalStats":
        allr = [_r(r["words"]) for r in records if r["words"]]
        if not allr:
            return cls()
        v = np.concatenate(allr)
        med = float(np.median(v))
        mad = float(np.median(np.abs(v - med))) * 1.4826
        return cls(med, mad if mad > 1e-9 else 1.0)


def signal_features(rec: dict, st: SignalStats) -> np.ndarray:
    """rec = {"words": [{text,norm,start,end,prob}], "n_samples": int}. Returns a vector aligned with NAMES."""
    words = rec["words"]
    T = max(rec["n_samples"] / SR, 1e-3)
    if not words:
        v = np.zeros(len(NAMES))
        v[1] = np.log(T)
        v[-1] = 1.0
        return v
    z = (_r(words) - st.med) / st.scale
    d = np.array([max(w["end"] - w["start"], 0.0) for w in words])
    L = np.array([max(len(w["norm"]), 1) for w in words])
    prob = np.array([w["prob"] for w in words])
    short = L <= 3
    gaps = np.array(
        [words[i + 1]["start"] - words[i]["end"] for i in range(len(words) - 1)]
    )
    pos_gaps = np.clip(gaps, 0.0, None)
    overlap = gaps < -0.01
    same = np.array(
        [words[i]["norm"] == words[i + 1]["norm"] for i in range(len(words) - 1)], bool
    )
    close = same & ((gaps < 0.15) | overlap)
    z15_gap = 0
    for i in range(len(words)):
        nb = []
        if i > 0:
            nb.append(gaps[i - 1])
        if i < len(words) - 1:
            nb.append(gaps[i])
        if z[i] > 1.5 and any(g > 0.2 for g in nb):
            z15_gap += 1
    v = [
        np.log1p(len(words)),
        np.log(T),
        float(L.sum() / T),
        float(d.sum() / T),
        float(z.max()),
        float((z > 2).sum()),
        float(d[short].max()) if short.any() else 0.0,
        float(pos_gaps.max()) if len(gaps) else 0.0,
        float((pos_gaps > 0.25).sum()) if len(gaps) else 0.0,
        float(pos_gaps.sum() / T) if len(gaps) else 0.0,
        float(prob.min()),
        float(prob.mean()),
        float((prob < 0.3).sum()),
        float(prob[short].min()) if short.any() else 1.0,
        float(overlap.sum()) if len(gaps) else 0.0,
        float(same.sum()) if len(gaps) else 0.0,
        float(close.sum()) if len(gaps) else 0.0,
        float(min(min(prob[i], prob[i + 1]) for i in np.flatnonzero(same)))
        if same.any()
        else 1.0,
        float(z15_gap),
        float(max(T - words[-1]["end"], 0.0)),
        0.0,
    ]
    return np.array(v, float)


def signal_matrix(recs: list[dict], st: SignalStats) -> np.ndarray:
    return np.vstack([signal_features(r, st) for r in recs])
