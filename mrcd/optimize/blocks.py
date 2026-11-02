"""Feature blocks. A record is a dict with the fields listed in ``data.py``; every block is a pure function of the
rows it is given, except ``sig`` / ``lenx`` which depend on a duration normaliser fitted on TRAIN rows only."""

from __future__ import annotations

import re
import zlib

import numpy as np

from mrcd.asr_recovery import signals as SG
from mrcd.context_eval.textfeat import featurize, rule_rep

FILL_A = re.compile(
    r"^(e{2,}h*|eh+|ehm+|em+|mm+|hm+|ah+|aah+)$"
)  # parity with scripts/baselines/text_baselines
_TOK = re.compile(r"[^\W\d_]+", re.UNICODE)
FUNC = frozenset(
    "el la los las un una de del y o que en a con por para se me te le lo es no si pues".split()
)
DISC = frozenset("este bueno pues vale entonces o sea".split())
K_CTX = 2  # +-k words of context around a filler candidate


def toks(t: str) -> list[str]:
    return _TOK.findall((t or "").lower())


def rule_filler(text: str) -> int:
    return int(any(FILL_A.match(w) for w in toks(text)))


def _san(a) -> np.ndarray:
    return np.clip(
        np.nan_to_num(np.asarray(a, float), nan=0.0, posinf=1e3, neginf=-1e3), -1e3, 1e3
    )


# ------------------------------------------------------------------ stateless blocks
def b_rule(recs, profile, cls):
    f = rule_filler if cls == "filler_word" else rule_rep
    return np.array([[float(f(r["text"][profile]))] for r in recs])


def b_v0(recs, profile, cls):
    out = []
    for r in recs:
        p = np.asarray(r["probs"], float)
        out.append(np.concatenate([p.max(0), p.mean(0), [np.log1p(len(p))]]))
    return np.array(out)


def b_emb(recs, profile, cls):
    return np.array(
        [
            np.concatenate(
                [
                    np.asarray(r["pooled"], float).max(0),
                    np.asarray(r["pooled"], float).mean(0),
                ]
            )
            for r in recs
        ]
    )


def b_pause(recs, profile, cls):
    return np.array(
        [
            [
                r["pause"]["max_internal_pause_s"],
                r["pause"]["edge_silence_s"],
                r["pause"]["speech_frac"],
                np.log1p(r["pause"]["n_frames"]),
            ]
            for r in recs
        ],
        float,
    )


def ctxfill_row(words: list[dict]) -> list[float]:
    n = len(words)
    cand = [i for i, w in enumerate(words) if FILL_A.match(w["norm"] or "")]
    disc = [i for i, w in enumerate(words) if (w["norm"] or "") in DISC]
    if not cand:
        base = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, float(len(disc)), 0.0]
        return base
    dur = [words[i]["end"] - words[i]["start"] for i in cand]
    gb = [words[i]["start"] - words[i - 1]["end"] if i > 0 else 1.0 for i in cand]
    ga = [words[i + 1]["start"] - words[i]["end"] if i < n - 1 else 1.0 for i in cand]
    prv = [(words[i - 1]["norm"] in FUNC) if i > 0 else False for i in cand]
    nxt = [(words[i + 1]["norm"] in FUNC) if i < n - 1 else False for i in cand]
    pa = [words[i].get("punct_after", "") != "" for i in cand]
    pr = [
        min(
            words[j]["prob"]
            for j in range(max(0, i - K_CTX), min(n, i + K_CTX + 1))
            if j != i
        )
        if n > 1
        else 1.0
        for i in cand
    ]
    return [
        float(len(cand)),
        float(np.log1p(len(cand))),
        float(max(dur)),
        float(max(gb)),
        float(max(ga)),
        float(np.mean(prv)),
        float(np.mean(nxt)),
        float(np.mean(pa)),
        float(np.mean([words[i]["prob"] for i in cand])),
        float(min(pr)),
        float(len(disc)),
        float(cand[0] == 0),
    ]


def b_ctxfill(recs, profile, cls):
    return np.array([ctxfill_row(r["words"][profile]) for r in recs], float)


def b_reptxt(recs, profile, cls):
    return featurize([r["text"][profile] for r in recs], "rep_literal")


def loop_row(text: str) -> list[float]:
    t = toks(text)
    n = len(t)
    if n == 0:
        return [0.0] * 6
    run = best = 1
    for i in range(1, n):
        run = run + 1 if t[i] == t[i - 1] else 1
        best = max(best, run)
    comp = len(zlib.compress(" ".join(t).encode())) / max(len(" ".join(t).encode()), 1)
    ng = max(
        (
            max(
                sum(1 for i in range(n - k + 1) if t[i : i + k] == t[j : j + k])
                for j in range(n - k + 1)
            )
            if n >= k
            else 0
        )
        for k in (2, 3)
    )
    return [
        float(best),
        float(np.log1p(n)),
        float(comp),
        float(ng),
        float(len(set(t)) / n),
        float(best >= 4),
    ]


def b_loop(recs, profile, cls):
    return np.array([loop_row(r["text"][profile]) for r in recs], float)


def b_acoustic(recs, profile, cls):
    if any(r.get("acoustic") is None for r in recs):
        raise KeyError("acoustic block unavailable for these rows")
    return np.array([r["acoustic"] for r in recs], float)


STATELESS = {
    "rule": b_rule,
    "v0": b_v0,
    "emb": b_emb,
    "pause": b_pause,
    "ctxfill": b_ctxfill,
    "reptxt": b_reptxt,
    "loop": b_loop,
    "acoustic": b_acoustic,
}
STATEFUL = ("sig", "lenx")
BLOCKS = tuple(STATELESS) + STATEFUL


def lenx_row(rec: dict, st: SG.SignalStats, profile: str) -> list[float]:
    words = rec["words"][profile]
    if not words:
        return [0.0] * 8
    z = (SG._r(words) - st.med) / st.scale
    zs = np.sort(z)[::-1]
    top = [float(zs[i]) if i < len(zs) else 0.0 for i in range(3)]
    return top + [
        float((z > 1.0).sum()),
        float((z > 1.5).sum()),
        float(z.mean()),
        float(z.std()),
        float(z[-1]),
    ]


def fit_state(recs, profile) -> SG.SignalStats:
    return SG.SignalStats.fit(
        [{"words": r["words"][profile], "n_samples": r["n_samples"]} for r in recs]
    )


def stateful_block(name, recs, profile, st):
    if name == "sig":
        return np.vstack(
            [
                SG.signal_features(
                    {"words": r["words"][profile], "n_samples": r["n_samples"]}, st
                )
                for r in recs
            ]
        )
    if name == "lenx":
        return np.array([lenx_row(r, st, profile) for r in recs], float)
    raise KeyError(name)


class FeatureStore:
    """Computes each stateless block once for the whole dataset; stateful blocks are fitted on the given train rows."""

    def __init__(self, recs: list[dict], cls: str):
        self.recs, self.cls = recs, cls
        self._cache: dict = {}
        self._st: dict = {}

    def stateless(self, name: str, profile: str) -> np.ndarray:
        k = (name, profile)
        if k not in self._cache:
            self._cache[k] = _san(STATELESS[name](self.recs, profile, self.cls))
        return self._cache[k]

    def stats(self, profile: str, train_idx: np.ndarray) -> SG.SignalStats:
        k = (profile, hash(np.asarray(train_idx).tobytes()))
        if k not in self._st:
            self._st[k] = fit_state([self.recs[i] for i in train_idx], profile)
        return self._st[k]

    def matrix(self, blocks, profile: str, train_idx: np.ndarray) -> np.ndarray:
        """Full-dataset matrix; stateful normalisers come from ``train_idx`` only."""
        parts = []
        for b in blocks:
            if b in STATELESS:
                parts.append(self.stateless(b, profile))
            else:
                k = ("S", b, profile, hash(np.asarray(train_idx).tobytes()))
                if k not in self._cache:
                    self._cache[k] = _san(
                        stateful_block(
                            b, self.recs, profile, self.stats(profile, train_idx)
                        )
                    )
                parts.append(self._cache[k])
        return np.hstack(parts)
