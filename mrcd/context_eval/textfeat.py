"""Simple rules and context features for repetition / revision candidates (text only).

The same functions run on the marks-stripped reference transcript (condition A) and on the frozen ASR text
(condition B). No timing, no ASR confidence, no audio. Evidence BEFORE and AFTER each candidate is used.
"""

from __future__ import annotations

import re
import unicodedata

import numpy as np

from mrcd.constants import FUNCTION_WORDS

_TOK = re.compile(r"[^\W_]+|\.\.\.|…|[,;:.\-?!¿¡]")
_LETTERS = re.compile(r"[^\W\d_]+", re.UNICODE)


def _strip(s: str) -> str:
    return "".join(
        c
        for c in unicodedata.normalize("NFD", s.lower())
        if unicodedata.category(c) != "Mn"
    )


FUNC = {_strip(w) for w in FUNCTION_WORDS}
FILLER_RE = re.compile(r"^(e+h*|eh+|ehm+|em+|mm+|hm+|ah+|aa+h*|este+)$")
POLARITY = {
    "si",
    "no",
    "ya",
    "vale",
    "claro",
    "bueno",
    "ok",
    "bien",
    "gracias",
    "perfecto",
}
EDIT_EXPL = (("o", "sea"), ("es", "decir"))
EDIT_CORR = (
    ("digo",),
    ("perdon",),
    ("perdona",),
    ("quiero", "decir"),
    ("mejor", "dicho"),
)
EDIT_ALL = EDIT_EXPL + EDIT_CORR


def tokenize(text: str):
    """(words, punct_before, end_punct): normalised words and the punctuation string preceding each word."""
    words: list[str] = []
    pb: list[str] = []
    cur = ""
    for m in _TOK.findall(text or ""):
        if m[0].isalnum():
            words.append(_strip(m))
            pb.append(cur)
            cur = ""
        else:
            cur += m
    return words, pb, cur


# ----------------------------------------------------------------------------- simple rules
def rule_rep(text: str) -> int:
    """Immediate exact repetition of 1-4 tokens (same rule as scripts/baselines/text_baselines.rule_rep)."""
    t = _LETTERS.findall((text or "").lower())
    return int(
        any(
            t[i : i + n] == t[i + n : i + 2 * n]
            for n in range(1, 5)
            for i in range(len(t) - 2 * n + 1)
        )
    )


def _edit_positions(w: list[str]):
    out = []
    for i in range(len(w)):
        for term in EDIT_ALL:
            if tuple(w[i : i + len(term)]) == term:
                out.append((i, term))
    return out


def cut_fragment_positions(w: list[str]) -> list[int]:
    """Token a (len>=2) immediately followed by a longer token that starts with it."""
    return [
        i
        for i in range(len(w) - 1)
        if len(w[i]) >= 2 and w[i] != w[i + 1] and w[i + 1].startswith(w[i])
    ]


def rule_rev(text: str) -> int:
    """Agreed-in-advance trigger rule: edit-term lexicon OR cut fragment followed by a longer word."""
    w, _, _ = tokenize(text)
    return int(bool(_edit_positions(w)) or bool(cut_fragment_positions(w)))


# ----------------------------------------------------------------------------- candidates
def exact_candidates(w: list[str], max_n: int = 4, max_gap: int = 2):
    """(i, n, gap, j) with w[i:i+n] == w[j:j+n], j = i+n+gap."""
    out = []
    for n in range(1, max_n + 1):
        for gap in range(0, max_gap + 1):
            for i in range(0, len(w) - 2 * n - gap + 1):
                j = i + n + gap
                if w[i : i + n] == w[j : j + n]:
                    out.append((i, n, gap, j))
    return out


def fragment_candidates(w: list[str], max_gap: int = 1):
    out = []
    for gap in range(0, max_gap + 1):
        for i in range(0, len(w) - 1 - gap):
            a, b = w[i], w[i + 1 + gap]
            if a != b and len(b) > len(a) and b.startswith(a) and len(a) <= 4:
                out.append((i, gap, i + 1 + gap))
    return out


REP_NAMES = (
    "g0_n1",
    "g0_n2",
    "g0_n3",
    "g0_n4",
    "gap1",
    "gap2",
    "frag0",
    "frag1",
    "max_n",
    "run3",
    "log_cands",
    "func_only",
    "content",
    "polarity",
    "punct_between",
    "filler_between",
    "has_cand",
    "first_pos",
    "at_start",
    "prev_func",
    "prev_punct",
    "after_end",
    "after_func",
    "after_content",
    "log_len",
    "log_fillers",
    "log_commas",
    "log_ellipsis",
    "log_dash",
)
REV_NAMES = (
    "edit_expl",
    "edit_corr",
    "edit_any",
    "edit_prev_func",
    "edit_prev_punct",
    "edit_after_len",
    "cut_frag",
    "restart_anchor",
    "g0_any",
    "gap_any",
    "frag_any",
    "log_len",
    "log_fillers",
    "log_commas",
    "log_ellipsis",
    "log_dash",
    "log_cut",
)
NAMES = {"rep_literal": REP_NAMES, "rep_full": REP_NAMES, "revision": REV_NAMES}


def _punct_counts(pb: list[str], end: str):
    s = "".join(pb) + end
    return s.count(","), s.count("...") + s.count("…"), s.count("-")


def rep_features(text: str) -> dict[str, float]:
    w, pb, end = tokenize(text)
    L = len(w)
    ex = exact_candidates(w)
    fr = fragment_candidates(w)
    g0 = [c for c in ex if c[2] == 0]
    f: dict[str, float] = {k: 0.0 for k in REP_NAMES}
    for c in g0:
        f[f"g0_n{c[1]}"] = 1.0
    f["gap1"] = float(any(c[2] == 1 for c in ex))
    f["gap2"] = float(any(c[2] == 2 for c in ex))
    f["frag0"] = float(any(c[1] == 0 for c in fr))
    f["frag1"] = float(any(c[1] == 1 for c in fr))
    f["max_n"] = max([c[1] for c in ex], default=0) / 4.0
    f["run3"] = float(any(w[i] == w[i + 1] == w[i + 2] for i in range(L - 2)))
    f["log_cands"] = float(np.log1p(len(ex) + len(fr)))
    f["func_only"] = float(
        any(all(t in FUNC for t in w[c[0] : c[0] + c[1]]) for c in g0)
    )
    f["content"] = float(
        any(any(t not in FUNC for t in w[c[0] : c[0] + c[1]]) for c in g0)
    )
    f["polarity"] = float(any(c[1] == 1 and w[c[0]] in POLARITY for c in g0))
    pbt = False
    fil = False
    for i, n, gap, j in ex:
        if pb[j] or any(pb[k] for k in range(i + n, j)):
            pbt = True
        if gap and any(FILLER_RE.match(t) for t in w[i + n : j]):
            fil = True
    f["punct_between"], f["filler_between"] = float(pbt), float(fil)
    cands = [(c[0], c[3] + c[1]) for c in ex] + [(c[0], c[2] + 1) for c in fr]
    if cands:
        i, e = min(cands)
        f["has_cand"] = 1.0
        f["first_pos"] = i / max(L, 1)
        f["at_start"] = float(i == 0)
        f["prev_func"] = float(i > 0 and w[i - 1] in FUNC)
        f["prev_punct"] = float(bool(pb[i]) and i > 0)
        f["after_end"] = float(e >= L)
        f["after_func"] = float(e < L and w[e] in FUNC)
        f["after_content"] = float(e < L and w[e] not in FUNC)
    f["log_len"] = float(np.log1p(L))
    f["log_fillers"] = float(np.log1p(sum(bool(FILLER_RE.match(t)) for t in w)))
    c, el, d = _punct_counts(pb, end)
    f["log_commas"], f["log_ellipsis"], f["log_dash"] = (
        float(np.log1p(c)),
        float(np.log1p(el)),
        float(np.log1p(d)),
    )
    return f


def rev_features(text: str) -> dict[str, float]:
    w, pb, end = tokenize(text)
    L = len(w)
    f: dict[str, float] = {k: 0.0 for k in REV_NAMES}
    eds = _edit_positions(w)
    if eds:
        f["edit_expl"] = float(any(t in EDIT_EXPL for _, t in eds))
        f["edit_corr"] = float(any(t in EDIT_CORR for _, t in eds))
        f["edit_any"] = 1.0
        i, t = eds[0]
        f["edit_prev_func"] = float(i > 0 and w[i - 1] in FUNC)
        f["edit_prev_punct"] = float(bool(pb[i]) and i > 0)
        f["edit_after_len"] = float(np.log1p(max(0, L - (i + len(t)))))
    cuts = cut_fragment_positions(w)
    f["cut_frag"] = float(bool(cuts))
    f["log_cut"] = float(np.log1p(len(cuts)))
    anchors = 0
    for i in range(L - 1):
        for j in range(i + 2, min(L - 1, i + 6) + 1):
            if j + 1 < L and w[i] == w[j] and w[i + 1] != w[j + 1]:
                anchors += 1
    f["restart_anchor"] = float(np.log1p(anchors))
    ex = exact_candidates(w)
    f["g0_any"] = float(any(c[2] == 0 for c in ex))
    f["gap_any"] = float(any(c[2] > 0 for c in ex))
    f["frag_any"] = float(bool(fragment_candidates(w)))
    f["log_len"] = float(np.log1p(L))
    f["log_fillers"] = float(np.log1p(sum(bool(FILLER_RE.match(t)) for t in w)))
    c, el, d = _punct_counts(pb, end)
    f["log_commas"], f["log_ellipsis"], f["log_dash"] = (
        float(np.log1p(c)),
        float(np.log1p(el)),
        float(np.log1p(d)),
    )
    return f


def featurize(texts, task: str) -> np.ndarray:
    names = NAMES[task]
    fn = rev_features if task == "revision" else rep_features
    return np.array([[fn(t)[k] for k in names] for t in texts], float).reshape(
        len(texts), len(names)
    )
