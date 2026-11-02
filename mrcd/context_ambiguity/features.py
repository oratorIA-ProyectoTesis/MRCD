"""Text and acoustic features of one occurrence. Words are accent-free lower-case, no punctuation."""

from __future__ import annotations

import collections
from collections.abc import Iterable, Sequence

UNK, BOS, EOS = "<unk>", "<s>", "</s>"
ACOUSTIC_KEYS = (
    "a_dur",
    "a_dur_pc",
    "a_gap_b",
    "a_gap_a",
    "a_prob",
    "a_rms",
    "a_f0m",
    "a_f0s",
    "a_f0_ok",
)


def build_vocab(turn_words: Iterable[Sequence[str]], min_count: int = 3) -> frozenset[str]:
    c: collections.Counter = collections.Counter()
    for w in turn_words:
        c.update(w)
    return frozenset(x for x, n in c.items() if n >= min_count)


def _tok(w: Sequence[str], i: int, vocab: frozenset[str]) -> str:
    if i < 0:
        return BOS
    if i >= len(w):
        return EOS
    return w[i] if w[i] in vocab else UNK


def text_features(
    words: Sequence[str],
    s: int,
    e: int,
    form: str,
    vocab: frozenset[str],
    punct: tuple[str, str] | None = None,
) -> dict[str, float]:
    """Context features of the span ``words[s:e]``. ``punct`` = (before, after) marks, sensitivity only."""
    n = len(words)
    f: dict[str, float] = {f"form={form}": 1.0}
    for k, name in ((-2, "L2"), (-1, "L1"), (e - s, "R1"), (e - s + 1, "R2")):
        t = _tok(words, s + k, vocab)
        f[f"{name}={t}"] = 1.0
        if name in ("L1", "R1"):
            f[f"{form}|{name}={t}"] = 1.0
    f["first"] = float(s == 0)
    f["last"] = float(e == n)
    f["standalone"] = float(n == e - s)
    f["len<=3"] = float(n <= 3)
    f["len<=6"] = float(n <= 6)
    f["len<=10"] = float(n <= 10)
    f[f"{form}|first"] = float(s == 0)
    f[f"{form}|last"] = float(e == n)
    f[f"{form}|standalone"] = float(n == e - s)
    if punct is not None:
        f[f"pb={punct[0]}"] = 1.0
        f[f"pa={punct[1]}"] = 1.0
    return f
