"""Alignment of reference occurrences to ASR words (normalised tokens, exact-equal blocks only)."""

from __future__ import annotations

import difflib
from collections.abc import Sequence


def map_equal(ref: Sequence[str], asr: Sequence[str]) -> dict[int, int]:
    """ref word index -> asr word index for every token inside an ``equal`` block."""
    sm = difflib.SequenceMatcher(None, list(ref), list(asr), autojunk=False)
    m: dict[int, int] = {}
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for k in range(i2 - i1):
                m[i1 + k] = j1 + k
    return m


def map_span(m: dict[int, int], s: int, e: int) -> tuple[int, int] | None:
    """ASR span for ref span [s, e) if every token is mapped and contiguous, else None (deleted/changed)."""
    if any(i not in m for i in range(s, e)):
        return None
    a = [m[i] for i in range(s, e)]
    if a != list(range(a[0], a[0] + len(a))):
        return None
    return a[0], a[-1] + 1
