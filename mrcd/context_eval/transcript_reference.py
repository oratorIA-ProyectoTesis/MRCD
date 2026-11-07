"""Evaluation reference derived ONLY from an orthographic transcription without disfluency marks (CHM150).

Used to build the reference (y) and descriptive counts of an external test, never as a model input.

* ``rep_literal`` - the DIHANA A-literal criterion re-applied to plain text: an immediate exact repetition of
  1-4 contiguous tokens (``m == c``), the same n-gram range and adjacency as ``textfeat.rule_rep`` and the
  ``exact`` branch of ``survival.adj_repeat``. Same exclusion as the DIHANA text view: non-lexical filled pauses
  are not words there (``<f>`` is an empty element), so they are removed before the check. DIHANA's
  indeterminate sub-types (B cut-word restart, C corrected mispronunciation, D other) need marks that a plain
  orthographic transcription does not have; they cannot be identified and no utterance is excluded for them.
* Marks are emitted in the shape of ``reference.repetition_marks`` (``span`` / ``mt`` / ``ct`` / ``subtype``) so
  that ``survival.turn_survives`` is applied unchanged.
* Filled pauses: counted by form, descriptive only. ``este`` is NOT treated as a filled pause (lexically
  ambiguous demonstrative / discourse marker, and not annotated). ``e`` before a word starting with ``i`` / ``hi``
  is the conjunction and is not counted. Prolongation is not marked at all and is not evaluable.
"""

from __future__ import annotations

import re
from collections import Counter

from mrcd.context_eval.reference import A_LITERAL, _norm_tokens

#: Non-lexical filled pauses (``textfeat.FILLER_RE`` without ``este``, which is lexical and ambiguous).
NONLEXICAL_FILLER_RE = re.compile(r"^(e+h*|eh+|ehm+|em+|mm+|hm+|ah+|aa+h*)$")
MAX_N = 4


def is_nonlexical_filler(tok: str, nxt: str | None) -> bool:
    if not NONLEXICAL_FILLER_RE.match(tok):
        return False
    if tok == "e" and nxt is not None and (nxt.startswith("i") or nxt.startswith("hi")):
        return False  # conjunction "e" (e indigenas, e hijos)
    return True


def lexical_tokens(text: str) -> tuple[list[str], list[str]]:
    """(tokens without non-lexical filled pauses, removed filler tokens). Tokens as ``reference._norm_tokens``."""
    toks = _norm_tokens(text)
    keep, fill = [], []
    for i, t in enumerate(toks):
        (
            fill
            if is_nonlexical_filler(t, toks[i + 1] if i + 1 < len(toks) else None)
            else keep
        ).append(t)
    return keep, fill


def literal_repetition_marks(tokens: list[str], max_n: int = MAX_N) -> list[dict]:
    out = []
    for n in range(1, max_n + 1):
        for i in range(len(tokens) - 2 * n + 1):
            mt, ct = tokens[i : i + n], tokens[i + n : i + 2 * n]
            if mt == ct:
                out.append(
                    {
                        "subtype": A_LITERAL,
                        "m": " ".join(mt),
                        "c": " ".join(ct),
                        "span": " ".join(tokens[i : i + 2 * n]),
                        "mt": mt,
                        "ct": ct,
                        "start_tok": i,
                        "n": n,
                    }
                )
    return out


def utterance_reference(text: str) -> dict:
    """Reference of one utterance. ``rep_literal`` is 1 / 0 (never indeterminate: see module docstring)."""
    toks, fill = lexical_tokens(text)
    marks = literal_repetition_marks(toks)
    raw_marks = literal_repetition_marks(_norm_tokens(text))
    return {
        "rep_literal": int(bool(marks)),
        "rep_literal_without_filler_removal": int(bool(raw_marks)),
        "rep_marks": marks,
        "n_rep_marks": len(marks),
        "fillers": dict(Counter(fill)),
        "n_fillers": len(fill),
        "n_este": sum(t == "este" for t in _norm_tokens(text)),
        "n_tokens": len(toks),
    }
