"""Occurrences of ambiguous Spanish forms in DIHANA user turns, with the reference mark kept apart.

The *input view* is the turn text with every tag removed (same string as
``mrcd.corpora.dihana.dihana_input_view``).  The reference ``d`` mark that covers an occurrence is
returned only as a label field and is never part of the text a model sees.
"""

from __future__ import annotations

import re
import unicodedata
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

# surface forms (normalised, accent-free, lower-case) -> canonical form name
FORMS: dict[tuple[str, ...], str] = {
    ("este",): "este",
    ("bueno",): "bueno",
    ("o", "sea"): "o sea",
    ("pues",): "pues",
    ("entonces",): "entonces",
    ("mira",): "mira/mire",
    ("mire",): "mira/mire",
    ("vale",): "vale",
    ("a", "ver"): "a ver",
    ("eh",): "eh",
}
MAX_LEN = max(len(k) for k in FORMS)


def norm(s: str) -> str:
    s = "".join(
        c
        for c in unicodedata.normalize("NFD", s.lower())
        if unicodedata.category(c) != "Mn"
    )
    return re.sub(r"[^a-z0-9ñ]+", "", s)


@dataclass
class Token:
    text: str
    norm: str
    d_id: int | None  # index of the enclosing <d> element (reference only)
    d_type: str | None  # its tipo, accent-free (reference only)


@dataclass
class Occurrence:
    form: str
    start: int  # token index in the turn (input view)
    end: int  # exclusive
    d_type: str | None  # reference: tipo of the covering <d>, None if not covered
    f_before: int = 0  # reference: <f/> marks immediately before the occurrence
    f_after: int = 0
    extra: dict = field(default_factory=dict)


def _walk(
    el: ET.Element, d_id: int | None, d_type: str | None, out: list, counter: list
):
    """Append (char, d_id, d_type) triples of the element text in document order."""
    cur_id, cur_type = d_id, d_type
    if el.tag == "d":
        counter[0] += 1
        cur_id, cur_type = counter[0], norm(el.get("tipo") or "") or "sin_tipo"
    if el.text:
        out.extend((c, cur_id, cur_type) for c in el.text)
    for ch in el:
        _walk(ch, cur_id, cur_type, out, counter)
        if ch.tail:
            out.extend((c, cur_id, cur_type) for c in ch.tail)


def tokens_of_turn(t: ET.Element) -> list[Token]:
    chars: list = []
    _walk(t, None, None, chars, [0])
    toks: list[Token] = []
    cur: list = []

    def flush():
        if cur:
            text = "".join(c for c, _, _ in cur)
            toks.append(Token(text, norm(text), cur[0][1], cur[0][2]))
            cur.clear()

    for c, i, ty in chars:
        if c.isspace():
            flush()
        else:
            cur.append((c, i, ty))
    flush()
    return toks


def f_marks_between(t: ET.Element) -> list[int]:
    """Number of ``<f/>`` marks that precede each token index (cumulative), reference only."""
    # Re-walk emitting a sentinel for each <f/> so we can place it among tokens.
    seq: list[str] = []

    def rec(el):
        if el.tag == "f":
            seq.append("\x00")
        if el.text:
            seq.append(el.text)
        for ch in el:
            rec(ch)
            if ch.tail:
                seq.append(ch.tail)

    rec(t)
    out: list[int] = []
    pending = 0
    in_tok = False
    for piece in seq:
        if piece == "\x00":
            pending += 1
            in_tok = False
            continue
        for c in piece:
            if c.isspace():
                in_tok = False
            elif not in_tok:
                out.append(pending)
                in_tok = True
    return out


def find_occurrences(
    toks: list[Token], f_cum: list[int] | None = None
) -> list[Occurrence]:
    """Non-overlapping occurrences (longest form first) over the word tokens of a turn."""
    occs: list[Occurrence] = []
    i = 0
    n = len(toks)
    while i < n:
        hit = None
        for L in range(MAX_LEN, 0, -1):
            if i + L > n:
                continue
            key = tuple(t.norm for t in toks[i : i + L])
            if key in FORMS and all(key):
                hit = (L, FORMS[key])
                break
        if hit is None:
            i += 1
            continue
        L, form = hit
        span = toks[i : i + L]
        ids = {t.d_id for t in span}
        d_type = span[0].d_type if len(ids) == 1 and span[0].d_id is not None else None
        o = Occurrence(form, i, i + L, d_type)
        if f_cum is not None and len(f_cum) == n:
            prev = f_cum[i - 1] if i > 0 else 0
            o.f_before = f_cum[i] - prev
            nxt = f_cum[i + L] if i + L < n else None
            o.f_after = (nxt - f_cum[i + L - 1]) if nxt is not None else 0
        occs.append(o)
        i += L
    return occs
