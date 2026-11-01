"""Reference derivation from DIHANA turn XML (``<t>`` string). Re-derives the sub-types of REPETICION_REVISION.

Used ONLY to build the reference (y) and descriptive breakdowns, never as a model input.
"""

from __future__ import annotations

import re
import unicodedata
import xml.etree.ElementTree as ET

A_LITERAL = "A_literal_repeat"
B_CUT = "B_cut_word_restart"
C_MISPRON = "C_mispronunciation_corrected"
D_OTHER = "D_other_or_expanded"
REV_TYPES = {"sustitucion", "insercion", "borrado"}


def _strip_accents(s: str) -> str:
    return unicodedata.normalize("NFD", s).encode("ascii", "ignore").decode().lower()


def _norm_tokens(s: str) -> list[str]:
    s = "".join(
        c
        for c in unicodedata.normalize("NFD", s.lower())
        if unicodedata.category(c) != "Mn"
    )
    return re.sub(r"[^a-z0-9ñ ]+", " ", s).split()


def _txt(el: ET.Element | None) -> str:
    return " ".join("".join(el.itertext()).split()) if el is not None else ""


def repetition_marks(t: ET.Element) -> list[dict]:
    """One dict per ``<r tipo=repeticion>`` element with its sub-type (A / B / C / D)."""
    out = []
    for e in t.iter("r"):
        if _strip_accents(e.get("tipo") or "") != "repeticion":
            continue
        m, c = e.find("m"), e.find("c")
        mt, ct = _norm_tokens(_txt(m)), _norm_tokens(_txt(c))
        inner = {
            d.tag + (":" + d.get("tipo") if d.get("tipo") else "")
            for d in e.iter()
            if d is not e and d.tag not in ("m", "s", "c")
        }
        has_cut, has_mis = "l:cortada" in inner, "l:mal_pronunciada" in inner
        if mt == ct:
            rel = "identical"
        elif mt and ct and ct[: len(mt)] == mt:
            rel = "m_prefix_of_c"
        elif (
            mt
            and ct
            and len(ct) >= len(mt)
            and ct[: len(mt) - 1] == mt[:-1]
            and ct[len(mt) - 1].startswith(mt[-1])
        ):
            rel = "fragment_then_word"
        else:
            rel = "other"
        if rel == "identical" and not has_cut and not has_mis:
            sub = A_LITERAL
        elif has_mis:
            sub = C_MISPRON
        elif has_cut or rel == "fragment_then_word":
            sub = B_CUT
        elif rel == "identical":
            sub = A_LITERAL
        elif len(set(mt + ct)) == 1:
            sub = A_LITERAL
        else:
            sub = D_OTHER
        out.append(
            {
                "subtype": sub,
                "m": _txt(m),
                "c": _txt(c),
                "span": _txt(e),
                "mt": mt,
                "ct": ct,
            }
        )
    return out


def revision_marks(t: ET.Element) -> list[dict]:
    out = []
    for e in t.iter("r"):
        if _strip_accents(e.get("tipo") or "") not in REV_TYPES:
            continue
        m, s, c = e.find("m"), e.find("s"), e.find("c")
        inner = {
            d.tag + (":" + d.get("tipo") if d.get("tipo") else "")
            for d in e.iter()
            if d is not e and d.tag not in ("m", "s", "c")
        }
        if "l:cortada" in inner:
            sub = "R2_cut_word_then_restart"
        elif _txt(s):
            sub = "R1_explicit_edit_signal"
        elif any(k.startswith("r:") for k in inner):
            sub = "R3_contains_nested_repetition"
        else:
            sub = "R4_silent_substitution_insertion_deletion"
        out.append(
            {
                "subtype": sub,
                "m": _txt(m),
                "s": _txt(s),
                "c": _txt(c),
                "span": _txt(e),
                "mt": _norm_tokens(_txt(m)),
                "ct": _norm_tokens(_txt(c)),
            }
        )
    return out


def turn_reference(t_xml: str) -> dict:
    """Turn-level reference from the marked ``<t>`` XML string.

    rep_literal: 1 (>=1 A mark), 0 (no repetition mark), None (only B/C/D marks: indeterminate).
    """
    t = ET.fromstring(t_xml)
    reps, revs = repetition_marks(t), revision_marks(t)
    subs = {r["subtype"] for r in reps}
    if A_LITERAL in subs:
        rep_literal: int | None = 1
    elif reps:
        rep_literal = None
    else:
        rep_literal = 0
    return {
        "rep_literal": rep_literal,
        "rep_full": int(bool(reps)),
        "revision": int(bool(revs)),
        "rep_subtypes": sorted(subs),
        "rev_subtypes": sorted({r["subtype"] for r in revs}),
        "n_rep_marks": len(reps),
        "n_rev_marks": len(revs),
        "rep_marks": reps,
        "rev_marks": revs,
    }
