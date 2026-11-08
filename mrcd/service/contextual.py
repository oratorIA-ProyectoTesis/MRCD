"""Candidate generation + contextual demotion of decision profile ``o2026_12`` (plan A, exploratory).

Protocol: ``results/plan_a_20261108/PROTOCOL_EXPLORATORY_A.md`` (frozen before any run). Pure functions of ASR words
(dicts with ``text``, ``norm``, ``start``, ``end``, ``prob``, ``punct_after``); no audio, no model, no randomness.

Repetition
    Generator = the confirmatory lexical ``rule`` (``scripts/confirmatoria/run_confirmatoria.py``: ``norm_tokens``,
    non-lexical fillers removed, immediate exact repetition of 1-4 tokens, interval = the second copy). The contextual
    layer may only DEMOTE a candidate (never create, move or extend one):

    R1 ``loop``     candidate inside a hallucination loop (``asr_profiles._scan_loops``)        -> ``uncertain``
    R2 ``gap``      silence between the copies > ``H["repetition_max_gap_s"]`` (1.5 s)        -> ``uncertain``
    R3 ``seam``     the two copies were decoded by different 8 s / 2 s ASR chunks               -> ``uncertain``
    R4 ``polarity`` one-token repeat of a ``textfeat.POLARITY`` item ("si, si")                -> ``functional``

    DIHANA has no function labels, so ``functional`` is not validated (``DISAMBIGUATION_SPEC.md`` section 10).

Filler subclasses (plan E)
    ``filled_pause``   non-lexical filled pause: the ``filler_word`` unit pipeline (``decision_profile``).
    ``lexical_filler`` lexical items of ``constants.FILLER_LEXICON["es"]``; state by the auto-labeler rule for ambiguous
                       items (adjacent gap >= 0.15 s, duration >= 0.35 s or comma/ellipsis -> ``disfluent``, otherwise
                       ``uncertain``).
"""

from __future__ import annotations

import re
import unicodedata

from mrcd.asr_profiles import _scan_loops
from mrcd.constants import FILLER_LEXICON, H
from mrcd.context_eval.textfeat import POLARITY

DISFLUENT, FUNCTIONAL, UNCERTAIN = "disfluent", "functional", "uncertain"
CONTEXT_STATES = (DISFLUENT, FUNCTIONAL, UNCERTAIN)
FILLED_PAUSE, LEXICAL_FILLER = "filled_pause", "lexical_filler"
FILLER_SUBCLASSES = (FILLED_PAUSE, LEXICAL_FILLER)
REPETITION_RULES = ("loop", "gap", "seam", "polarity")
#: rule -> state it demotes to (evaluation order = this order; the first match sets the state)
RULE_STATE = {
    "loop": UNCERTAIN,
    "gap": UNCERTAIN,
    "seam": UNCERTAIN,
    "polarity": FUNCTIONAL,
}
MAX_N = 4
#: identical to the confirmatory ``NONLEXICAL_FILLER_RE`` and ``transcript_reference.NONLEXICAL_FILLER_RE``
NONLEXICAL_FILLER_RE = re.compile(r"^(e+h*|eh+|ehm+|em+|mm+|hm+|ah+|aa+h*)$")
#: ASR chunking of the p1 profile (``asr_profiles``: chunk_s 8.0, overlap_s 2.0)
CHUNK_S, OVERLAP_S = 8.0, 2.0

#: ``annotation/auto_labeler.detect_fillers``: items with a full lexical use that need hesitation evidence
_AMBIGUOUS = frozenset(
    {"este", "bueno", "pues", "tipo", "o sea", "ya", "a ver", "digamos", "pe"}
)
#: lexical entries of the existing Spanish filler lexicon (not extended; protocol section 2)
LEXICAL_FILLERS = frozenset(
    x for x in FILLER_LEXICON["es"] if x in _AMBIGUOUS or x.startswith("este")
)
EVIDENCE_GAP_S, EVIDENCE_DUR_S = 0.15, 0.35
EVIDENCE_PUNCT = frozenset({",", "...", "…"})


# ------------------------------------------------------------------------------------------- tokens (confirmatory)
def norm_tokens(s: str) -> list[str]:
    s = "".join(
        c
        for c in unicodedata.normalize("NFD", (s or "").lower())
        if unicodedata.category(c) != "Mn"
    )
    return re.sub(r"[^a-z0-9ñ ]+", " ", s).split()


def is_filler(tok: str, nxt: str | None) -> bool:
    if not NONLEXICAL_FILLER_RE.match(tok):
        return False
    if tok == "e" and nxt is not None and (nxt.startswith("i") or nxt.startswith("hi")):
        return False  # conjunction "e"
    return True


def lexical_stream(words: list[dict]) -> list[tuple[str, float, float]]:
    """Filler-stripped token stream ``(token, start, end)`` exactly as the confirmatory ``rule_events``."""
    toks = []
    for w in words or []:
        for t in norm_tokens(str(w.get("text", ""))):
            toks.append((t, float(w["start"]), float(w["end"])))
    rest = []
    for i, (t, s, e) in enumerate(toks):
        nxt = toks[i + 1][0] if i + 1 < len(toks) else None
        if not is_filler(t, nxt):
            rest.append((t, s, e))
    return rest


def repeat_positions(tokens: list[str], max_n: int = MAX_N) -> list[tuple[int, int]]:
    """(start index of the SECOND occurrence, n) of every immediate exact repetition of 1..max_n tokens."""
    out = []
    for n in range(1, max_n + 1):
        for i in range(len(tokens) - 2 * n + 1):
            if tokens[i : i + n] == tokens[i + n : i + 2 * n]:
                out.append((i + n, n))
    return out


# ------------------------------------------------------------------------------------------- repetition
def chunk_seams(
    duration_s: float | None, chunk_s: float = CHUNK_S, overlap_s: float = OVERLAP_S
) -> list[float]:
    """Hand-over points of ``transcribe_chunked``: chunk k (k >= 1) starts at ``k * step`` and keeps words whose
    midpoint is >= ``k * step + overlap``. No seam when the audio is not chunked (duration <= chunk_s)."""
    if duration_s is None or duration_s <= chunk_s or chunk_s <= 0:
        return []
    step = chunk_s - overlap_s
    out, k = [], 1
    while (
        k * step < duration_s and duration_s - k * step >= 1.0
    ):  # a chunk shorter than 1 s is not decoded
        out.append(k * step + overlap_s)
        k += 1
    return out


def _loop_ranges(tokens: list[str]) -> list[tuple[int, int]]:
    out = []
    for unit, k, i in _scan_loops(tokens):
        out.append((i, i + k * len(unit)))
    return out


def classify_repetitions(
    words: list[dict],
    duration_s: float | None = None,
    rules: tuple[str, ...] = REPETITION_RULES,
) -> list[dict]:
    """Every generator candidate with its contextual state. ``rules`` selects the active demotion rules (the profile
    uses all four; the protocol's secondary reading ``o2026_12_u`` uses loop, gap and seam only)."""
    rest = lexical_stream(words)
    toks = [t for t, _, _ in rest]
    loops = _loop_ranges(toks)
    seams = chunk_seams(duration_s)
    out = []
    for pos, n in repeat_positions(toks):
        a, b = pos - n, pos + n  # token range of both copies
        c1_last, c2_first = rest[pos - 1], rest[pos]
        mid1 = (c1_last[1] + c1_last[2]) / 2
        mid2 = (c2_first[1] + c2_first[2]) / 2
        matched = []
        if any(lo < b and a < hi for lo, hi in loops):
            matched.append("loop")
        if c2_first[1] - c1_last[2] > float(H["repetition_max_gap_s"]):
            matched.append("gap")
        if any(mid1 < s <= mid2 for s in seams):
            matched.append("seam")
        if n == 1 and toks[pos] in POLARITY:
            matched.append("polarity")
        active = [r for r in REPETITION_RULES if r in matched and r in rules]
        out.append(
            {
                "pos": pos,
                "n": n,
                "tokens": toks[pos : pos + n],
                "start_s": rest[pos][1],
                "end_s": rest[pos + n - 1][2],
                "state": RULE_STATE[active[0]] if active else DISFLUENT,
                "rule": active[0] if active else None,
                "matched_rules": matched,
            }
        )
    return out


def merge_disfluent(cands: list[dict]) -> list[dict]:
    """Disfluent candidates merged into events: overlapping or touching second-copy token ranges become one event."""
    spans = sorted(
        (c["pos"], c["pos"] + c["n"], c["start_s"], c["end_s"])
        for c in cands
        if c["state"] == DISFLUENT
    )
    merged: list[list] = []
    for a, b, s, e in spans:
        if merged and a <= merged[-1][1]:
            m = merged[-1]
            m[1], m[3] = max(m[1], b), max(m[3], e)
        else:
            merged.append([a, b, s, e])
    return [{"start_s": s, "end_s": e, "token_range": (a, b)} for a, b, s, e in merged]


def demotion_stats(cands: list[dict]) -> dict:
    st = {s: 0 for s in CONTEXT_STATES}
    by_rule = {r: 0 for r in REPETITION_RULES}
    matched = {r: 0 for r in REPETITION_RULES}
    for c in cands:
        st[c["state"]] += 1
        if c["rule"]:
            by_rule[c["rule"]] += 1
        for r in c["matched_rules"]:
            matched[r] += 1
    return {
        "n_candidates": len(cands),
        "by_state": st,
        "demoted_by_rule": by_rule,
        "matched_by_rule": matched,
    }


# ------------------------------------------------------------------------------------------- lexical filler
def _norm(w: dict) -> str:
    return str(w.get("norm") or "").strip()


def classify_lexical_fillers(words: list[dict]) -> list[dict]:
    """Lexical filler candidates (``LEXICAL_FILLERS``) with their state, following ``auto_labeler.detect_fillers``:
    a two-word item is matched first; a missing neighbour counts as a 1.0 s gap."""
    words = list(words or [])
    out, skip = [], set()
    for i, w in enumerate(words):
        if i in skip:
            continue
        nxt = words[i + 1] if i + 1 < len(words) else None
        prev = words[i - 1] if i else None
        tok, end = _norm(w), float(w["end"])
        if nxt is not None and f"{_norm(w)} {_norm(nxt)}" in LEXICAL_FILLERS:
            tok, end = f"{_norm(w)} {_norm(nxt)}", float(nxt["end"])
            skip.add(i + 1)
        if tok not in LEXICAL_FILLERS:
            continue
        gap_b = float(w["start"]) - float(prev["end"]) if prev is not None else 1.0
        if nxt is not None and tok == _norm(w):
            gap_a = float(nxt["start"]) - end
        else:
            gap_a = float(words[i + 2]["start"]) - end if i + 2 < len(words) else 1.0
        dur = end - float(w["start"])
        punct = str(w.get("punct_after") or "")
        if tok in _AMBIGUOUS:
            evidence = (
                gap_b >= EVIDENCE_GAP_S
                or gap_a >= EVIDENCE_GAP_S
                or dur >= EVIDENCE_DUR_S
                or punct in EVIDENCE_PUNCT
            )
        else:  # lengthened "esteee": the auto-labeler needs no extra evidence
            evidence = True
        out.append(
            {
                "token": tok,
                "start_s": float(w["start"]),
                "end_s": end,
                "state": DISFLUENT if evidence else UNCERTAIN,
                "evidence": {
                    "gap_before_s": round(gap_b, 3),
                    "gap_after_s": round(gap_a, 3),
                    "duration_s": round(dur, 3),
                    "punct_after": punct,
                },
            }
        )
    return out


def lexical_stats(cands: list[dict]) -> dict:
    st = {s: 0 for s in CONTEXT_STATES}
    for c in cands:
        st[c["state"]] += 1
    return {"n_candidates": len(cands), "by_state": st}
