"""Does a human mark survive in the ASR text? Descriptive heuristic of REPETICION_REVISION (not a label).

Same rule as scripts/audit2/b1: the human span is recovered in order with >= 80% of tokens and, for
repetitions, an adjacent duplicate exists inside the aligned ASR region; for revisions both the reparandum
and the correction are recovered at >= 80%.
"""

from __future__ import annotations

import difflib

from mrcd.context_eval.reference import _norm_tokens


def seq_cov(asr: list[str], ref: list[str]):
    if not ref:
        return 0.0, None
    sm = difflib.SequenceMatcher(None, asr, ref, autojunk=False)
    blocks = [b for b in sm.get_matching_blocks() if b.size]
    cov = sum(b.size for b in blocks) / len(ref)
    region = (blocks[0].a, blocks[-1].a + blocks[-1].size) if blocks else None
    return cov, region


def adj_repeat(tk: list[str], lo: int = 0, hi: int | None = None) -> list[tuple]:
    hi = len(tk) if hi is None else hi
    out = []
    for n in (1, 2, 3, 4):
        for i in range(lo, hi - 2 * n + 1):
            if tk[i : i + n] == tk[i + n : i + 2 * n]:
                out.append((i, n, "exact"))
    for i in range(lo, hi - 1):
        a, b = tk[i], tk[i + 1]
        if a != b and 1 <= len(a) < len(b) and b.startswith(a) and len(a) <= 3:
            out.append((i, 1, "fragment"))
    return out


def mark_survives(kind: str, mark: dict, asr_text: str) -> bool:
    """``kind`` is 'repetition' or 'revision'; ``mark`` from reference.*_marks."""
    asr = _norm_tokens(asr_text)
    if not asr:
        return False
    span = _norm_tokens(mark["span"])
    cov_span, region = seq_cov(asr, span)
    if kind == "repetition":
        return cov_span >= 0.8 and bool(region and adj_repeat(asr, *region))
    cov_m, _ = seq_cov(asr, mark["mt"])
    cov_c, _ = seq_cov(asr, mark["ct"])
    return cov_m >= 0.8 and cov_c >= 0.8 and cov_span >= 0.8


def turn_survives(
    kind: str, marks: list[dict], asr_text: str, subtypes: set[str] | None = None
) -> bool:
    return any(
        mark_survives(kind, m, asr_text)
        for m in marks
        if subtypes is None or m["subtype"] in subtypes
    )
