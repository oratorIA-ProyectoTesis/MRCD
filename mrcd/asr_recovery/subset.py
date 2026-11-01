"""O1 frame: all positives of either task + per-speaker sampled negatives with inverse-sampling weights (protocol 3.2)."""

from __future__ import annotations

import math
from collections import defaultdict

from mrcd.asr_recovery.common import sha_text


def is_positive(u: dict) -> bool:
    return u["ref"]["rep_literal"] == 1 or u["ref"]["revision"] == 1


def is_eligible_negative(u: dict) -> bool:
    return u["ref"]["rep_literal"] == 0 and u["ref"]["revision"] == 0


def build_subset(pool: list[dict], ratio: int, seed: int = 13) -> list[dict]:
    """Deterministic frame. Speakers without a positive do not enter. Negatives of speaker s: min(m_s, ceil(ratio * n_pos_s))."""
    pos = defaultdict(list)
    neg = defaultdict(list)
    for u in pool:
        if is_positive(u):
            pos[u["speaker"]].append(u)
        elif is_eligible_negative(u):
            neg[u["speaker"]].append(u)
    out: list[dict] = []
    for s in sorted(pos):
        for u in sorted(pos[s], key=lambda x: x["unit_id"]):
            out.append(_row(u, "pos", 1.0))
        cand = sorted(neg.get(s, []), key=lambda x: sha_text(f"{seed}|asr_recovery|{x['unit_id']}"))
        m = len(cand)
        take = min(m, math.ceil(ratio * len(pos[s])))
        for u in cand[:take]:
            out.append(_row(u, "neg", m / take))
    return out


def _row(u: dict, kind: str, w: float) -> dict:
    return {
        "unit_id": u["unit_id"],
        "speaker": u["speaker"],
        "fold": u["fold"],
        "kind": kind,
        "w": w,
        "rep_literal": u["ref"]["rep_literal"],
        "revision": u["ref"]["revision"],
    }


def choose_ratio(total_seconds_by_ratio: dict[int, float], rtf: float, budget_min: float = 90.0, margin: float = 1.5) -> int:
    """Largest r in {1,2,3} whose estimated single-process time (margin * RTF * seconds) fits the budget; else 1."""
    ok = [r for r, sec in sorted(total_seconds_by_ratio.items()) if margin * rtf * sec / 60.0 <= budget_min]
    return max(ok) if ok else 1
