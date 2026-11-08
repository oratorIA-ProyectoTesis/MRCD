"""Shared constants for the discourse-marker function layer (annotation kit revision 2026-11-08b).

Field contract (do not rename): columns `marker_token` and `marker_function`.
A marker row in the sheet has class=discourse_marker, state=none, the span of the
marker, and both marker columns filled. Rows of every other class leave them empty.
"""
from __future__ import annotations

MARKER_CLASS = "discourse_marker"
# Closed list. Normalized forms use underscores so they survive Audacity labels.
# "eh" and "em" / "mm" are not in the list: they stay filled pauses (FP).
MARKER_TOKENS = ("este", "o_sea", "bueno", "pues", "a_ver", "tipo")
MARKER_FUNCTIONS = ("functional", "disfluent", "uncertain")
# Pre-declared minimum of matched marker spans for kappa / alpha.
MIN_MARKER_SPANS = 30
# Pre-declared: below this detection F1 the function kappa is descriptive only.
MIN_DETECTION_F1 = 0.5
# Exhaustive decision order for marker_function (GUIA_ANOTACION.md section 3.2).
DECISION_ORDER = (
    ("functional", "observable syntactic role"),
    ("disfluent", "no syntactic role AND deletable AND at least one adjacency cue"),
    ("uncertain", "everything else, including no role but no adjacency cue, or unclear audio"),
)
ADJACENCY_CUES = ("silent_or_filled_pause", "prolongation", "restart_or_repair")


def check_marker(token: str, function: str) -> tuple[str, str]:
    t, f = (token or "").strip().lower(), (function or "").strip().lower()
    if t not in MARKER_TOKENS:
        raise ValueError(f"marker_token {token!r} not in closed list {MARKER_TOKENS}")
    if f not in MARKER_FUNCTIONS:
        raise ValueError(f"marker_function {function!r} not in {MARKER_FUNCTIONS}")
    return t, f
