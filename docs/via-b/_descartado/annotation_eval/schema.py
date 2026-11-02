"""Names and vocabularies shared by the EAF template, the validator, the importer and the agreement code."""

from __future__ import annotations

import re

from mrcd.constants import TAXONOMY

#: Core classes of the confirmatory evaluation (fixed by the pre-registration).
CORE_CLASSES: tuple[str, ...] = ("filler_word", "prolongation", "repetition")
#: Optional layers: annotated and agreement-checked, never part of the confirmatory family unless the gate passes.
OPTIONAL_CLASSES: tuple[str, ...] = ("revision", "block", "rhetorical_pause", "neutral_pause")
ALL_CLASSES: tuple[str, ...] = CORE_CLASSES + OPTIONAL_CLASSES
assert set(ALL_CLASSES) == set(TAXONOMY)

#: ELAN tiers: pauses share one tier (the function of a silent pause) because the two functions are exclusive.
TIER_KEYS: tuple[str, ...] = ("filler_word", "prolongation", "repetition", "revision", "block", "pause_function", "excluir")

#: Controlled vocabulary of each tier (the value written in ELAN) and the MRCD class it maps to.
VOCAB: dict[str, dict[str, str | None]] = {
    "filler_word": {"filler_word": "filler_word", "filler_word_incierto": None},
    "prolongation": {"prolongation": "prolongation", "prolongation_incierto": None},
    "repetition": {"repetition": "repetition", "repetition_incierto": None},
    "revision": {"revision": "revision", "revision_incierto": None},
    "block": {"block": "block", "block_incierto": None},
    "pause_function": {
        "retorica": "rhetorical_pause",
        "neutra": "neutral_pause",
        "pausa_incierto": None,
    },
    "excluir": {"claqueta": None, "otra_voz": None, "ruido": None, "fuera_de_tarea": None},
}
#: Which MRCD class each uncertain value belongs to (for ignore regions).
UNCERTAIN_CLASS: dict[str, tuple[str, ...]] = {
    "filler_word_incierto": ("filler_word",),
    "prolongation_incierto": ("prolongation",),
    "repetition_incierto": ("repetition",),
    "revision_incierto": ("revision",),
    "block_incierto": ("block",),
    "pausa_incierto": ("rhetorical_pause", "neutral_pause"),
}
#: Classes whose coverage is declared per tier in the EAF header.
TIER_CLASSES: dict[str, tuple[str, ...]] = {
    "filler_word": ("filler_word",),
    "prolongation": ("prolongation",),
    "repetition": ("repetition",),
    "revision": ("revision",),
    "block": ("block",),
    "pause_function": ("rhetorical_pause", "neutral_pause"),
}
COVERAGE_VALUES = ("complete", "not_annotated")
ANNOTATOR_RE = re.compile(r"^(A\d{1,2}|ADJ)$")
TIER_RE = re.compile(r"^(?P<ann>A\d{1,2}|ADJ)_(?P<key>[a-z_]+)$")
#: Pseudonymous recording identifier: VB<phase>_S<nn>_<task><k>; phase P = practice (calibration), T = test.
RECORDING_RE = re.compile(r"^VB(?P<phase>[PT])_S(?P<spk>\d{2})_(?P<task>PRES|ESP)(?P<n>\d)$")
PROPERTY_PREFIX_COVERAGE = "coverage."
GUIDE_PROPERTY = "guide_version"


def tier_id(annotator: str, key: str) -> str:
    if not ANNOTATOR_RE.match(annotator) or key not in TIER_KEYS:
        raise ValueError(f"invalid annotator/tier: {annotator!r}/{key!r}")
    return f"{annotator}_{key}"


def parse_recording_id(rec: str) -> dict:
    m = RECORDING_RE.match(rec)
    if not m:
        raise ValueError(f"recording id {rec!r} does not match VB<P|T>_S<nn>_<PRES|ESP><k>")
    d = m.groupdict()
    d["speaker"] = f"VB{d['phase']}_S{d['spk']}"
    return d
