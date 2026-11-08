"""Verbatim target text and speaker split for Plan B (results/plan_b_20261108/PROTOCOL_EXPLORATORY_B.md). Pure functions.

DIHANA ``<t>`` text already keeps what was said: literal repetitions (``<r tipo=repeticion>``), cut-word fragments
(``<l tipo=cortada>`` keeps the fragment, the intended word is only an attribute), mispronunciations as pronounced, and
the words inside discourse (``<d>``), abandoned-segment (``<b>``), prolongation (``<a>``) and prolonged-noise (``<n>``)
marks. Filled pauses are EMPTY ``<f tipo=a|e|m|sin_identificar/>`` elements, so the filler word is absent from the text
and is reconstructed here from the tag (``FILLER_WORD``). Everything else that carries no word (``<p/>`` pauses, point
noises ``<n/>``) is dropped, and the punctuation spacing of the transcripts (``" ."``, ``"¿ "``) is closed.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
import xml.etree.ElementTree as ET

#: ``<f tipo=...>`` -> written filled pause. ``a``/``e``/``m`` are the guide's vowel/nasal realisations;
#: ``sin_identificar`` ("trash": unidentified sound) is written as the most frequent form ``eh`` (convention).
FILLER_WORD = {"a": "ah", "e": "eh", "m": "mm", "sin_identificar": "eh"}
SPLIT_SEED = "20261108"


def _strip(s: str) -> str:
    return (
        unicodedata.normalize("NFD", s or "").encode("ascii", "ignore").decode().lower()
    )


def _close_punct(s: str) -> str:
    s = re.sub(r"\s+([,.;:?!])", r"\1", s)
    s = re.sub(r"([¿¡])\s+", r"\1", s)
    return " ".join(s.split())


def verbatim_target(t_xml: str) -> str:
    """Training target from one marked ``<t>`` element: every written word kept, filled pauses reconstructed."""
    t = ET.fromstring(t_xml)
    parts: list[str] = []

    def rec(el: ET.Element) -> None:
        if el.tag == "f":
            kind = _strip(el.get("tipo"))
            if kind not in FILLER_WORD:
                raise ValueError(f"unknown filled-pause type {el.get('tipo')!r}")
            parts.append(f" {FILLER_WORD[kind]} ")
            if el.text and el.text.strip():
                raise ValueError("filled-pause element with text")
            return
        if el.text:
            parts.append(el.text)
        for ch in el:
            rec(ch)
            if ch.tail:
                parts.append(ch.tail)

    rec(t)
    return _close_punct("".join(parts))


def filler_free(t_xml: str) -> str:
    """Same text as the corpus input view (``"".join(itertext())``), for consistency checks."""
    return " ".join("".join(ET.fromstring(t_xml).itertext()).split())


def _key(speaker: str, seed: str) -> str:
    return hashlib.sha256(f"{seed}:{speaker}".encode()).hexdigest()


def split_speakers(
    speakers_pos: dict[str, bool], frac: float, seed: str = SPLIT_SEED, exclude=()
) -> set[str]:
    """Speakers drawn for a held-out set: ``round(frac * n)`` of the speakers WITH a literal-repetition turn and of
    those WITHOUT, each stratum by ascending ``sha256(seed:speaker)``. ``exclude`` speakers are never drawn."""
    ex = set(exclude)
    pos = sorted(
        (s for s, p in speakers_pos.items() if p and s not in ex),
        key=lambda s: _key(s, seed),
    )
    neg = sorted(
        (s for s, p in speakers_pos.items() if not p and s not in ex),
        key=lambda s: _key(s, seed),
    )
    return set(pos[: round(frac * len(pos))]) | set(neg[: round(frac * len(neg))])


def make_split(
    speakers_pos: dict[str, bool], seed: str = SPLIT_SEED
) -> dict[str, set[str]]:
    """val = 20 % of all speakers (stratified); dev = 10 % of the rest (checkpoint selection); train = the others."""
    val = split_speakers(speakers_pos, 0.2, seed)
    dev = split_speakers(speakers_pos, 0.1, seed + ":dev", exclude=val)
    train = set(speakers_pos) - val - dev
    assert not (val & dev) and not (val & train) and not (dev & train)
    return {"train": train, "dev": dev, "val": val}
