"""DIHANA user-turn importer.

``scripts/baselines/dihana_prepare.py`` runs ``main()`` at import time, so it cannot be
imported; its parsing rules are reproduced here verbatim and a parity test compares both
(when the corpus and ``turns.csv`` are available):

* input text = ``"".join(<t>.itertext())`` whitespace-normalised: it keeps every word that
  was written (literal repeats, ``<l cortada>`` fragments) and drops tags/attributes
  (``<f>`` fillers are empty elements, so the filler word is NOT in the text and is not
  reconstructed; ``palabra="..."`` attributes of ``<l>`` carry the intended word and are
  never exposed).
* labels (adapter convention, negative coverage documented by the DIHANA manual for these
  four families only): f -> filler_word, a -> prolongation, r:repeticion -> repetition,
  r:sustitucion|insercion|borrado -> revision. ``b``, ``l``, ``d``, ``n``, ``p`` are not
  mapped; block and the pause classes stay unknown.
* Known caveat (REPETICION_REVISION audit): DIHANA files cut-word restarts and corrected
  mispronunciations under ``r:repeticion``; the mapping is therefore ``approximate``.
"""

from __future__ import annotations

import collections
import unicodedata
import xml.etree.ElementTree as ET
from pathlib import Path

from mrcd.corpora.base import (
    CLASSES,
    Annotation,
    CorpusFormatError,
    Example,
    SourceFile,
    sha256_file,
)

CORPUS = "dihana"
_FAMILY = {
    "filler_word": "declared by the DIHANA manual (turn-level presence of <f>)",
    "prolongation": "declared by the DIHANA manual (turn-level presence of <a>)",
    "repetition": "declared by the DIHANA manual (turn-level presence of <r repeticion>)",
    "revision": "declared by the DIHANA manual (turn-level presence of <r sustitucion/insercion/borrado>)",
}


def _strip(s: str) -> str:
    return unicodedata.normalize("NFD", s).encode("ascii", "ignore").decode().lower()


def turn_tag_counts(t: ET.Element) -> collections.Counter:
    tags: collections.Counter = collections.Counter()
    for el in t.iter():
        if el is t:
            continue
        ty = el.get("tipo")
        tags[f"{el.tag}:{_strip(ty)}" if ty else el.tag] += 1
    return tags


def dihana_input_view(t: ET.Element) -> str:
    return " ".join("".join(t.itertext()).split())


def _class_counts(tags: collections.Counter) -> dict[str, int]:
    return {
        "filler_word": tags["f:a"]
        + tags["f:e"]
        + tags["f:m"]
        + tags["f:sin_identificar"],
        "prolongation": tags["a"],
        "repetition": tags["r:repeticion"],
        "revision": tags["r:sustitucion"] + tags["r:insercion"] + tags["r:borrado"],
    }


def import_dihana_turns(
    xml_path: str | Path, root: str | Path, fold_of: dict[str, int] | None = None
) -> list[Example]:
    """All user turns of one DIHANA dialogue XML. Group = speaker (speaker-disjoint folds)."""
    xml_path, root = Path(xml_path), Path(root)
    try:
        rt = ET.parse(xml_path).getroot()
    except ET.ParseError as e:
        raise CorpusFormatError(f"{xml_path}: {e}") from e
    spk = [h.get("nombre") for h in rt.iter("hablante") if h.get("id") == "usuario"]
    if not spk or not spk[0]:
        raise CorpusFormatError(f"{xml_path}: no user speaker")
    speaker = spk[0]
    from mrcd.corpora.base import relative_posix

    src = SourceFile(relative_posix(root, xml_path), sha256_file(xml_path))
    out = []
    for tn in rt.iter("turno"):
        if tn.get("hablante") != "usuario":
            continue
        t = tn.find("t")
        if t is None:
            raise CorpusFormatError(f"{xml_path}: user turn without <t>")
        tags = turn_tag_counts(t)
        counts = _class_counts(tags)
        anns = []
        for el in t.iter():
            if el is t:
                continue
            ty = el.get("tipo")
            key = f"{el.tag}:{_strip(ty)}" if ty else el.tag
            cls = None
            if key in ("f:a", "f:e", "f:m", "f:sin_identificar"):
                cls = "filler_word"
            elif key == "a":
                cls = "prolongation"
            elif key == "r:repeticion":
                cls = "repetition"
            elif key in ("r:sustitucion", "r:insercion", "r:borrado"):
                cls = "revision"
            anns.append(
                Annotation(
                    "dihana_tag",
                    key,
                    "human",
                    text=" ".join("".join(el.itertext()).split()) or None,
                    proposed_class=cls,
                    mapping_strength=("approximate" if cls == "repetition" else "exact")
                    if cls
                    else "none",
                    justification=(
                        "Adapter convention (scripts/baselines/dihana_prepare.py)."
                        + (
                            " Also files cut-word restarts / corrected mispronunciations."
                            if cls == "repetition"
                            else ""
                        )
                    )
                    if cls
                    else "",
                )
            )
        cov = {c: "unknown" for c in CLASSES}
        for c, why in _FAMILY.items():
            cov[c] = "pos" if counts[c] else "neg"
        turn_id = f"{xml_path.stem}_{tn.get('id')}"
        out.append(
            Example(
                corpus=CORPUS,
                recording_id=turn_id,
                speaker_id=speaker,
                group_id=speaker,
                source_files=(src,),
                original_text=ET.tostring(t, encoding="unicode"),
                original_annotations=tuple(anns),
                temporal_unit="untimed",
                start_s=None,
                end_s=None,
                source_label=None,
                proposed_mrcd_mapping=None,
                mapping_justification="Turn-level presence labels; see original_annotations.",
                coverage=cov,
                negative_basis={
                    c: why for c, why in _FAMILY.items() if cov[c] == "neg"
                },
                provenance="human",
                input_view=dihana_input_view(t),
                extra={
                    "dialogue": xml_path.stem,
                    "fold": (fold_of or {}).get(speaker),
                    "tag_counts": dict(tags),
                    "audio_attr": tn.get("fichero_audio_ss"),
                },
            )
        )
    return out
