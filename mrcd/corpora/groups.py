"""Group assignment and leakage checks for train/dev/test partitions.

The independence unit is the recording (DIME dialogue, ESLORA recording). Every
derived artefact (AVI/WAV copies, fragments, caches, samples cut from a dialogue)
inherits the group of its parent. Sample parents are NOT hardcoded: they are read
from the audited cross-correlation CSV (``xcorr_samples.csv``).
"""

from __future__ import annotations

import csv
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping

from mrcd.corpora.base import CorpusFormatError


class AmbiguousGroupError(ValueError):
    """A file name points to more than one parent recording."""


class UnresolvedGroupError(ValueError):
    """A file name carries no recording identifier (or its parent is not audited)."""


class PartitionLeakError(ValueError):
    """The same group appears in more than one split, or a source is duplicated."""


_DIALOG_RE = re.compile(r"^(d\d{2})(?!\d)", re.IGNORECASE)
_SAMPLE_RE = re.compile(r"(utt\d+)(?!\d)", re.IGNORECASE)
_ESLORA_RE = re.compile(r"(SCOM_[A-Z0-9]+_\d+)")


def load_sample_parents(
    csv_path: str | Path, *, min_ncc: float = 0.9, min_margin: float = 0.3
) -> dict[str, str]:
    """sample -> parent dialogue from the audited normalised cross-correlation table.

    A parent is accepted only if its best ncc >= ``min_ncc`` and beats the runner-up
    by ``min_margin``; otherwise the sample stays unresolved (never guessed).
    """
    scores: dict[str, list[tuple[float, str]]] = defaultdict(list)
    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            try:
                scores[row["sample"].lower()].append(
                    (float(row["max_ncc"]), row["dialog"])
                )
            except (KeyError, ValueError) as e:
                raise CorpusFormatError(f"Malformed correspondence row {row}") from e
    parents: dict[str, str] = {}
    for sample, rows in scores.items():
        rows.sort(reverse=True)
        best = rows[0]
        second = rows[1][0] if len(rows) > 1 else 0.0
        if best[0] >= min_ncc and best[0] - second >= min_margin:
            parents[sample] = best[1].lower()
    return parents


def group_for_name(
    name: str | Path, sample_parents: Mapping[str, str] | None = None
) -> str:
    """Group id for any file derived from a recording, from its base name alone."""
    base = Path(str(name).replace("\\", "/")).name
    found: set[str] = set()
    m = _ESLORA_RE.search(base)
    if m:
        found.add(m.group(1))
    m = _DIALOG_RE.match(base)
    if m:
        found.add(m.group(1).lower())
    m = _SAMPLE_RE.search(base)
    if m:
        sample = m.group(1).lower()
        parent = (sample_parents or {}).get(sample)
        if parent is None:
            raise UnresolvedGroupError(
                f"{base}: parent of {sample} not established by the audit"
            )
        found.add(parent)
    if not found:
        raise UnresolvedGroupError(f"{base}: no recording identifier")
    if len(found) > 1:
        raise AmbiguousGroupError(f"{base}: points to several parents {sorted(found)}")
    return found.pop()


def find_group_crossings(
    split_of: Mapping[str, str], group_of: Mapping[str, str]
) -> dict[str, set[str]]:
    """Groups that appear in more than one split. Items missing a group raise."""
    seen: dict[str, set[str]] = defaultdict(set)
    for item, split in split_of.items():
        if item not in group_of:
            raise UnresolvedGroupError(f"{item}: no group assigned")
        seen[group_of[item]].add(split)
    return {g: s for g, s in seen.items() if len(s) > 1}


def assert_no_group_crossing(
    split_of: Mapping[str, str], group_of: Mapping[str, str]
) -> None:
    bad = find_group_crossings(split_of, group_of)
    if bad:
        raise PartitionLeakError(f"Groups spanning several splits: {bad}")


def find_duplicate_sources(
    sources: Iterable[tuple[str, str, str]],
) -> list[tuple[str, ...]]:
    """(item_id, group_id, sha256) triples -> byte-identical sources filed under >1 item.

    Identical bytes under different items are duplicates; if their groups differ the
    split can leak, so callers should fail on any returned entry with distinct groups.
    """
    by_hash: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for item, group, sha in sources:
        by_hash[sha].append((item, group))
    return [
        tuple(sorted({i for i, _ in v}))
        for v in by_hash.values()
        if len({i for i, _ in v}) > 1
    ]
