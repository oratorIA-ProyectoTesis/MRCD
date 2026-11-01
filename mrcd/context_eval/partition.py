"""Speaker-grouped partitions (outer / inner) with explicit no-crossing checks."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

import numpy as np


class GroupCrossingError(ValueError):
    """A group (speaker / dialogue) appears in more than one fold."""


def group_folds(groups: Sequence[str], k: int, seed: int = 13) -> np.ndarray:
    """Fold id per item; all items of a group share one fold. Deterministic (sha256 order of groups)."""
    g = [str(x) for x in groups]
    if k < 2:
        raise ValueError("k must be >= 2")
    ug = sorted(
        set(g), key=lambda x: hashlib.sha256(f"{seed}|{x}".encode()).hexdigest()
    )
    if len(ug) < k:
        raise ValueError(f"{len(ug)} groups cannot fill {k} folds")
    fold_of = {x: i % k for i, x in enumerate(ug)}
    return np.array([fold_of[x] for x in g], dtype=int)


def find_crossings(labels: Sequence[str], folds: Sequence[int]) -> list[str]:
    """Group names that appear in more than one fold. ``labels`` is aligned with ``folds``."""
    seen: dict[str, set[int]] = {}
    for lab, f in zip(labels, folds):
        seen.setdefault(str(lab), set()).add(int(f))
    return sorted(k for k, v in seen.items() if len(v) > 1)


def assert_no_crossing(
    labels: Sequence[str], folds: Sequence[int], what: str = "group"
) -> None:
    bad = find_crossings(labels, folds)
    if bad:
        raise GroupCrossingError(f"{what} crosses folds: {bad[:5]} (total {len(bad)})")


def outer_splits(groups: Sequence[str], k: int = 5, seed: int = 13):
    """Yield (fold, train_idx, test_idx). Train and test never share a group."""
    f = group_folds(groups, k, seed)
    g = np.asarray([str(x) for x in groups])
    for i in range(k):
        te = np.flatnonzero(f == i)
        tr = np.flatnonzero(f != i)
        if set(g[te]) & set(g[tr]):
            raise GroupCrossingError("train/test share a group")
        yield i, tr, te


def inner_splits(groups_train: Sequence[str], k: int = 3, seed: int = 13):
    """Same as outer_splits on the training subset (indices are local to that subset)."""
    yield from outer_splits(groups_train, k, seed + 1)
