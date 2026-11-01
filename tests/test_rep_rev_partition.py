"""Speaker-grouped partitions: no group crosses folds; the historical fold is refused."""

import numpy as np
import pytest

from mrcd.context_eval.partition import (
    GroupCrossingError,
    assert_no_crossing,
    find_crossings,
    group_folds,
    inner_splits,
    outer_splits,
)
from mrcd.context_eval.pool import HistoricalFoldError, assert_no_historical


def _groups(n_groups=23, per=7):
    return [f"S{g:02d}" for g in range(n_groups) for _ in range(per)]


def test_every_group_lands_in_exactly_one_fold_and_folds_are_used():
    g = _groups()
    f = group_folds(g, 5)
    assert find_crossings(g, f) == []
    assert set(f.tolist()) == set(range(5))
    sizes = np.bincount(f)
    assert (
        sizes.max() - sizes.min() <= 2 * 7
    )  # groups are dealt round-robin: at most 2 groups apart


def test_partition_is_deterministic_and_seed_dependent():
    g = _groups()
    assert (group_folds(g, 5, 13) == group_folds(g, 5, 13)).all()
    assert not (group_folds(g, 5, 13) == group_folds(g, 5, 14)).all()


def test_order_of_units_does_not_change_the_fold_of_a_group():
    g = _groups(10, 3)
    f = dict(zip(g, group_folds(g, 3)))
    rev = list(reversed(g))
    f2 = dict(zip(rev, group_folds(rev, 3)))
    assert f == f2


def test_crossing_detector_flags_a_leaking_assignment():
    g = ["a", "a", "b", "b"]
    assert find_crossings(g, [0, 1, 1, 1]) == ["a"]
    with pytest.raises(GroupCrossingError, match="speaker"):
        assert_no_crossing(g, [0, 1, 1, 1], "speaker")


def test_outer_and_inner_splits_never_share_groups_and_cover_all_units_once():
    g = np.array(_groups())
    seen = np.zeros(len(g), int)
    for _, tr, te in outer_splits(g, 5):
        assert not (set(g[tr]) & set(g[te]))
        seen[te] += 1
        for _, itr, ite in inner_splits(g[tr], 3):
            assert not (set(g[tr][itr]) & set(g[tr][ite]))
    assert (seen == 1).all()


def test_dialogues_nested_in_speakers_cannot_cross_when_speakers_do_not():
    spk = ["s1", "s1", "s2", "s2", "s3", "s3"]
    dlg = ["d1", "d2", "d3", "d3", "d4", "d5"]  # each dialogue belongs to one speaker
    f = group_folds(spk, 3)
    assert_no_crossing(dlg, f, "dialogue")


def test_fewer_groups_than_folds_is_an_error():
    with pytest.raises(ValueError):
        group_folds(["a", "a", "b"], 5)


def test_fold_5_is_refused():
    assert_no_historical([1, 2, 3, 4])
    with pytest.raises(HistoricalFoldError):
        assert_no_historical([1, 4, 5])
