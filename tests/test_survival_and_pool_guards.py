"""Pins for REVISION4 surviving mutants M3b (ALLOWED_FOLDS includes 5) and M4 (repetition survival threshold)."""

from __future__ import annotations

import pytest

from mrcd.context_eval import pool
from mrcd.context_eval.survival import mark_survives

SPAN = "uno uno dos tres cuatro cinco seis siete ocho nueve"  # 10 tokens, adjacent duplicate at the start


def _mark(span=SPAN, mt=("uno",), ct=("dos", "tres")):
    return {"span": span, "mt": list(mt), "ct": list(ct), "subtype": "x"}


@pytest.mark.parametrize(
    "asr,expected",
    [
        (SPAN, True),  # 10/10
        ("uno uno dos tres cuatro cinco seis siete ocho", True),  # 9/10
        ("uno uno dos tres cuatro cinco seis siete", True),  # 8/10: the threshold is inclusive
        ("uno uno dos tres cuatro cinco seis", False),  # 7/10: below 0.8 (kills thresholds 0.5-0.7)
        ("uno uno dos tres cuatro cinco", False),  # 6/10
        ("uno uno dos tres cuatro", False),  # 5/10 (kills 0.5)
    ],
)
def test_repetition_survival_threshold_is_exactly_80_percent(asr, expected):
    assert mark_survives("repetition", _mark(), asr) is expected


def test_repetition_needs_an_adjacent_duplicate_inside_the_aligned_region():
    assert mark_survives("repetition", _mark(), "uno dos tres cuatro cinco seis siete ocho nueve") is False


@pytest.mark.parametrize(
    "asr,expected",
    [(SPAN, True), ("uno uno dos tres cuatro cinco seis siete", True), ("uno uno dos tres cuatro cinco seis", False)],
)
def test_revision_survival_threshold_is_exactly_80_percent_on_the_span_too(asr, expected):
    assert mark_survives("revision", _mark(mt=("uno", "uno"), ct=("dos", "tres")), asr) is expected


def test_allowed_folds_exclude_the_historical_fold():
    assert tuple(pool.ALLOWED_FOLDS) == (1, 2, 3, 4)
    assert pool.HISTORICAL_FOLD == 5 and pool.HISTORICAL_FOLD not in pool.ALLOWED_FOLDS


def test_build_pool_never_selects_fold_5_rows(tmp_path):
    """A CSV that only has fold-5 rows must yield an empty pool WITHOUT touching the corpus (fold 5 is not indexed)."""
    csv_path = tmp_path / "turns.csv"
    csv_path.write_text(
        "turn_id,fold,speaker,dialogue,ref_text_marks_stripped\n"
        "T1,5,S1,D1,hola\nT2,5,S2,D2,adios\n",
        encoding="utf-8",
    )
    assert pool.build_pool(tmp_path / "no_corpus", csv_path, tmp_path / "no_cache") == []


def test_assert_no_historical_still_raises():
    with pytest.raises(pool.HistoricalFoldError):
        pool.assert_no_historical([1, 5])


# ---- revision 6: pins for survival branches that the REVISION6 mutants N1-N3 left alive ----
def _rep_mark(span: str):
    return {"span": span, "mt": [], "ct": [], "subtype": "x"}


@pytest.mark.parametrize(
    "span,expected",
    [
        ("alfa alfa beta gamma delta", True),  # n = 1
        ("alfa beta alfa beta gamma delta", True),  # n = 2
        ("alfa beta gamma alfa beta gamma delta", True),  # n = 3 (mutant (1,2) loses it)
        ("alfa beta gamma delta alfa beta gamma delta epsilon", True),  # n = 4 (mutant (1,2) / (1,2,3) loses it)
        ("alfa beta gamma delta epsilon alfa beta gamma delta epsilon zeta", False),  # n = 5 is NOT counted
    ],
)
def test_adjacent_repetition_ngram_sizes_are_exactly_one_to_four(span, expected):
    assert mark_survives("repetition", _rep_mark(span), span) is expected


@pytest.mark.parametrize(
    "span,expected",
    [
        ("el hor horario de", True),  # prefix of 3 characters: a fragment restart
        ("el h horario de", True),  # prefix of 1
        ("el hora horario de", False),  # prefix of 4: not counted (kills a looser bound)
        ("el horario de", False),  # no prefix at all
    ],
)
def test_fragment_prefix_length_is_at_most_three_characters(span, expected):
    assert mark_survives("repetition", _rep_mark(span), span) is expected


_W = [f"{c}{c}x" for c in "bcdfghjklmnpqrstvwxz"]  # 20 distinct tokens


def _rev_mark():
    return {"span": " ".join(_W), "mt": [_W[0]], "ct": _W[10:15], "subtype": "x"}


@pytest.mark.parametrize(
    "dropped,expected",
    [
        ([], True),
        ([10], True),  # correction coverage 4/5 = 0.8: inclusive threshold; span coverage 19/20
        ([10, 11], False),  # correction coverage 3/5 = 0.6 while the span still covers 18/20 (kills 0.0-0.7)
        ([10, 11, 12, 13, 14], False),  # correction missing entirely
    ],
)
def test_revision_needs_the_correction_recovered_at_80_percent(dropped, expected):
    asr = " ".join(w for i, w in enumerate(_W) if i not in dropped)
    assert mark_survives("revision", _rev_mark(), asr) is expected
