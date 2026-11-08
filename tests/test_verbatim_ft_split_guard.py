"""Plan B split guard (scripts/verbatim_ft/dihana_io.assert_split_manifest). Synthetic manifests only."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "verbatim_ft"))
import dihana_io as dio  # noqa: E402


def _manifest():
    return {
        "train": {"speakers": ["a", "b"], "turn_ids": ["t1", "t2"]},
        "dev": {"speakers": ["c"], "turn_ids": ["t3"]},
        "val": {"speakers": ["d"], "turn_ids": ["t4"]},
    }


FOLDS = {"t1": 1, "t2": 2, "t3": 3, "t4": 4}


def test_valid_split_passes():
    dio.assert_split_manifest(_manifest(), FOLDS)


def test_overlapping_speakers_are_rejected():
    m = _manifest()
    m["val"]["speakers"] = ["d", "a"]
    with pytest.raises(ValueError, match="speakers shared by train and val"):
        dio.assert_split_manifest(m, FOLDS)


def test_turn_outside_folds_1_to_4_is_rejected():
    with pytest.raises(ValueError):
        dio.assert_split_manifest(_manifest(), {**FOLDS, "t4": 5})
    with pytest.raises(ValueError, match="not in folds"):
        dio.assert_split_manifest(_manifest(), {k: v for k, v in FOLDS.items() if k != "t2"})
