"""Pins the corrected unit_flags writers (REVISION independent M-3: CHM150 rows sat under the 56-column DIHANA header)."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "asr_survival_robustness"))
import unit_flags_io as U  # noqa: E402

HEADER = ["corpus", "unit_id", "kind", "rep_pos"] + [f"s{i}:{m}" for i in range(13) for m in ("rep", "rev", "fill", "rule")]
DIH = ["dihana", "u1", "pos", "1"] + ["1"] * 52
# five systems, legacy layout (surv, "", "", rule); surv=system index, rule=10+index so positions are distinguishable
CHM = ["chm150", "c1", "pos", "1"] + [x for i in range(5) for x in (str(i), "", "", str(10 + i))]


def test_split_gives_width_consistent_files():
    dih, chm = U.split_mixed_rows([HEADER, DIH, CHM])
    assert all(len(r) == len(dih[0]) for r in dih)
    assert all(len(r) == len(chm[0]) for r in chm)
    assert len(chm[0]) == 14


def test_chm_values_land_under_their_own_system_by_header():
    _, chm = U.split_mixed_rows([HEADER, CHM])
    row = dict(zip(chm[0], chm[1]))
    assert row["small_P0:surv"] == "0" and row["small_P0:rule"] == "10"
    assert row["lv3_pnone:surv"] == "3" and row["cw_large:surv"] == "4" and row["cw_large:rule"] == "14"


def test_write_roundtrip_and_malformed_rows_are_refused(tmp_path):
    dih, chm = U.split_mixed_rows([HEADER, DIH, CHM])
    U.write_unit_flags(tmp_path, dih, chm)
    for n in ("unit_flags.csv", "unit_flags_chm150.csv"):
        rows = list(csv.reader((tmp_path / n).open(encoding="utf8")))
        assert len({len(r) for r in rows}) == 1
    with pytest.raises(ValueError):
        U.write_unit_flags(tmp_path, [HEADER, CHM], chm)  # 24 fields under a 56-wide header: the original bug
