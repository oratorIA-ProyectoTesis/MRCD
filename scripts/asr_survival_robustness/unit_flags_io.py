"""Corrected writers for the unit_flags CSVs of asr_survival_robustness_20261104.

Why this exists: the original writer in analyze_robustness.py put the 500 CHM150 rows (24 fields) under the
56-column DIHANA header. analyze_robustness.py is hash-pinned by the frozen ADDENDUM.sha256.json and by
metrics.json, so it is left byte-identical; this module is the corrected writer. CHM150 rows now live in their own
file with their own header. No decoding and no metric is recomputed here.

CLI: python unit_flags_io.py <results_dir>   (splits an existing mixed unit_flags.csv in place, no re-decoding)
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

DIHANA_FIELDS = 56  # corpus, unit_id, kind, rep_pos + 13 systems x (rep, rev, fill, rule)
CHM_SYSTEMS = ("small_P0", "small_P1", "small_pnone", "lv3_pnone", "cw_large")  # order of metrics.json chm150.summary
CHM_HEADER = ["corpus", "unit_id", "kind", "rep_pos"] + [f"{n}:{m}" for n in CHM_SYSTEMS for m in ("surv", "rule")]


def split_mixed_rows(rows: list[list[str]]) -> tuple[list[list[str]], list[list[str]]]:
    """Split the legacy mixed CSV (header first) into (dihana incl. header, chm150 rows with corrected header)."""
    header, body = rows[0], rows[1:]
    if len(header) != DIHANA_FIELDS:
        raise ValueError(f"unexpected header width {len(header)}")
    dih, chm = [header], [CHM_HEADER]
    for r in body:
        if r[0] == "dihana":
            if len(r) != DIHANA_FIELDS:
                raise ValueError(f"dihana row {r[1]} has {len(r)} fields")
            dih.append(r)
        elif r[0] == "chm150":
            if len(r) != 4 + 4 * len(CHM_SYSTEMS):
                raise ValueError(f"chm150 row {r[1]} has {len(r)} fields")
            # legacy layout per system: surv, "", "", rule  (the two empty cells carried no data)
            chm.append(r[:4] + [x for i in range(len(CHM_SYSTEMS)) for x in (r[4 + 4 * i], r[7 + 4 * i])])
        else:
            raise ValueError(f"unknown corpus {r[0]!r}")
    return dih, chm


def _write(path: Path, rows: list[list]) -> None:
    with path.open("w", newline="", encoding="utf8") as f:
        csv.writer(f).writerows(rows)


def write_unit_flags(outdir: Path, dihana_rows: list[list], chm_rows: list[list]) -> None:
    """Write unit_flags.csv (DIHANA only) and unit_flags_chm150.csv, each width-consistent with its header."""
    for name, rows in (("unit_flags.csv", dihana_rows), ("unit_flags_chm150.csv", chm_rows)):
        w = len(rows[0])
        bad = [r[1] for r in rows[1:] if len(r) != w]
        if bad:
            raise ValueError(f"{name}: {len(bad)} rows do not match the header width {w}")
        _write(Path(outdir) / name, rows)


def main(d: str) -> None:
    d = Path(d)
    with (d / "unit_flags.csv").open(newline="", encoding="utf8") as f:
        dih, chm = split_mixed_rows(list(csv.reader(f)))
    write_unit_flags(d, dih, chm)
    print(f"dihana rows {len(dih) - 1}, chm150 rows {len(chm) - 1}")


if __name__ == "__main__":
    main(sys.argv[1])
