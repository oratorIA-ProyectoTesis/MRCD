#!/usr/bin/env python
"""Verify EVIDENCIAS.md: every listed path exists and every sha256 matches.

Exit code 0 only when every checked row passes; 1 on any failure.

Rows are the markdown table lines whose first cell is an id like ``E01``:

    | E01 | area | claim | `path` | `sha256` | status | review doc | date | versioned |

* ``versioned`` = ``yes``: the file is tracked by git; a missing file or a hash mismatch fails.
* ``versioned`` = ``no``: the file exists only on the author's disk (``results/`` is git-ignored for most
  files). It is checked when present; when absent it is reported as SKIPPED and does not fail, unless
  ``--strict-all`` is given. Use ``--root <other checkout>`` to check those rows against a checkout that has them.

Hashing: text files are hashed after CRLF -> LF normalisation (Windows ``autocrlf`` makes working-tree bytes
differ from the committed blob); the raw-bytes hash is accepted too. Binary files are hashed as they are.

Usage: python scripts/verify_evidence_index.py [--root DIR] [--index FILE] [--strict-all] [-q]
"""

from __future__ import annotations

import argparse
import hashlib
import re
import subprocess
import sys
from pathlib import Path

CRLF, LF = bytes([13, 10]), bytes([10])
TEXT_SUFFIXES = {
    ".md",
    ".json",
    ".csv",
    ".txt",
    ".log",
    ".py",
    ".tex",
    ".mmd",
    ".svg",
    ".html",
    ".sha256",
    ".yml",
    ".yaml",
    ".toml",
}
STATUSES = {"self", "independent", "none"}
ROW = re.compile(r"^\|\s*(E\d+)\s*\|")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def file_hashes(path: Path) -> set[str]:
    data = path.read_bytes()
    out = {hashlib.sha256(data).hexdigest()}
    if path.suffix.lower() in TEXT_SUFFIXES:
        out.add(hashlib.sha256(data.replace(b"\r\n", b"\n")).hexdigest())
    return out


def _normalised(path: Path) -> str:
    data = path.read_bytes()
    if path.suffix.lower() in TEXT_SUFFIXES:
        data = data.replace(CRLF, LF)
    return hashlib.sha256(data).hexdigest()


def is_tracked(root: Path, rel: str) -> bool | None:
    try:
        r = subprocess.run(
            ["git", "-C", str(root), "ls-files", "--error-unmatch", "--", rel],
            capture_output=True,
        )
    except OSError:
        return None
    return r.returncode == 0


def parse_rows(text: str) -> list[dict]:
    rows = []
    for ln, line in enumerate(text.splitlines(), 1):
        if not ROW.match(line):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        rows.append({"line": ln, "cells": cells})
    return rows


def verify(
    root: Path, index: Path, strict_all: bool = False, quiet: bool = False
) -> int:
    failures: list[str] = []
    skipped = 0
    checked = 0
    seen_ids: set[str] = set()
    rows = parse_rows(index.read_text(encoding="utf-8"))
    if not rows:
        failures.append(f"{index}: no evidence rows found")
    for r in rows:
        c, ln = r["cells"], r["line"]
        if len(c) != 9:
            failures.append(f"line {ln}: expected 9 cells, got {len(c)}")
            continue
        rid, _area, _claim, path_c, hash_c, status, _doc, date, versioned = c
        if rid in seen_ids:
            failures.append(f"line {ln}: duplicate id {rid}")
        seen_ids.add(rid)
        rel = path_c.strip("`")
        want = hash_c.strip("`").lower()
        if not HEX64.match(want):
            failures.append(f"{rid} line {ln}: sha256 is not 64 hex chars")
            continue
        if status not in STATUSES:
            failures.append(
                f"{rid} line {ln}: review status {status!r} not in {sorted(STATUSES)}"
            )
        if not DATE.match(date):
            failures.append(f"{rid} line {ln}: date {date!r} is not YYYY-MM-DD")
        if versioned not in {"yes", "no"}:
            failures.append(f"{rid} line {ln}: versioned must be yes or no")
            continue
        p = root / rel
        if not p.is_file():
            if versioned == "yes" or strict_all:
                failures.append(f"{rid}: missing file {rel}")
            else:
                skipped += 1
                if not quiet:
                    print(f"SKIPPED {rid} (not versioned, absent here): {rel}")
            continue
        checked += 1
        if want not in file_hashes(p):
            failures.append(f"{rid}: sha256 mismatch for {rel}")
        if versioned == "yes":
            t = is_tracked(root, rel)
            if t is False:
                failures.append(
                    f"{rid}: marked versioned=yes but {rel} is not tracked by git"
                )
    for f in failures:
        print(f"FAIL {f}", file=sys.stderr)
    if not quiet:
        print(
            f"{len(rows)} rows, {checked} checked, {skipped} skipped, {len(failures)} failures"
        )
    return 1 if failures else 0


def main(argv: list[str] | None = None) -> int:
    here = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--root",
        type=Path,
        default=here,
        help="checkout that holds the evidence files (default: this repo)",
    )
    ap.add_argument(
        "--index",
        type=Path,
        default=None,
        help="index file (default: <repo>/EVIDENCIAS.md)",
    )
    ap.add_argument(
        "--strict-all",
        action="store_true",
        help="fail also on absent non-versioned files",
    )
    ap.add_argument("-q", "--quiet", action="store_true")
    ap.add_argument("--print-hash", type=Path, metavar="FILE", help="print the normalised sha256 of FILE and exit")
    a = ap.parse_args(argv)
    if a.print_hash:
        print(_normalised(a.print_hash))
        return 0
    return verify(
        a.root.resolve(),
        (a.index or here / "EVIDENCIAS.md").resolve(),
        a.strict_all,
        a.quiet,
    )


if __name__ == "__main__":
    sys.exit(main())
