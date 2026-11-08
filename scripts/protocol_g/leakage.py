"""Leakage check (protocol G section 14): the 19 file stems and the 19 transcript sha256 digests must not appear in any
tracked file outside the allow-list, nor in ANY file (tracked, ignored, caches) under ``results/`` outside it.

The corpus name is deliberately not a search term. Token match for the stems: a neighbour must not be ``[A-Za-z0-9]``
(an underscore neighbour DOES match, so ``<stem>_ts.txt`` and ``<stem>_pass1`` are leaks); digests are matched
case-insensitively. The relative path of every non-allowed file (its file name included) is scanned with the same terms.
Files are streamed in chunks, so large binaries are fine.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from scripts.protocol_g import common as C

CHUNK = 1 << 24


def _allowed(rel: str, allow) -> bool:
    return any(rel.startswith(a) for a in allow)


def compile_terms(stems: list[str], digests: list[str]):
    parts = []
    if stems:
        alt = b"|".join(
            re.escape(s.encode()) for s in sorted(set(stems), key=len, reverse=True)
        )
        parts.append(re.compile(rb"(?<![A-Za-z0-9])(?:" + alt + rb")(?![A-Za-z0-9])"))
    if digests:
        alt = b"|".join(re.escape(d.lower().encode()) for d in sorted(set(digests)))
        parts.append(re.compile(rb"(?:" + alt + rb")", re.I))
    return parts


def scan_file(path: Path, patterns) -> list[str]:
    hits: list[str] = []
    overlap = 256
    try:
        with path.open("rb") as f:
            tail = b""
            while True:
                blk = f.read(CHUNK)
                if not blk:
                    break
                buf = tail + blk
                for pat in patterns:
                    for m in pat.finditer(buf):
                        hits.append(m.group(0).decode("latin-1"))
                tail = buf[-overlap:]
    except OSError:
        return []
    return sorted(set(hits))


def leakage_check(
    repo: Path, stems: list[str], digests: list[str], allow=C.ALLOW_LIST
) -> dict:
    repo = Path(repo)
    patterns = compile_terms(stems, digests)
    files: set[str] = set()
    for f in C.H.git("ls-files", "-z", cwd=repo).split("\0"):
        if f:
            files.add(f)
    res = repo / "results"
    if res.is_dir():
        for root, _dirs, names in os.walk(res):
            for n in names:
                files.add((Path(root) / n).relative_to(repo).as_posix())
    hits, scanned = [], 0
    for rel in sorted(files):
        if _allowed(rel, allow):
            continue
        p = repo / rel
        if not p.is_file():
            continue
        scanned += 1
        found = sorted(
            set(scan_file(p, patterns))
            | {
                m.group(0).decode("latin-1")
                for pat in patterns
                for m in pat.finditer(rel.encode("utf8"))
            }
        )
        if found:
            hits.append({"path": rel, "terms": found})
    return {
        "passed": not hits,
        "n_files_scanned": scanned,
        "n_terms": len(stems) + len(digests),
        "hits": hits,
    }
