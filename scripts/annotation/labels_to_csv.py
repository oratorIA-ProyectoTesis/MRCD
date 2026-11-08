"""Convert Audacity label exports (one .txt per clip) to the annotation sheet CSV.

Audacity label line:  start<TAB>end<TAB>CODE state [note words]
  CODE  in FP, LF, REP, PRO, INAUD ; state in d, f, i (INAUD takes none)
Discourse-marker function layer (same passes): MK <token> <d|f|i> [note]
  token from the closed list in markers.py (este, o_sea, bueno, pues, a_ver, tipo).
  Emits class=discourse_marker, state=none, marker_token, marker_function.
Every clip in --clips-dir must have a label file (an empty file means
"looked, found no event"); a missing file is an error, never an implicit "none".
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from scripts.annotation.markers import MARKER_CLASS, check_marker

CLASSES = {
    "FP": "filled_pause",
    "LF": "lexical_filler",
    "REP": "repetition",
    "PRO": "prolongation",
    "INAUD": "inaudible",
}
STATES = {"d": "disfluent", "f": "functional", "i": "uncertain"}
HEADER = ["item_id", "annotator", "class", "state", "start_s", "end_s", "note", "marker_token", "marker_function"]


def parse_marker(text: str) -> tuple[str, str, str]:
    """`MK <token> <d|f|i> [note]` -> (token, function, note)."""
    parts = text.split()
    if len(parts) >= 3 and parts[2].lower() not in STATES:
        two = f"{parts[1]}_{parts[2]}".lower()
        if two in ("o_sea", "a_ver"):
            raise ValueError(
                f"write {two!r} with an underscore: 'MK {two} <d|f|i>', got {text!r}"
            )
    if len(parts) < 3 or parts[2].lower() not in STATES:
        raise ValueError(f"marker label needs 'MK <token> <d|f|i>', got {text!r}")
    token, function = check_marker(parts[1], STATES[parts[2].lower()])
    return token, function, " ".join(parts[3:])


def parse_label(text: str) -> tuple[str, str, str]:
    parts = text.split()
    code = parts[0].upper()
    if code not in CLASSES:
        raise ValueError(f"unknown class code {code!r}")
    if code == "INAUD":
        return CLASSES[code], "uncertain", " ".join(parts[1:])
    if len(parts) < 2 or parts[1].lower() not in STATES:
        raise ValueError(f"missing or invalid state in {text!r}")
    return CLASSES[code], STATES[parts[1].lower()], " ".join(parts[2:])


def read_events(path: Path) -> list[tuple[float, float, str]]:
    """Parse one Audacity label export. Errors name the clip id and line number."""
    events = []
    clip = path.stem
    for n, ln in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        if not ln.strip() or ln.startswith("\\"):  # blank or frequency-range line
            continue
        parts = ln.split("\t", 2)
        if len(parts) < 3 or not parts[2].strip():
            raise ValueError(f"{clip} line {n}: need start<TAB>end<TAB>label, got {ln!r}")
        try:
            s, e = float(parts[0]), float(parts[1])
        except ValueError:
            raise ValueError(f"{clip} line {n}: non-numeric times in {ln!r}") from None
        if e <= s:
            raise ValueError(f"{clip} line {n}: end <= start (point label?); select a region")
        events.append((s, e, parts[2]))
    return events


def convert(clips_dir: Path, labels_dir: Path, annotator: str) -> tuple[list[list], list[str]]:
    clips = sorted(p.stem for p in clips_dir.glob("*.wav"))
    rows = []
    for cid in clips:
        f = labels_dir / f"{cid}.txt"
        if not f.exists():
            raise ValueError(f"missing label file for {cid}")
        events = read_events(f)
        if not events:
            rows.append([cid, annotator, "none", "none", "", "", "", "", ""])
        for s, e, text in events:
            try:
                if text.split()[0].upper() == "MK":
                    tok, fn, note = parse_marker(text)
                    rows.append([cid, annotator, MARKER_CLASS, "none", f"{s:.3f}", f"{e:.3f}", note, tok, fn])
                    continue
                cls, state, note = parse_label(text)
            except ValueError as exc:
                raise ValueError(f"{cid}: {exc}") from None
            rows.append([cid, annotator, cls, state, f"{s:.3f}", f"{e:.3f}", note, "", ""])
    orphans = sorted(p.stem for p in labels_dir.glob("*.txt") if p.stem not in set(clips))
    return rows, orphans


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--clips-dir", type=Path, required=True)
    ap.add_argument("--labels-dir", type=Path, required=True)
    ap.add_argument("--annotator", required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    try:
        rows, orphans = convert(a.clips_dir, a.labels_dir, a.annotator)
    except ValueError as exc:
        raise SystemExit(str(exc)) from None
    for o in orphans:
        print(f"WARNING: label file {o}.txt matches no clip in {a.clips_dir}", file=sys.stderr)
    with a.out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(HEADER)
        w.writerows(rows)
    print(f"{len(rows)} rows -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
