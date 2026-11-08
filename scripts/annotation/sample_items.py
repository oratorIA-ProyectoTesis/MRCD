"""Draw a seeded, time-stratified sample of unlabeled windows from NOCANDO audio.

Hard rules (thesis integrity):
* Sampling depends ONLY on file names and durations. No ASR, detector or system
  output is ever consulted, so the sample cannot be steered by the system.
* Every recording that has a TextGrid is excluded by stem (spent confirmatory cell).
  TextGrid files are listed by name only and never opened.
* Audio is only read for header info (soundfile.info); nothing is decoded.
* Output holds file name, sha256, offsets and opaque ids only (no audio, no text).

Design: recordings get a number of items proportional to duration (largest
remainder). Inside a recording, the timeline is cut into k equal strata and one
window (length uniform in [min_len, max_len], 0.1 s grid) is placed uniformly at
random inside each stratum, so windows never overlap and time is covered evenly.
"""

from __future__ import annotations

import argparse
import os
import csv
import hashlib
import random
import re
from pathlib import Path

import soundfile as sf

ROOT_ENV = "MRCD_NOCANDO_ROOT"  # path to the 02_NOCANDO_ES folder (with audio/ and alineamientos/)
FIELDS = ["item_id", "file", "sha256", "start_s", "end_s", "speaker_if_in_name"]


def speaker_from_name(stem: str) -> str:
    m = re.match(r"^(.*?)_es_\d+$", stem)
    return m.group(1) if m else ""


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def list_candidates(root: Path) -> tuple[list[Path], list[str]]:
    """Return (mp3 without TextGrid, sorted names of excluded stems)."""
    grids = {p.stem for p in (root / "alineamientos").glob("*.TextGrid")}
    mp3s = sorted((root / "audio").glob("*.mp3"))
    kept = [p for p in mp3s if p.stem not in grids]
    excluded = sorted(p.stem for p in mp3s if p.stem in grids)
    return kept, excluded


def allocate(durations: dict[str, float], total: int) -> dict[str, int]:
    """Largest-remainder allocation of `total` items proportional to duration."""
    s = sum(durations.values())
    raw = {k: total * v / s for k, v in durations.items()}
    out = {k: int(v) for k, v in raw.items()}
    left = total - sum(out.values())
    for k in sorted(raw, key=lambda k: (raw[k] - out[k], k), reverse=True)[:left]:
        out[k] += 1
    return out


def windows_for(rng: random.Random, dur: float, n: int, lo: float, hi: float):
    n = min(n, int(dur // hi))  # strata must be able to hold a max-length window
    if n <= 0:
        return []
    width = dur / n
    res = []
    for i in range(n):
        length = round(rng.uniform(lo, hi), 1)
        a, b = i * width, (i + 1) * width - length
        start = round(rng.uniform(a, max(a, b)), 1)
        res.append((start, round(start + length, 1)))
    return res


def build(
    root: Path, total: int, lo: float, hi: float, seed: int, exclude_speakers: tuple[str, ...] = ()
):
    files, excluded = list_candidates(root)
    files = [p for p in files if speaker_from_name(p.stem) not in exclude_speakers]
    durs = {p.name: sf.info(str(p)).duration for p in files}
    alloc = allocate(durs, total)
    rng = random.Random(seed)
    rows = []
    by_name = {p.name: p for p in files}
    for name in sorted(durs):
        sha = sha256_file(by_name[name])
        for s, e in windows_for(rng, durs[name], alloc[name], lo, hi):
            iid = "it_" + hashlib.sha1(f"{seed}|{name}|{s}".encode()).hexdigest()[:8]
            rows.append(
                {
                    "item_id": iid,
                    "file": name,
                    "sha256": sha,
                    "start_s": f"{s:.1f}",
                    "end_s": f"{e:.1f}",
                    "speaker_if_in_name": speaker_from_name(Path(name).stem),
                }
            )
    rng.shuffle(rows)  # presentation order is random, not chronological
    return rows, excluded, durs


def calibration(rows, durs, n, lo, hi, seed):
    """Draw n extra windows that do not overlap any main item (calibration set).

    Recordings are visited round-robin in a seeded shuffled order; windows are
    rejection-sampled uniformly in time. Still independent of any system output.
    """
    rng = random.Random(seed + 1)
    taken = {}
    for r in rows:
        taken.setdefault(r["file"], []).append((float(r["start_s"]), float(r["end_s"])))
    shas = {r["file"]: r["sha256"] for r in rows}
    names = sorted(durs)
    rng.shuffle(names)
    out, i = [], 0
    while len(out) < n:
        name = names[i % len(names)]
        i += 1
        for _ in range(200):
            length = round(rng.uniform(lo, hi), 1)
            st = round(rng.uniform(0, durs[name] - length), 1)
            if all(st + length <= a or st >= b for a, b in taken[name]):
                taken[name].append((st, st + length))
                iid = (
                    "ca_"
                    + hashlib.sha1(f"{seed}|cal|{name}|{st}".encode()).hexdigest()[:8]
                )
                out.append(
                    {
                        "item_id": iid,
                        "file": name,
                        "sha256": shas[name],
                        "start_s": f"{st:.1f}",
                        "end_s": f"{st + length:.1f}",
                        "speaker_if_in_name": speaker_from_name(Path(name).stem),
                    }
                )
                break
    return out


def write(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, FIELDS, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", type=Path, default=os.environ.get(ROOT_ENV),
                    help=f"02_NOCANDO_ES folder; default: ${ROOT_ENV}")
    ap.add_argument("--exclude-speaker", action="append", default=[],
                    help="speaker label to drop (repeatable), e.g. maria")
    ap.add_argument("--n-items", type=int, default=150)
    ap.add_argument("--min-len", type=float, default=6.0)
    ap.add_argument("--max-len", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=20261108)
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("docs/annotation_20261108/manifest_nocando_unlabeled.csv"),
    )
    ap.add_argument("--n-calibration", type=int, default=20)
    ap.add_argument(
        "--calibration-out",
        type=Path,
        default=Path("docs/annotation_20261108/manifest_nocando_calibration.csv"),
    )
    a = ap.parse_args()
    if a.root is None:
        ap.error(f"--root or ${ROOT_ENV} is required")
    rows, excluded, durs = build(
        a.root, a.n_items, a.min_len, a.max_len, a.seed, tuple(a.exclude_speaker)
    )
    write(a.out, rows)
    if a.n_calibration:
        cal = calibration(rows, durs, a.n_calibration, a.min_len, a.max_len, a.seed)
        write(a.calibration_out, cal)
        print(f"calibration items: {len(cal)} -> {a.calibration_out}")
    spk: dict[str, int] = {}
    for r in rows:
        spk[r["speaker_if_in_name"]] = spk.get(r["speaker_if_in_name"], 0) + 1
    print("items per speaker:", dict(sorted(spk.items())))
    print(f"excluded (TextGrid, {len(excluded)}): {', '.join(excluded)}")
    print(f"recordings used: {len(durs)}, total {sum(durs.values()) / 60:.1f} min")
    print(f"items written: {len(rows)} -> {a.out}")


if __name__ == "__main__":
    main()
