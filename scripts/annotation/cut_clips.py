"""Export manifest windows as local WAV clips (16 kHz mono) with ffmpeg.

Clips are named by item_id only (use the blinded key file from blind_session.py
so names are blind ids). The output folder MUST be outside the repository; clips
are never committed. Every source mp3 is checked against the manifest sha256
before anything is cut, and container metadata is stripped from the clips.
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import sys
from pathlib import Path

from scripts.annotation.sample_items import ROOT_ENV, sha256_file


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "--manifest", type=Path, required=True, help="manifest or blinded key CSV"
    )
    ap.add_argument(
        "--out-dir", type=Path, required=True, help="local folder OUTSIDE the repo"
    )
    ap.add_argument("--root", type=Path, default=os.environ.get(ROOT_ENV))
    a = ap.parse_args()
    if shutil.which("ffmpeg") is None:
        print("ffmpeg not found on PATH", file=sys.stderr)
        return 2
    if a.root is None:
        print(f"--root or ${ROOT_ENV} is required", file=sys.stderr)
        return 2
    repo = Path(__file__).resolve().parents[2]
    if repo in a.out_dir.resolve().parents or a.out_dir.resolve() == repo:
        print("refusing to write clips inside the repository", file=sys.stderr)
        return 2
    with a.manifest.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    checked: dict[str, str] = {}
    for r in rows:  # verify every source file before cutting anything
        if r["file"] not in checked:
            checked[r["file"]] = sha256_file(a.root / "audio" / r["file"])
        if checked[r["file"]] != r["sha256"]:
            print(f"sha256 mismatch for {r['file']}: aborting", file=sys.stderr)
            return 3
    a.out_dir.mkdir(parents=True, exist_ok=True)
    for r in rows:
        s, e = float(r["start_s"]), float(r["end_s"])
        dst = a.out_dir / f"{r['item_id']}.wav"
        cmd = [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-ss",
            f"{s:.1f}",
            "-t",
            f"{e - s:.1f}",
            "-i",
            str(a.root / "audio" / r["file"]),
            "-map_metadata",
            "-1",
            "-ac",
            "1",
            "-ar",
            "16000",
            str(dst),
        ]
        subprocess.run(cmd, check=True)
    print(f"{len(rows)} clips written to {a.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
