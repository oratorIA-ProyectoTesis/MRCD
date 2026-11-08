"""Draw the random subset of items for rater 2 (seeded, label-independent).

Uses only item ids from the manifest, so the draw cannot depend on any label or
system output. The seed is required and must be declared before drawing.
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path


def draw(item_ids: list[str], n: int, seed: int) -> list[str]:
    if n > len(item_ids):
        raise ValueError(f"cannot draw {n} from {len(item_ids)} items")
    return sorted(random.Random(seed).sample(sorted(item_ids), n))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if a.n < 40:
        raise SystemExit(
            "rater 2 needs at least 40 items (annex 2026-11-08, control d)"
        )
    with a.manifest.open(encoding="utf-8", newline="") as fh:
        ids = [r["item_id"] for r in csv.DictReader(fh)]
    chosen = draw(ids, a.n, a.seed)
    with a.out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["item_id"])
        w.writerows([[i] for i in chosen])
    print(f"{len(chosen)} items -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
