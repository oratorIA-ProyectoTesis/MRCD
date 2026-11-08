"""Build a blinded presentation for one annotation pass.

Reads the committed manifest and, for a given pass seed, assigns every item a
fresh random blind id (so the same clip gets a different id in each pass) and a
random order. Outputs:

* key file (LOCAL, outside the repo): blind_id,orig_item_id,file,start_s,end_s,
  speaker_if_in_name. Its first columns match the manifest, so
  `cut_clips.py --manifest <key>` names clips by blind id.
* annotator-facing list: position,blind_id only (no file name, speaker, offsets
  or original id).

The key never enters the repository. Only unblind.py reads it, after the
annotation CSV has been hashed.
"""
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
KEY_FIELDS = ["item_id", "orig_item_id", "file", "sha256", "start_s", "end_s", "speaker_if_in_name"]


def inside_repo(path: Path) -> bool:
    p = path.resolve()
    return p == REPO or REPO in p.parents


def make_blinded(rows: list[dict], seed: int) -> tuple[list[dict], list[dict]]:
    rng = random.Random(seed)
    ids: set[str] = set()
    key = []
    for r in rows:
        while True:
            bid = "b" + "".join(rng.choice("0123456789abcdefghjkmnpqrstvwxyz") for _ in range(7))
            if bid not in ids:
                ids.add(bid)
                break
        key.append(
            {
                "item_id": bid,
                "orig_item_id": r["item_id"],
                "file": r["file"],
                "sha256": r["sha256"],
                "start_s": r["start_s"],
                "end_s": r["end_s"],
                "speaker_if_in_name": r["speaker_if_in_name"],
            }
        )
    rng.shuffle(key)
    listing = [{"position": i + 1, "blind_id": k["item_id"]} for i, k in enumerate(key)]
    return key, listing


def check_fresh(key: list[dict], old: list[dict]) -> str | None:
    """Return a message if the new pass reuses blind ids or the old presentation order."""
    if {k["item_id"] for k in key} & {k["item_id"] for k in old}:
        return "blind ids overlap with the previous pass"
    if [k["orig_item_id"] for k in key] == [k["orig_item_id"] for k in old]:
        return "presentation order is identical to the previous pass"
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--seed", type=int, required=True, help="use a different seed per pass")
    ap.add_argument("--key-out", type=Path, required=True, help="LOCAL path outside the repo")
    ap.add_argument("--list-out", type=Path, required=True)
    ap.add_argument("--prev-key", type=Path, action="append", default=[],
                    help="key file of an earlier pass (repeatable); refuses a reused order or ids")
    ap.add_argument("--only-ids", type=Path, help="CSV with an item_id column (e.g. rater-2 subset)")
    a = ap.parse_args()
    if inside_repo(a.key_out):
        raise SystemExit("refusing to write the key file inside the repository")
    with a.manifest.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    if a.only_ids:
        with a.only_ids.open(encoding="utf-8", newline="") as fh:
            keep = {r["item_id"] for r in csv.DictReader(fh)}
        rows = [r for r in rows if r["item_id"] in keep]
    key, listing = make_blinded(rows, a.seed)
    for prev in a.prev_key:
        with prev.open(encoding="utf-8", newline="") as fh:
            old = list(csv.DictReader(fh))
        problem = check_fresh(key, old)
        if problem:
            raise SystemExit(f"{problem} (previous key: {prev}); use a different seed")
    a.key_out.parent.mkdir(parents=True, exist_ok=True)
    with a.key_out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, KEY_FIELDS, lineterminator="\n")
        w.writeheader()
        w.writerows(key)
    a.list_out.parent.mkdir(parents=True, exist_ok=True)
    with a.list_out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, ["position", "blind_id"], lineterminator="\n")
        w.writeheader()
        w.writerows(listing)
    print(f"{len(key)} items blinded (seed {a.seed}); key -> {a.key_out}; list -> {a.list_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
