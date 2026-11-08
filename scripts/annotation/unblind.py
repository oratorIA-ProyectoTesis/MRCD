"""Join a blinded annotation CSV back to original item ids.

Refuses to run unless `<annotations>.sha256` exists and matches the file, i.e.
the annotation CSV was hashed (and, per the annex, committed) BEFORE unblinding.
Output keeps the sheet schema with item_id = original manifest item id, so it can
be fed to agreement.py.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_hash(csv_path: Path) -> None:
    side = csv_path.with_name(csv_path.name + ".sha256")
    if not side.exists():
        raise SystemExit(f"missing {side.name}: hash the annotation CSV before unblinding")
    recorded = side.read_text(encoding="utf-8").split()[0].lower()
    if recorded != sha256_of(csv_path):
        raise SystemExit("sha256 mismatch: the annotation CSV changed after hashing")


def unblind(ann_rows: list[dict], key_rows: list[dict]) -> list[dict]:
    mapping = {k["item_id"]: k["orig_item_id"] for k in key_rows}
    out = []
    for r in ann_rows:
        if r["item_id"] not in mapping:
            raise SystemExit(f"unknown blind id {r['item_id']!r}")
        out.append({**r, "item_id": mapping[r["item_id"]]})
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--annotations", type=Path, required=True)
    ap.add_argument("--key", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--write-hash", action="store_true", help="write the .sha256 sidecar and exit")
    a = ap.parse_args()
    if a.write_hash:
        side = a.annotations.with_name(a.annotations.name + ".sha256")
        side.write_text(f"{sha256_of(a.annotations)}  {a.annotations.name}\n", encoding="utf-8")
        print(f"wrote {side}")
        return 0
    check_hash(a.annotations)
    with a.annotations.open(encoding="utf-8", newline="") as fh:
        rdr = csv.DictReader(fh)
        fields, ann = rdr.fieldnames, list(rdr)
    with a.key.open(encoding="utf-8", newline="") as fh:
        key = list(csv.DictReader(fh))
    rows = unblind(ann, key)
    with a.out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fields, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} rows unblinded -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
