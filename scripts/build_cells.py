#!/usr/bin/env python3
"""Build additive v2 cells from explicit permitted local raw/evidence manifests."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.cell_dataset import build_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True, help="new directory; existing paths refused")
    args = parser.parse_args()
    result = build_dataset(args.manifest, args.out)
    print(f"built {len(result['recordings'])} {result['source_kind']} recordings; not scientific validation")


if __name__ == "__main__":
    main()
