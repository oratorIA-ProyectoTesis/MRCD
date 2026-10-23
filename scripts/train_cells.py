#!/usr/bin/env python3
"""Train v2 masked heads on explicit persisted train rows; no product promotion."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.training_v2 import train_cells


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out", required=True, help="new checkpoint; no overwrite")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--d-model", type=int, default=32)
    parser.add_argument("--mode", choices=("audio_only", "audio_text", "trimodal"), default="audio_text")
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()
    print(json.dumps(train_cells(args.dataset, args.out, epochs=args.epochs, batch_size=args.batch_size,
                                d_model=args.d_model, mode=args.mode, seed=args.seed), allow_nan=False))


if __name__ == "__main__":
    main()
