#!/usr/bin/env python3
"""Pointer to the frozen report for the rejected audio_text v2 experiment."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    report = ROOT / "results/audio_text_v2_comparison.md"
    print("[rejected] audio_text_v2 is not an active inference path.")
    print(f"[rejected] frozen report -> {report}")


if __name__ == "__main__":
    main()
