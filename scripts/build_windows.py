#!/usr/bin/env python3
"""Ensambla data/windows_<source>.npz a partir del manifest (etiquetas auto o humanas)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.constants import LABELS  # noqa: E402
from core.dataset import build_recording_windows  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(ROOT / "data/dataset_manifest.json"))
    ap.add_argument("--labels", choices=["auto", "human", "gold"], default="auto")
    ap.add_argument("--av-offset-ms", type=float, default=0.0, help="desfase A/V calibrado (video retrasado > 0)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    man = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    parts = [build_recording_windows(r, ROOT, args.labels, args.av_offset_ms) for r in man["recordings"]]
    parts = [p for p in parts if len(p["y"])]
    if not parts:
        sys.exit("Sin ventanas.")
    # alinear ancho de rasgos lingüísticos (por si unas grabaciones tienen RoBERTa y otras no)
    F = min(p["ling"].shape[2] for p in parts)
    data = {k: np.concatenate([p[k][..., :F] if k == "ling" else p[k] for p in parts]) for k in parts[0]}
    out = Path(args.out or ROOT / f"data/windows_{args.labels}.npz")
    src = {"auto": "auto", "human": "human", "gold": "gold_llm"}[args.labels]
    np.savez_compressed(out, label_source=np.array(src), labels=np.array(LABELS), **data)
    counts = {l: int((data["y"] == i).sum()) for i, l in enumerate(LABELS)}
    print(f"{out}: {len(data['y'])} ventanas | con video: {data['has_video'].mean():.0%} | por clase: {counts}")


if __name__ == "__main__":
    main()
