#!/usr/bin/env python3
"""Remove the rejected duration_ratio column from persisted MRCD features."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DURATION_RATIO_INDEX = 10


def rollback(npz_path: Path) -> tuple[int, int]:
    with np.load(npz_path, allow_pickle=False) as source:
        arrays = {name: source[name] for name in source.files}
    old = arrays["word_feats"]
    if old.shape[1] <= DURATION_RATIO_INDEX:
        return old.shape[1], old.shape[1]
    arrays["word_feats"] = np.delete(old, DURATION_RATIO_INDEX, axis=1).astype(np.float32)
    temporary = npz_path.with_suffix(".rollback.tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(npz_path)
    return old.shape[1], arrays["word_feats"].shape[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=str(ROOT / "data/dataset_manifest.json"))
    parser.add_argument("--vid", default="Ka_okSSytes")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    names = manifest.get("schema", {}).get("word_feature_names", [])
    has_duration = "duration_ratio" in names
    changes = []
    if has_duration:
        for recording in manifest["recordings"]:
            path = ROOT / recording["features_npz"]
            changes.append(rollback(path))
        case_path = ROOT / f"data/case/{args.vid}.npz"
        case_change = rollback(case_path)
        schema = manifest["schema"]
        schema["version"] = "0.2.0"
        schema["word_feature_names"] = [name for name in names if name != "duration_ratio"]
        schema.pop("duration_ratio_source", None)
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        case_change = (10, 10)
    print(f"[rollback] corpus: {len(changes)} recordings · widths {sorted(set(changes))}")
    print(f"[rollback] case {args.vid}: {case_change[0]} -> {case_change[1]}")


if __name__ == "__main__":
    main()
