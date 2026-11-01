"""Shared paths for the S6 scripts. No EPIC text is ever written out: only ids, times, label and speaker."""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# Corpus location: MRCD_EPIC_DIR (full path) or MRCD_CORPORA_ROOT/16_EPIC_v2; fallback is relative to the repo.
EPIC = Path(
    os.environ.get("MRCD_EPIC_DIR")
    or Path(os.environ.get("MRCD_CORPORA_ROOT") or ROOT / "data" / "corpora") / "16_EPIC_v2"
)
OUT = ROOT / "results" / "mejora_objetivos_20261101" / "S6"
WORK = OUT / "work"  # large/per-frame intermediates: gitignored, never committed
WORK.mkdir(parents=True, exist_ok=True)
FACE_MODEL = str(ROOT / "models" / "face_landmarker.task")
WHISPER = (
    Path.home()
    / ".cache/huggingface/hub/models--Systran--faster-whisper-small/snapshots/536b0662742c02347bc0e980a01041f333bce120"
)


def video_paths():
    return {p.stem: p for p in sorted(EPIC.rglob("*.mp4"))}
