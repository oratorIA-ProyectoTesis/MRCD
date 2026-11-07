"""Import this FIRST in every entry point: OMP_NUM_THREADS=4 must be in the environment before numpy is imported."""

import os
import sys
from pathlib import Path

if "numpy" in sys.modules:
    raise SystemExit("numpy was imported before the OMP pin: refusing to run (non-reproducible ASR text)")
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
CORPUS_ROOT = REPO / "data" / "external" / "CHM150" / "CHM150"
OUT = REPO / "results" / "external_chm150_20261102"
SEED = 13
WHISPER = (
    Path.home()
    / ".cache/huggingface/hub/models--Systran--faster-whisper-small/snapshots/536b0662742c02347bc0e980a01041f333bce120"
)
CKPT = REPO / "models" / "ckpt_audio_text.pt"
