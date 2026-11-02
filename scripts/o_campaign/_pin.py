"""Import this FIRST in every entry point: OMP_NUM_THREADS=4 must be in the environment before numpy is imported."""

import os
import sys

NUMPY_ALREADY_IMPORTED = "numpy" in sys.modules
os.environ["OMP_NUM_THREADS"] = "4"
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
if NUMPY_ALREADY_IMPORTED:
    raise SystemExit("numpy was imported before the OMP pin: refusing to run (non-reproducible ASR text)")
sys.path.insert(0, r"<repo>")
