"""Regenerate tests/data/audio_features_golden.json (run only when the feature recipe changes on purpose).

    OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 python scripts/hardening/make_features_golden.py
"""

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from mrcd.engine import DisfluencyEngine  # noqa: E402
from tests.golden_features_audio import WORDS, synthetic_clip  # noqa: E402

eng = DisfluencyEngine.load(ROOT / "models" / "ckpt_audio_text.pt", device="cpu")
f = eng.unit_features(synthetic_clip(), WORDS)
out = {
    "note": "synthetic clip (tests/golden_features_audio.py); regenerate only on purpose",
    "atol": 1e-4,
    "probs": f["probs"].tolist(),
    "pooled": f["pooled"].tolist(),
    "pause": {k: float(v) for k, v in f["pause"].items()},
    "n_samples": int(f["n_samples"]),
}
(ROOT / "tests" / "data" / "audio_features_golden.json").write_text(json.dumps(out, indent=1), "utf8")
print({k: (len(v) if isinstance(v, list) else v) for k, v in out.items()})
