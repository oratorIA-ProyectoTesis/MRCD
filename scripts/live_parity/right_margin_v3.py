"""v3 residual: does a larger RIGHT margin remove the partition-independent leftovers (seed 23: 438.35 s, seed 111: 321.27 s)?
usage: python right_margin_v3.py SEED R  -> prints one JSON line (tick partition, viterbi_carry=True, L=3)."""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import _pin_omp  # noqa: E402,F401

sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))
import numpy as np  # noqa: E402

from synth_session import build_session  # noqa: E402

from mrcd.extractors.acoustic import extract_acoustic  # noqa: E402
from mrcd.live.session_features import SessionFeatureTracker  # noqa: E402

seed, R = int(sys.argv[1]), float(sys.argv[2])
x = build_session(10.5, seed)[0]
o = extract_acoustic(x, use_silero=True).frame_matrix()
tr = SessionFeatureTracker(f0_center=0.0, left_margin_s=3.0, right_margin_s=R, viterbi_carry=True)
for i in range(0, len(x), 8000):
    tr.append(x[i : i + 8000])
c = tr.frame_matrix()
dc = np.nonzero(np.abs(c - o[: len(c)]).max(1) > 0)[0]
tr.finalize()
g = tr.frame_matrix()
d = np.nonzero(np.abs(g - o).max(1) > 0)[0]
print(json.dumps({"seed": seed, "R": R, "committed_bad": len(dc), "final_bad": len(d), "bad_s": (d * 0.01).round(2).tolist()[:6]}))
