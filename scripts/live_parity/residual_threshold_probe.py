"""Post-hoc diagnostic (declared as such): is the R needed to remove the v3 residual of seed 311 about the length of the
digital silence that follows the clip (15.05 s)? R = 14 (< silence) and R = 16 (> silence), same criterion as
``residual_vs_right_margin.py``. usage: python residual_threshold_probe.py --out X.jsonl"""

import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from residual_vs_right_margin import job  # noqa: E402

PROBES = ((311, 14.0), (311, 16.0))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    with Pool(len(PROBES)) as pool, open(a.out, "w", encoding="utf8") as fh:
        for r in pool.imap_unordered(job, PROBES):
            fh.write(json.dumps(r) + "\n")
            fh.flush()
            print(json.dumps(r), flush=True)
