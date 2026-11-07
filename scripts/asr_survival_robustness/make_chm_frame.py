"""Second-corpus sub-frame (results/asr_survival_robustness_20261104/ADDENDUM.md section 4).

Reads ONLY reference.json of results/external_chm150_20261102 (speaker, rep_literal). All 76 reference-positive utterances
plus 424 negatives drawn with random.Random(20261104) from the sorted negative ids. Writes chm_frame.json (ids only; no text,
no ASR output).
"""

import json
import random
from pathlib import Path

R = Path(__file__).resolve().parents[2]
ref = json.loads(
    (R / "results/external_chm150_20261102/reference.json").read_text("utf8")
)
pos = sorted(u for u, r in ref.items() if r["rep_literal"] == 1)
neg = sorted(u for u, r in ref.items() if r["rep_literal"] == 0)
assert (len(pos), len(neg)) == (76, 2587)
samp = sorted(random.Random(20261104).sample(neg, 424))
ids = sorted(pos + samp)
out = R / "results/asr_survival_robustness_20261104/chm_frame.json"
out.write_text(
    json.dumps(
        {
            "seed": 20261104,
            "n_pos": 76,
            "n_neg_sampled": 424,
            "ids": ids,
            "speakers": {u: ref[u]["speaker"] for u in ids},
        },
        indent=1,
    ),
    "utf8",
)
print(len(ids), len({ref[u]["speaker"] for u in ids}), "speakers")
