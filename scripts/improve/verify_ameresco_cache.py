"""Check that the frozen AMERESCO ASR cache (OMP=4) reproduces the pilot's cached words for the 32 pilot clips."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from common import OUT, CACHE, manifest, write_cfg  # noqa
PIL = Path(r"C:\zArnes-tesis\mrcd\results\reproduccion_piloto_20261030")
rows, same_text, same_all, maxd = [], 0, 0, 0.0
for m in manifest("ameresco"):
    if not m.get("in_pilot32"): continue
    a = json.loads((CACHE / "ameresco" / f"{m['unit_id']}.json").read_text("utf8"))["words"]
    b = json.loads((PIL / f"{m['pilot_sample_id']}_asr.json").read_text("utf8"))
    t = [w["text"] for w in a] == [w["text"] for w in b]
    d = max([abs(x[k] - y[k]) for x, y in zip(a, b) for k in ("start", "end", "prob")] or [0.0]) if len(a) == len(b) else float("inf")
    same_text += t; same_all += (t and d == 0.0); maxd = max(maxd, d) if d != float("inf") else maxd
    rows.append({"unit": m["unit_id"], "pilot_id": m["pilot_sample_id"], "same_text": t, "max_abs_diff_time_prob": d, "n_words": [len(a), len(b)]})
res = {"n": len(rows), "identical_text": same_text, "bit_identical_words_times_probs": same_all, "max_diff_when_same_length": maxd, "rows": rows}
(OUT / "asr_cache" / "ameresco_vs_pilot.json").write_text(json.dumps(res, indent=1))
print({k: v for k, v in res.items() if k != "rows"}); [print(r) for r in rows if not r["same_text"]]
