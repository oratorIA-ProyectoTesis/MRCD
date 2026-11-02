"""Fidelity check: unit features rebuilt from AUDIO by ``DisfluencyEngine.unit_features`` vs the stored campaign features.

Uses the cached ASR words of DIHANA units (no Whisper). Prints max abs differences of probs / pooled / pause and whether the
O-pipeline decision matches. Fold 5 is never used. Run from the repo root with OMP_NUM_THREADS=4.
"""

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "improve"))
from common import CACHE, manifest  # noqa: E402

from mrcd.engine import DisfluencyEngine  # noqa: E402
from mrcd.optimize import data as OD  # noqa: E402
from mrcd.service import decision_profile as DP  # noqa: E402


def main(n: int = 12) -> None:
    eng = DisfluencyEngine.load(ROOT / "models" / "ckpt_audio_text.pt", device="cpu")
    feats = OD.load_features()
    pipes = DP.o_pipelines()
    man = [m for m in manifest("dihana") if m["fold"] != 5 and m["unit_id"] in feats]
    rng = np.random.default_rng(13)
    pick = rng.choice(len(man), size=n, replace=False)
    worst = {"probs": 0.0, "pooled": 0.0, "pause": 0.0}
    agree = {c: 0 for c in DP.O_CLASSES}
    for i in pick:
        m = man[i]
        uid = m["unit_id"]
        audio, sr = sf.read(m["wav"], dtype="float32")
        words = json.loads((CACHE / "dihana" / f"{uid}.json").read_text("utf8"))["words"]
        f = eng.unit_features(audio, words)
        ref = feats[uid]
        worst["probs"] = max(worst["probs"], float(np.abs(f["probs"] - ref["probs"]).max()))
        worst["pooled"] = max(worst["pooled"], float(np.abs(f["pooled"] - ref["pooled"]).max()))
        worst["pause"] = max(worst["pause"], max(abs(f["pause"][k] - ref["pause"][k]) for k in ref["pause"]))
        txt = " ".join(w["text"] for w in words)
        base = {"text": {"p0": txt}, "words": {"p0": words}, "n_samples": len(audio), "acoustic": None}
        a = {**base, **{k: f[k] for k in ("probs", "pooled", "pause")}}
        b = {**base, **{k: ref[k] for k in ("probs", "pooled", "pause")}}
        for c in DP.O_CLASSES:
            agree[c] += int(pipes[c].predict([a])["decision"][0] == pipes[c].predict([b])["decision"][0])
    print(json.dumps({"n": n, "max_abs_diff": worst, "decision_agreement": agree}, indent=1))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 12)
