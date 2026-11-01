"""Determinism audit of the frozen ASR caches: re-transcribe a seeded sample in a FRESH process (OMP=4 before numpy, canary first)
and compare words/times/probs with the cache. Usage: python asr_audit.py N_DIHANA N_EPIC N_AMERESCO"""
import _pin_env  # noqa: F401  (must be first)
import json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from common import OUT, CACHE, WHISPER, manifest, write_cfg  # noqa
import numpy as np, soundfile as sf  # noqa
import asr_cache as AC  # noqa  (re-uses canary; importing does not transcribe)


def main():
    n = {"dihana": int(sys.argv[1]), "epic": int(sys.argv[2]), "ameresco": int(sys.argv[3])}
    from mrcd.extractors.linguistic import VerbatimASR
    asr = VerbatimASR(str(WHISPER), device="cpu", compute_type="int8")
    AC.canary(asr)
    rng = np.random.default_rng(13); res = {}
    for c, k in n.items():
        man = manifest(c); pick = [man[i] for i in sorted(rng.choice(len(man), min(k, len(man)), replace=False))]
        rows = []
        for m in pick:
            a, _ = sf.read(m["wav"], dtype="float32")
            ws = [w.to_dict() for w in asr.transcribe_chunked(a, chunk_s=8.0, overlap_s=2.0)]
            ref = json.loads((CACHE / c / f"{m['unit_id']}.json").read_text("utf8"))["words"]
            same_text = [w["text"] for w in ws] == [w["text"] for w in ref]
            exact = same_text and all(w["start"] == r["start"] and w["end"] == r["end"] and w["prob"] == r["prob"] for w, r in zip(ws, ref))
            rows.append({"unit": m["unit_id"], "same_text": same_text, "bit_identical": exact})
        res[c] = {"n": len(rows), "same_text": sum(r["same_text"] for r in rows), "bit_identical": sum(r["bit_identical"] for r in rows), "mismatches": [r["unit"] for r in rows if not r["bit_identical"]]}
        print(c, res[c], flush=True)
    (OUT / "asr_cache" / "determinism_audit.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
