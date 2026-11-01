"""AMERESCO prolongation (presence per annotated segment). Pool = ALL 98 eligible segments (3-8 s, no automatic exclusion reasons).
The 32 pilot segments are REUSED (same wav, same hash) as DIAGNOSTIC; the other 66 are built with the pilot recipe
(mono mp3 -> librosa 16 kHz, +-5 s real context). Partition unit = conversation (4). Nothing is tuned: all comparators are fixed.
Original label kept: <alargamiento/> presence (source_alargamiento_present); absence = not marked (coverage across teams unverified).
"""
import os
import csv, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np, soundfile as sf
from bl_common import CORPORA, OUT, write_config, sha256_file

P = Path(os.environ.get("MRCD_PILOT_ROOT", r"<PILOT_ROOT>"))
C = CORPORA / "12_AMERESCO"
D = OUT / "ameresco"; (D / "inputs").mkdir(parents=True, exist_ok=True)


def main():
    import librosa
    rows = list(csv.DictReader((C / "source_annotation_index.csv").open(encoding="utf-8-sig")))
    pool = [r for r in rows if not r["automatic_exclusion_reasons"] and 3000 <= int(r["end_ms"]) - int(r["start_ms"]) <= 8000]
    assert len(pool) == 98
    pil = {(x["recording_id"], x["source_annotation_id"]): x for x in json.loads((P / "input_manifest.json").read_text("utf8"))}
    cache = {}; man = []; recipe_check = []
    for r in sorted(pool, key=lambda r: (r["recording_id"], int(r["start_ms"]), r["source_annotation_id"])):
        key = (r["recording_id"], r["source_annotation_id"]); rid = r["recording_id"]
        start, end = int(r["start_ms"]) / 1000, int(r["end_ms"]) / 1000
        uid = f"{rid}_{r['source_annotation_id']}"
        if rid not in cache:
            path = next((C / rid).glob("*.mp3"))
            y, sr = sf.read(path, dtype="float32", always_2d=True); y = y.mean(axis=1)
            cache[rid] = librosa.resample(y, orig_sr=sr, target_sr=16000).astype(np.float32)
        y = cache[rid]
        cs = max(0, start - 5); ce = min(len(y) / 16000, end + 5)
        a, b = round(cs * 16000), round(ce * 16000)
        wav = D / "inputs" / f"{uid}.wav"
        sf.write(wav, y[a:b], 16000, subtype="PCM_16")
        in_pilot = key in pil
        if in_pilot:  # recipe reproducibility check vs the pilot's own wav
            recipe_check.append({"unit_id": uid, "same_sha256_as_pilot_wav": sha256_file(wav) == pil[key]["input_sha256"]})
        man.append({"unit_id": uid, "wav": str(wav), "wav_sha256": sha256_file(wav), "group": rid, "tier": r["source_tier"],
                    "target_start_s": start - a / 16000, "target_end_s": end - a / 16000, "in_pilot32": in_pilot,
                    "pilot_sample_id": pil[key]["sample_id"] if in_pilot else None,
                    "source_start_ms": int(r["start_ms"]), "source_end_ms": int(r["end_ms"])})
    refs = [{"unit_id": m["unit_id"], "group": m["group"], "in_pilot32": m["in_pilot32"], "source_label": "alargamiento",
             "prolongation_present": int(next(r for r in pool if f"{r['recording_id']}_{r['source_annotation_id']}" == m["unit_id"])["source_alargamiento_present"])}
            for m in man]
    (D / "manifest.json").write_text(json.dumps(man, indent=1))
    with (D / "references.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(refs[0])); w.writeheader(); w.writerows(refs)
    write_config(D / "prepare_run", "ameresco_prepare", __file__, {"pool": 98, "recipe_check_vs_pilot_wavs": recipe_check,
                 "unit": "ELAN speech interval 3-8 s", "partition": "conversation"},
                 {"source_annotation_index.csv": sha256_file(C / "source_annotation_index.csv"), "manifest.json": sha256_file(D / "manifest.json")})
    print(len(man), sum(r["prolongation_present"] for r in refs), "recipe identical to pilot wav:", sum(x["same_sha256_as_pilot_wav"] for x in recipe_check), "/", len(recipe_check))


main()
