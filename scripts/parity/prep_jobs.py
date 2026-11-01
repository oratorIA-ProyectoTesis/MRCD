"""Build the job list: 32 pilot clips + one 40 s excerpt from data/clips (new derived WAV)."""
import os
import hashlib, json
from pathlib import Path
import numpy as np, soundfile as sf

REPO = Path(r"C:\zArnes-tesis\mrcd")
P = Path(os.environ.get("MRCD_PILOT_ROOT", r"<PILOT_ROOT>"))
OUT = REPO / "results" / "paridad_20261030"
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()

jobs = []
for row in json.loads((P / "input_manifest.json").read_text("utf8")):
    wav = P / row["input_wav"]
    assert sha(wav) == row["input_sha256"]
    jobs.append({"id": row["sample_id"], "wav": str(wav), "sha256": row["input_sha256"],
                 "a": row["target_start_s"], "b": row["target_end_s"], "kind": "pilot",
                 "pilot_words": str(P / "results" / f"{row['sample_id']}_asr.json")})
src = REPO / "data" / "clips" / "yD9Igp27VNE.wav"
audio, sr = sf.read(src, dtype="float32")
assert sr == 16000 and audio.ndim == 1
ex = OUT / "extra_audio" / "yD9Igp27VNE_30s_70s.wav"
sf.write(ex, audio[30 * sr:70 * sr], sr, subtype="PCM_16")
jobs.append({"id": "extra_yD9_30_70", "wav": str(ex), "sha256": sha(ex), "a": 5.0, "b": 35.0,
             "kind": "extra", "source_wav": str(src), "source_sha256": sha(src), "pilot_words": None})
(OUT / "jobs.json").write_text(json.dumps(jobs, indent=1), "utf8")
print(len(jobs))
