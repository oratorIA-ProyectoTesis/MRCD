"""Run frozen faster-whisper-small (CPU int8, transcribe_chunked chunk_s=8) with CURRENT mrcd; save words.
usage: asr_run.py <outdir> [--only-extra]"""
import os, sys, json, time
os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ["OMP_NUM_THREADS"] = "4"
from pathlib import Path
import numpy as np, soundfile as sf
sys.path.insert(0, r"C:\zArnes-tesis\mrcd")
from mrcd.extractors.linguistic import VerbatimASR
OUT = Path(r"C:\zArnes-tesis\mrcd\results\paridad_20261030")
outdir = Path(sys.argv[1]); outdir.mkdir(exist_ok=True)
jobs = json.loads((OUT / "jobs.json").read_text("utf8"))
if "--only-extra" in sys.argv: jobs = [j for j in jobs if j["kind"] == "extra"]
snap = Path.home() / ".cache/huggingface/hub/models--Systran--faster-whisper-small/snapshots/536b0662742c02347bc0e980a01041f333bce120"
asr = VerbatimASR(str(snap), device="cpu", compute_type="int8")
for j in jobs:
    f = outdir / f"{j['id']}_asr.json"
    if f.exists(): continue
    audio, sr = sf.read(j["wav"], dtype="float32"); t = time.time()
    words = asr.transcribe_chunked(audio, chunk_s=8.0)
    f.write_text(json.dumps([w.to_dict() for w in words], ensure_ascii=False, indent=2), "utf8")
    print(j["id"], len(words), round(time.time() - t, 1), flush=True)
