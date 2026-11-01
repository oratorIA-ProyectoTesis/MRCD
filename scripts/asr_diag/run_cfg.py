"""One ASR config per process. usage: run_cfg.py <tag> <variant> [clips...]
variants: parity (env OMP=4, VerbatimASR default model) | baseline (no OMP env, model replaced w/ cpu_threads=N) |
 omp4_replaced | noomp_default | threads_default etc. Env is set by caller via OMP_NUM_THREADS."""
import os, sys, json, hashlib
os.environ.setdefault("HF_HUB_OFFLINE", "1")
from pathlib import Path
import soundfile as sf
sys.path.insert(0, r"C:\zArnes-tesis\mrcd")
from mrcd.extractors.linguistic import VerbatimASR
from faster_whisper import WhisperModel
tag, variant, clips = sys.argv[1], sys.argv[2], sys.argv[3:]
P = Path(os.environ.get("MRCD_PILOT_ROOT", r"<PILOT_ROOT>"))
snap = str(Path.home() / ".cache/huggingface/hub/models--Systran--faster-whisper-small/snapshots/536b0662742c02347bc0e980a01041f333bce120")
asr = VerbatimASR(snap, device="cpu", compute_type="int8")
if variant.startswith("replaced"):
    ct = int(variant[len("replaced"):] or 0)
    asr.model = WhisperModel(snap, device="cpu", compute_type="int8", cpu_threads=ct)
out = {}
for c in clips:
    a, sr = sf.read(P / "inputs" / f"{c}.wav", dtype="float32")
    ws = asr.transcribe_chunked(a, 8.0)
    pil = " ".join(w["text"] for w in json.loads((P / f"results/{c}_asr.json").read_text("utf8")))
    t = " ".join(w.text for w in ws)
    out[c] = {"text": t, "same_as_pilot": t == pil, "sr": sr, "n": len(ws)}
print(json.dumps({"tag": tag, "env_OMP": os.environ.get("OMP_NUM_THREADS"), "variant": variant,
  "res": {k: v["same_as_pilot"] for k, v in out.items()}, "ncpu": os.cpu_count()}))
Path(r"C:\zArnes-tesis\mrcd\results\asr_discrepancia_20261030").joinpath(f"run_{tag}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), "utf8")
