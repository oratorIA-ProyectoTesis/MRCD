"""Is MRCD's ASR (faster-whisper small, CPU int8, transcribe_chunked 8 s) reproducible? Same wavs, repeated runs and different cpu_threads,
compared with the pilot's cached words. Writes results/baselines_20261030/asr_determinism.json."""
import json, os, sys
from pathlib import Path
os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, str(Path(__file__).parent))
import soundfile as sf
from bl_common import REPO, WHISPER, OUT, write_config
sys.path.insert(0, str(REPO))
from mrcd.extractors.linguistic import VerbatimASR
P = Path(os.environ.get("MRCD_PILOT_ROOT", r"<PILOT_ROOT>"))
inp = json.loads((P / "input_manifest.json").read_text("utf8"))[:6]
txt = lambda ws: " ".join(w.text for w in ws)
res = {}
for ct in [0, 1, 4]:
    from faster_whisper import WhisperModel
    asr = VerbatimASR(str(WHISPER), device="cpu", compute_type="int8")
    asr.model = WhisperModel(str(WHISPER), device="cpu", compute_type="int8", cpu_threads=ct)
    for rep in [0, 1]:
        res[f"threads{ct}_rep{rep}"] = {}
        for r in inp:
            a, _ = sf.read(P / r["input_wav"], dtype="float32")
            res[f"threads{ct}_rep{rep}"][r["sample_id"]] = txt(asr.transcribe_chunked(a, 8.0))
pilot = {r["sample_id"]: " ".join(w["text"] for w in json.loads((P / f"results/{r['sample_id']}_asr.json").read_text("utf8"))) for r in inp}
keys = list(res)
summ = {"identical_to_pilot": {k: sum(res[k][s] == pilot[s] for s in pilot) for k in keys},
        "rep0_vs_rep1_same_threads": {f"threads{ct}": sum(res[f"threads{ct}_rep0"][s] == res[f"threads{ct}_rep1"][s] for s in pilot) for ct in [0, 1, 4]},
        "threads0_vs_threads4": sum(res["threads0_rep0"][s] == res["threads4_rep0"][s] for s in pilot), "n": len(pilot)}
(OUT / "asr_determinism.json").write_text(json.dumps({"summary": summ, "texts": res, "pilot": pilot}, indent=1, ensure_ascii=True))
write_config(OUT / "asr_determinism_run", "asr_determinism", __file__, {"n": 6}, {}, ckpt=False)
print(json.dumps(summ))
