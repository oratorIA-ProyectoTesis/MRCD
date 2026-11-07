"""ASR survival, CTC acoustic model (results/asr_survival_ctc_20261107/PROTOCOL.md): wav2vec2 XLSR-53 Spanish, greedy CTC, no LM.

Decodes the 1025 frame units of results/asr_recovery_20261031/subset.json. One forward pass per unit (no chunking), float32 on
CUDA, argmax over the vocabulary, collapse repeats + drop blanks (the tokenizer's greedy decode). Resumable, atomic writes.
Then a determinism check: the first 20 units are decoded a second time and compared.
Inputs are read from the main checkout by absolute path (MRCD_MAIN, default /path/to/mrcd); outputs go to this repo.
usage (venv /path/to/_venv_crisper): python decode_ctc.py
"""

import json
import os
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor  # noqa: E402

torch.set_num_threads(4)
REPO = Path(__file__).resolve().parents[2]
MAIN = Path(os.environ.get("MRCD_MAIN", "/path/to/mrcd"))
SUBSET = MAIN / "results/asr_recovery_20261031/subset.json"
WAV = MAIN / "results/baselines_20261030/dihana/wav16k"
DEST = REPO / "results" / "asr_survival_ctc_20261107"
RUNS = DEST / "runs" / "asr" / "ctc_xlsr"
MODEL = "jonatasgrosman/wav2vec2-large-xlsr-53-spanish"
REV = "96d7e9b4e4a78af515a3c6d3cee7c0826045d276"


def decode_unit(model, proc, uid):
    audio, sr = sf.read(WAV / f"{uid}.wav", dtype="float32")
    assert sr == 16000 and audio.ndim == 1
    tic = time.perf_counter()
    if len(audio) < 400:  # post-freeze technical fix: below the 400-sample receptive field (0-frame wav): empty text, no forward pass
        return {"unit_id": uid, "model": f"{MODEL}@{REV} cuda float32 greedy-ctc no-lm", "n_samples": int(len(audio)), "sec": 0.0, "text": "", "too_short": True}
    x = proc(audio, sampling_rate=16000, return_tensors="pt").input_values.to("cuda")
    with torch.inference_mode():
        logits = model(x).logits
    ids = torch.argmax(logits, dim=-1)
    text = proc.batch_decode(ids)[0]
    return {
        "unit_id": uid,
        "model": f"{MODEL}@{REV} cuda float32 greedy-ctc no-lm",
        "n_samples": int(len(audio)),
        "sec": time.perf_counter() - tic,
        "text": text,
    }


def main():
    frame = json.loads(SUBSET.read_text("utf8"))["frame"]
    ids = [s["unit_id"] for s in frame]
    assert len(ids) == 1025 and all(int(s["fold"]) in (1, 2, 3, 4) for s in frame)
    torch.manual_seed(13)
    proc = Wav2Vec2Processor.from_pretrained(MODEL, revision=REV)
    model = Wav2Vec2ForCTC.from_pretrained(MODEL, revision=REV).to("cuda").eval()
    RUNS.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    for k, uid in enumerate(ids):
        p = RUNS / f"{uid}.json"
        if p.exists():
            continue
        rec = decode_unit(model, proc, uid)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(rec, ensure_ascii=False), "utf8")
        tmp.replace(p)
        if k % 100 == 0:
            print(f"ctc {k + 1}/{len(ids)} {time.perf_counter() - t0:.0f}s", flush=True)
    chk = RUNS.parent.parent / "env_check_ctc.json"
    if not chk.exists():
        res = []
        for uid in ids[:20]:
            new = decode_unit(model, proc, uid)["text"]
            old = json.loads((RUNS / f"{uid}.json").read_text("utf8"))["text"]
            res.append({"unit_id": uid, "identical": new == old})
        n_ok = sum(r["identical"] for r in res)
        chk.write_text(json.dumps({"n": len(res), "identical": n_ok, "units": res}, indent=1), "utf8")
        print(f"determinism check: {n_ok}/{len(res)} identical", flush=True)
    print("DONE", f"{time.perf_counter() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
