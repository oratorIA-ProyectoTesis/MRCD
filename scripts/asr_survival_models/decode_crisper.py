"""ASR survival on larger models (results/asr_survival_models_20261104/PROTOCOL.md), part 2: CrisperWhisper 2.0 large.

Verbatim mode, Spanish, package defaults (no prompt parameter exists), GPU float16, transformers backend.
Resumable, atomic writes; the first 20 units are decoded a second time as a determinism check.
usage (venv /path/to/_venv_crisper): python decode_crisper.py
"""

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results" / "asr_recovery_20261031"
WAV = REPO / "results/baselines_20261030/dihana/wav16k"
DEST = REPO / "results" / "asr_survival_models_20261104"
RUNS = DEST / "runs" / "asr"
MODEL = "nyralabs/CrisperWhisper2.0_large"
KW = dict(language="es", mode="verbatim", word_timestamps=False, sr=16000)

import soundfile as sf  # noqa: E402
import torch  # noqa: E402
from crisperwhisper import CrisperWhisperModel  # noqa: E402


def decode_unit(m, uid):
    audio, sr = sf.read(WAV / f"{uid}.wav", dtype="float32")
    assert sr == 16000 and audio.ndim == 1
    torch.manual_seed(13)
    tic = time.perf_counter()
    r = m.transcribe(audio, **KW)
    return {
        "unit_id": uid,
        "model": MODEL,
        "n_samples": int(len(audio)),
        "sec": time.perf_counter() - tic,
        "decode_kwargs": KW,
        "text": r.text,
    }


def main():
    frame = json.loads((OUT / "subset.json").read_text("utf8"))["frame"]
    ids = [s["unit_id"] for s in frame]
    assert len(ids) == 1025 and all(int(s["fold"]) in (1, 2, 3, 4) for s in frame)
    m = CrisperWhisperModel(MODEL, backend="transformers", device="cuda", compute_type="float16")
    d = RUNS / "cw_large"
    d.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    for k, uid in enumerate(ids):
        p = d / f"{uid}.json"
        if p.exists():
            continue
        rec = decode_unit(m, uid)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(rec, ensure_ascii=False), "utf8")
        tmp.replace(p)
        if k % 50 == 0:
            print(f"cw_large {k + 1}/{len(ids)} {time.perf_counter() - t0:.0f}s", flush=True)
    print(f"DONE cw_large {time.perf_counter() - t0:.0f}s", flush=True)
    chk = RUNS.parent / "env_check_cw_large.json"
    if not chk.exists():
        res = []
        for uid in ids[:20]:
            new = decode_unit(m, uid)["text"]
            old = json.loads((d / f"{uid}.json").read_text("utf8"))["text"]
            res.append({"unit_id": uid, "identical": new == old})
        n_ok = sum(r["identical"] for r in res)
        chk.write_text(json.dumps({"n": len(res), "identical": n_ok, "units": res}, indent=1), "utf8")
        print(f"determinism check cw_large re-decode: {n_ok}/{len(res)} identical", flush=True)


if __name__ == "__main__":
    main()
