"""ASR survival on larger models (results/asr_survival_models_20261104/PROTOCOL.md), part 1: faster-whisper large-v3 on GPU.

Decodes the 1025 frame units of results/asr_recovery_20261031/subset.json with the four fixed variants of
mrcd/asr_recovery/decode.py (P0, p_none, P1, C1; this order), 8 s chunks / 2 s overlap, exactly the settings of the
small-model runs except device/precision (CUDA float16 instead of CPU int8). Resumable, atomic writes.
Then a determinism check: the first 20 units of P0 are decoded a second time and compared word for word.
usage (mrcd venv): python decode_largev3.py [--variants P0 p_none ...]
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

# cuBLAS / cuDNN 12 DLLs come from the pip wheels of the GPU venv (ctranslate2 4.8 is built for CUDA 12).
_NV = REPO.parent / "_venv_gpu" / "Lib" / "site-packages" / "nvidia"
for sub in ("cublas", "cudnn", "cuda_nvrtc"):
    d = _NV / sub / "bin"
    if d.is_dir():
        os.add_dll_directory(str(d))
        os.environ["PATH"] = str(d) + os.pathsep + os.environ["PATH"]

import soundfile as sf  # noqa: E402

from mrcd.asr_recovery.common import OUT, WAV  # noqa: E402
from mrcd.asr_recovery.decode import decode_kwargs, transcribe_chunked  # noqa: E402

DEST = REPO / "results" / "asr_survival_models_20261104"
RUNS = DEST / "runs" / "asr"
SNAP = (
    Path.home()
    / ".cache/huggingface/hub/models--Systran--faster-whisper-large-v3/snapshots/edaa852ec7e145841d8ffdb056a99866b5f0a478"
)


class Shim:
    """The two attributes ``transcribe_chunked`` reads from a ``VerbatimASR``."""

    def __init__(self, model):
        self.model = model
        self.language = "es"


def decode_unit(asr, uid, variant):
    import ctranslate2

    audio, sr = sf.read(WAV / f"{uid}.wav", dtype="float32")
    assert sr == 16000 and audio.ndim == 1
    ctranslate2.set_random_seed(13)
    tic = time.perf_counter()
    words = transcribe_chunked(asr, audio, variant)
    return {
        "unit_id": uid,
        "variant": variant,
        "model": "faster-whisper-large-v3 cuda float16",
        "n_samples": int(len(audio)),
        "sec": time.perf_counter() - tic,
        "decode_kwargs": {
            k: (list(v) if isinstance(v, tuple) else v)
            for k, v in decode_kwargs(variant).items()
        },
        "words": [w.to_dict() for w in words],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=["P0", "p_none", "P1", "C1"])
    a = ap.parse_args()
    from faster_whisper import WhisperModel

    frame = json.loads((OUT / "subset.json").read_text("utf8"))["frame"]
    ids = [s["unit_id"] for s in frame]
    assert len(ids) == 1025 and all(int(s["fold"]) in (1, 2, 3, 4) for s in frame)
    assert SNAP.is_dir(), SNAP
    asr = Shim(WhisperModel(str(SNAP), device="cuda", compute_type="float16"))
    t00 = time.perf_counter()
    for v in a.variants:
        d = RUNS / f"lv3_{v}"
        d.mkdir(parents=True, exist_ok=True)
        t0 = time.perf_counter()
        for k, uid in enumerate(ids):
            p = d / f"{uid}.json"
            if p.exists():
                continue
            rec = decode_unit(asr, uid, v)
            tmp = p.with_suffix(".tmp")
            tmp.write_text(json.dumps(rec, ensure_ascii=False), "utf8")
            tmp.replace(p)
            if k % 50 == 0:
                print(
                    f"lv3 {v} {k + 1}/{len(ids)} {time.perf_counter() - t0:.0f}s",
                    flush=True,
                )
        print(f"DONE lv3 {v} {time.perf_counter() - t0:.0f}s", flush=True)
    chk = RUNS.parent / "env_check_lv3_P0.json"
    if "P0" in a.variants and not chk.exists():
        res = []
        for uid in ids[:20]:
            new = [w["text"] for w in decode_unit(asr, uid, "P0")["words"]]
            old = [
                w["text"]
                for w in json.loads((RUNS / "lv3_P0" / f"{uid}.json").read_text("utf8"))[
                    "words"
                ]
            ]
            res.append({"unit_id": uid, "identical": new == old})
        n_ok = sum(r["identical"] for r in res)
        chk.write_text(
            json.dumps({"n": len(res), "identical": n_ok, "units": res}, indent=1),
            "utf8",
        )
        print(f"determinism check lv3 P0 re-decode: {n_ok}/{len(res)} identical", flush=True)
    print("DONE all", f"{time.perf_counter() - t00:.0f}s", flush=True)


if __name__ == "__main__":
    main()
