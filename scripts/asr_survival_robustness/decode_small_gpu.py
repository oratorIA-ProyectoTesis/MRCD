"""Device-confound control (ADDENDUM section 2): Whisper small on GPU float16.

Same Systran snapshot as the cached CPU runs; variants P0, p_none, P1, C1 on the 1025 DIHANA frame units; everything else
identical to decode_largev3.py (8 s chunks / 2 s overlap, ctranslate2 seed 13 before each unit). Resumable, atomic writes.
Determinism check on the first 20 units of P0. usage (mrcd venv): python decode_small_gpu.py
"""

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
_NV = REPO.parent / "_venv_gpu" / "Lib" / "site-packages" / "nvidia"
for sub in ("cublas", "cudnn", "cuda_nvrtc"):
    d = _NV / sub / "bin"
    if d.is_dir():
        os.add_dll_directory(str(d))
        os.environ["PATH"] = str(d) + os.pathsep + os.environ["PATH"]

import soundfile as sf  # noqa: E402

from mrcd.asr_recovery.common import OUT, WAV  # noqa: E402
from mrcd.asr_recovery.decode import decode_kwargs, transcribe_chunked  # noqa: E402

DEST = REPO / "results" / "asr_survival_robustness_20261104"
RUNS = DEST / "runs" / "asr"
SNAP = (
    Path.home()
    / ".cache/huggingface/hub/models--Systran--faster-whisper-small/snapshots/536b0662742c02347bc0e980a01041f333bce120"
)


class Shim:
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
        "model": "faster-whisper-small cuda float16",
        "n_samples": int(len(audio)),
        "sec": time.perf_counter() - tic,
        "decode_kwargs": {
            k: (list(v) if isinstance(v, tuple) else v)
            for k, v in decode_kwargs(variant).items()
        },
        "words": [w.to_dict() for w in words],
    }


def main():
    from faster_whisper import WhisperModel

    frame = json.loads((OUT / "subset.json").read_text("utf8"))["frame"]
    ids = [s["unit_id"] for s in frame]
    assert len(ids) == 1025 and all(int(s["fold"]) in (1, 2, 3, 4) for s in frame)
    assert SNAP.is_dir(), SNAP
    asr = Shim(WhisperModel(str(SNAP), device="cuda", compute_type="float16"))
    t00 = time.perf_counter()
    for v in ["P0", "p_none", "P1", "C1"]:
        d = RUNS / f"sg_{v}"
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
            if k % 100 == 0:
                print(
                    f"sg {v} {k + 1}/{len(ids)} {time.perf_counter() - t0:.0f}s",
                    flush=True,
                )
        print(f"DONE sg {v} {time.perf_counter() - t0:.0f}s", flush=True)
    chk = RUNS.parent / "env_check_sg_P0.json"
    if not chk.exists():
        res = []
        for uid in ids[:20]:
            new = [w["text"] for w in decode_unit(asr, uid, "P0")["words"]]
            old = [
                w["text"]
                for w in json.loads((RUNS / "sg_P0" / f"{uid}.json").read_text("utf8"))[
                    "words"
                ]
            ]
            res.append({"unit_id": uid, "identical": new == old})
        chk.write_text(
            json.dumps(
                {"n": 20, "identical": sum(r["identical"] for r in res), "units": res},
                indent=1,
            ),
            "utf8",
        )
    print("DONE all", f"{time.perf_counter() - t00:.0f}s", flush=True)


if __name__ == "__main__":
    main()
