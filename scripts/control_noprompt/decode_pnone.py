"""No-prompt control (results/control_noprompt_20261103/PROTOCOL.md). ONE process, CPU int8, OMP_NUM_THREADS=4.

Step 1: environment check - re-decode P0 on the first 20 frame units and compare word for word with the cached P0.
Step 2: decode p_none on the 1025 frame units (resumable, atomic writes). Cached P0 / P1 / C1 are NOT re-decoded.
usage: python decode_pnone.py
"""

import os
import sys

NUMPY_FIRST = "numpy" in sys.modules
os.environ["OMP_NUM_THREADS"] = "4"
os.environ.setdefault("HF_HUB_OFFLINE", "1")
if NUMPY_FIRST:
    raise SystemExit("numpy imported before the OMP pin")
from pathlib import Path  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import json  # noqa: E402
import time  # noqa: E402

import soundfile as sf  # noqa: E402

from mrcd.asr_recovery.common import OUT, WAV, WHISPER  # noqa: E402
from mrcd.asr_recovery.decode import decode_kwargs, transcribe_chunked  # noqa: E402

DEST = REPO / "results" / "control_noprompt_20261103"
RUNS = DEST / "runs"


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
        "n_samples": int(len(audio)),
        "omp_num_threads": os.environ["OMP_NUM_THREADS"],
        "sec": time.perf_counter() - tic,
        "decode_kwargs": {
            k: (list(v) if isinstance(v, tuple) else v)
            for k, v in decode_kwargs(variant).items()
        },
        "words": [w.to_dict() for w in words],
    }


def main():
    frame = json.loads((OUT / "subset.json").read_text("utf8"))["frame"]
    ids = [s["unit_id"] for s in frame]
    assert len(ids) == 1025 and all(int(s["fold"]) in (1, 2, 3, 4) for s in frame)
    from mrcd.extractors.linguistic import VerbatimASR

    asr = VerbatimASR(str(WHISPER), device="cpu", compute_type="int8")

    # Step 1: determinism / environment check against the cached P0 (first 20 frame units).
    chk = RUNS / "env_check_p0.json"
    if not chk.exists():
        res = []
        for uid in ids[:20]:
            new = [w["text"] for w in decode_unit(asr, uid, "P0")["words"]]
            old = [
                w["text"]
                for w in json.loads(
                    (OUT / "asr" / "P0" / f"{uid}.json").read_text("utf8")
                )["words"]
            ]
            res.append({"unit_id": uid, "identical": new == old})
        n_ok = sum(r["identical"] for r in res)
        RUNS.mkdir(parents=True, exist_ok=True)
        chk.write_text(
            json.dumps({"n": len(res), "identical": n_ok, "units": res}, indent=1),
            "utf8",
        )
        print(
            f"env check P0 re-decode: {n_ok}/{len(res)} identical to cache", flush=True
        )

    # Step 2: p_none on the whole frame.
    d = RUNS / "asr" / "p_none"
    d.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    for k, uid in enumerate(ids):
        p = d / f"{uid}.json"
        if p.exists():
            continue
        rec = decode_unit(asr, uid, "p_none")
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(rec, ensure_ascii=False), "utf8")
        tmp.replace(p)
        if k % 50 == 0:
            print(
                f"p_none {k + 1}/{len(ids)} {time.perf_counter() - t0:.0f}s", flush=True
            )
    print("DONE p_none", f"{time.perf_counter() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
