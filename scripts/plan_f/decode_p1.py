"""Plan F: fresh P1 decodes of the 983 validation turns (protocol section 3: reused P1 decodes that are not identical to Plan B's
on the common turns are re-decoded). Same code path as Plan A / ASR recovery: ``asr_recovery.decode.transcribe_chunked(asr,
audio, "P1")``, CPU int8, ``OMP_NUM_THREADS=4``, ``ctranslate2`` seed 13 per turn. Resumable; cache ``cache/asr_p1_fresh/``.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402


def main() -> None:
    C.protocol_sha()
    import ctranslate2
    import numpy as np
    import soundfile as sf

    from mrcd.asr_recovery import common as RC
    from mrcd.asr_recovery.decode import transcribe_chunked
    from mrcd.extractors.linguistic import VerbatimASR

    ids = [u["unit_id"] for u in C.val_pool()]
    d = C.CACHE / "asr_p1_fresh"
    d.mkdir(parents=True, exist_ok=True)
    todo = [i for i in ids if not (d / f"{i}.json").exists()]
    print(f"{len(ids)} turns, {len(todo)} to do", flush=True)
    if not todo:
        print("DONE", flush=True)
        return
    asr = VerbatimASR(str(RC.WHISPER), device="cpu", compute_type="int8")
    t0 = time.perf_counter()
    for k, uid in enumerate(todo):
        audio, sr = sf.read(C.wav_path(uid), dtype="float32")
        assert sr == C.SR and audio.ndim == 1
        ctranslate2.set_random_seed(13)
        tic = time.perf_counter()
        words = transcribe_chunked(asr, np.asarray(audio, np.float32), "P1")
        rec = {
            "unit_id": uid,
            "variant": "P1",
            "sec": time.perf_counter() - tic,
            "omp_num_threads": __import__("os").environ["OMP_NUM_THREADS"],
            "words": [w.to_dict() for w in words],
        }
        tmp = d / f"{uid}.tmp"
        tmp.write_text(json.dumps(rec, ensure_ascii=False), "utf8")
        tmp.replace(d / f"{uid}.json")
        if k % 50 == 0:
            print(f"{k + 1}/{len(todo)} {time.perf_counter() - t0:.0f}s", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
