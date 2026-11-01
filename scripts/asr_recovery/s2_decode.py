"""O1 decoding of the frozen subset. Usage: python s2_decode.py VARIANT SHARD NSHARDS  (resumable, atomic writes)."""

import _pin  # noqa: F401  (MUST be first)
import json
import os
import sys
import time

import soundfile as sf

from mrcd.asr_recovery.common import OUT, WAV, WHISPER
from mrcd.asr_recovery.decode import VARIANTS, decode_kwargs, transcribe_chunked


def main(variant, shard, nshard):
    assert variant in VARIANTS
    frame = json.loads((OUT / "subset.json").read_text("utf8"))["frame"]
    ids = [s["unit_id"] for s in frame]
    mine = [u for i, u in enumerate(ids) if i % nshard == shard]
    d = OUT / "asr" / variant
    d.mkdir(parents=True, exist_ok=True)
    from mrcd.extractors.linguistic import VerbatimASR

    asr = VerbatimASR(str(WHISPER), device="cpu", compute_type="int8")
    import ctranslate2

    t0 = time.perf_counter()
    for k, uid in enumerate(mine):
        p = d / f"{uid}.json"
        if p.exists():
            continue
        audio, sr = sf.read(WAV / f"{uid}.wav", dtype="float32")
        assert sr == 16000 and audio.ndim == 1
        ctranslate2.set_random_seed(13)
        tic = time.perf_counter()
        words = transcribe_chunked(asr, audio, variant)
        rec = {
            "unit_id": uid,
            "variant": variant,
            "n_samples": int(len(audio)),
            "omp_num_threads": os.environ["OMP_NUM_THREADS"],
            "sec": time.perf_counter() - tic,
            "decode_kwargs": {k_: (list(v) if isinstance(v, tuple) else v) for k_, v in decode_kwargs(variant).items()},
            "words": [w.to_dict() for w in words],
        }
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(rec, ensure_ascii=False), "utf8")
        tmp.replace(p)
        if k % 25 == 0:
            print(f"{variant} shard {shard} {k + 1}/{len(mine)} {time.perf_counter() - t0:.0f}s", flush=True)
    print("DONE", variant, shard, flush=True)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))
