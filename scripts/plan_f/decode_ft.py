"""Plan F: decode the 983 validation turns with the ``ft_small_v1`` profile (resumable, atomic writes).

    python scripts/plan_f/decode_ft.py [--limit N] [--shard K/N]

Environment: PYTHONPATH=<worktree>, MRCD_REPO_ROOT=<main checkout>, OMP_NUM_THREADS=4, MRCD_FT_CT2_DIR optional (default:
the repo-relative ``_models_ft/ct2/capped_best_int8``). Goes through the PRODUCT path ``VerbatimASR(profile="ft_small_v1")``
+ ``transcribe_chunked`` (8 s chunks, 2 s overlap), CPU int8, ``ctranslate2`` seed 13 before each turn. Nothing here computes a
metric. One file per turn under ``results/plan_f_20261108/cache/asr_ft/`` (git-ignored); a finished turn is skipped.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard", default="0/1")
    a = ap.parse_args()
    C.protocol_sha()
    import ctranslate2
    import numpy as np
    import soundfile as sf

    from mrcd.extractors.linguistic import VerbatimASR

    rows = C.val_pool()
    k_, n_ = (int(x) for x in a.shard.split("/"))
    ids = [u["unit_id"] for u in rows][k_::n_]
    if a.limit:
        ids = ids[: a.limit]
    d = C.CACHE / "asr_ft"
    d.mkdir(parents=True, exist_ok=True)
    todo = [i for i in ids if not (d / f"{i}.json").exists()]
    print(f"{len(ids)} turns, {len(todo)} to do", flush=True)
    if not todo:
        print("DONE", flush=True)
        return
    asr = VerbatimASR(profile="ft_small_v1")
    info = asr.model_info()
    (C.CACHE / "ft_model_info.json").write_text(
        json.dumps(
            {
                "model_bin_sha256": info["model_bin_sha256"],
                "compute_type": info["compute_type"],
                "device": info["device"],
                "profile": asr.profile.describe(),
            },
            indent=1,
        ),
        "utf8",
    )
    t0 = time.perf_counter()
    for k, uid in enumerate(todo):
        audio, sr = sf.read(C.wav_path(uid), dtype="float32")
        assert sr == C.SR and audio.ndim == 1
        ctranslate2.set_random_seed(13)
        tic = time.perf_counter()
        words = asr.transcribe_chunked(np.asarray(audio, np.float32))
        rec = {
            "unit_id": uid,
            "profile": "ft_small_v1",
            "sec": time.perf_counter() - tic,
            "audio_s": len(audio) / C.SR,
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
