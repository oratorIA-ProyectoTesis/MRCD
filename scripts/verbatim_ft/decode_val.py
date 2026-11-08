"""Plan B val decodes on CPU int8, 4 threads (PROTOCOL_EXPLORATORY_B.md section 5). Resumable, atomic writes.

  P1  : the base Systran faster-whisper small, variant "P1" of mrcd.asr_recovery.decode (val turns missing from the
        O1-frame P1 cache only; the cached ones are reused by evaluate.py).
  FT_*: a fine-tuned checkpoint converted to CTranslate2 int8 (--model DIR), variant "p_none" (= p0 without prompt).
Both through VerbatimASR(device="cpu", compute_type="int8") and transcribe_chunked (8 s / 2 s), seed 13 per turn.

usage: python scripts/verbatim_ft/decode_val.py P1
       python scripts/verbatim_ft/decode_val.py FT_smoke --model <ft_models_dir>/ct2/smoke_int8
"""

import os

os.environ["OMP_NUM_THREADS"] = (
    "4"  # before numpy / ctranslate2 (the ASR text depends on it)
)
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import dihana_io as dio  # noqa: E402

DATA = REPO / "results" / "plan_b_20261108" / "cache" / "data"
ASR_OUT = REPO / "results" / "plan_b_20261108" / "cache" / "asr"
WHISPER = (
    Path.home()
    / ".cache/huggingface/hub/models--Systran--faster-whisper-small/snapshots/536b0662742c02347bc0e980a01041f333bce120"
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("arm")
    ap.add_argument("--model")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard", default="0/1", help="k/n: decode every n-th turn starting at k (resumable shards)")
    a = ap.parse_args()
    import ctranslate2

    from mrcd.asr_recovery.decode import transcribe_chunked
    from mrcd.extractors.linguistic import VerbatimASR

    variant, model = ("P1", str(WHISPER)) if a.arm == "P1" else ("p_none", a.model)
    if not model:
        raise SystemExit("--model is required for a fine-tuned arm")
    ids = json.loads((DATA / "split.json").read_text("utf8"))["val"]["turn_ids"]
    if a.arm == "P1":
        ids = [i for i in ids if not (dio.P1_DIR / f"{i}.json").exists()]
    k_, n_ = (int(x) for x in a.shard.split("/"))
    ids = ids[k_::n_]
    if a.limit:
        ids = ids[: a.limit]
    offs = json.loads((DATA / "offsets_val.json").read_text("utf8"))
    audio_all = np.load(DATA / "audio_val.npy", mmap_mode="r")
    asr = VerbatimASR(model, device="cpu", compute_type="int8")
    d = ASR_OUT / a.arm
    d.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    for k, uid in enumerate(ids):
        p = d / f"{uid}.json"
        if p.exists():
            continue
        s, n = offs[uid]
        audio = np.asarray(audio_all[s : s + n], np.float32) / 32768.0
        ctranslate2.set_random_seed(13)
        tic = time.perf_counter()
        words = transcribe_chunked(asr, audio, variant)
        rec = {
            "unit_id": uid,
            "arm": a.arm,
            "variant": variant,
            "model": model,
            "sec": time.perf_counter() - tic,
            "audio_s": n / dio.SR,
            "omp_num_threads": os.environ["OMP_NUM_THREADS"],
            "words": [w.to_dict() for w in words],
        }
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(rec, ensure_ascii=False), "utf8")
        tmp.replace(p)
        if k % 50 == 0:
            print(
                f"{a.arm} {k + 1}/{len(ids)} {time.perf_counter() - t0:.0f}s",
                flush=True,
            )
    print("DONE", a.arm, len(ids), flush=True)


if __name__ == "__main__":
    main()
