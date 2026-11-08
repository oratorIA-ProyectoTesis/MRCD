"""Plan B CPU speed probe (PROTOCOL_EXPLORATORY_B.md section 5, item 5). Run with no other decode/training job of ours.

For one arm per process: (a) RTF over a fixed seeded sample of 100 val turns; (b) total decode time of one synthetic
120 s "session" (val turns of the sample concatenated, 8 s / 2 s chunking), i.e. the wait if the whole session were
decoded after close (an upper bound for the incremental mode, whose wait-after-close G4 is <= 10 s).
CPU, int8, OMP_NUM_THREADS=4. Appends to results/plan_b_20261108/cache/rtf_probe.json.

usage: python scripts/verbatim_ft/rtf_probe.py P1|FT_<name> [--model DIR]
"""

import os

os.environ["OMP_NUM_THREADS"] = "4"
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from decode_val import DATA, WHISPER  # noqa: E402

OUT = REPO / "results" / "plan_b_20261108" / "cache" / "rtf_probe.json"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("arm")
    ap.add_argument("--model")
    a = ap.parse_args()
    import ctranslate2

    from mrcd.asr_recovery.decode import transcribe_chunked
    from mrcd.extractors.linguistic import VerbatimASR

    variant, model = ("P1", str(WHISPER)) if a.arm == "P1" else ("p_none", a.model)
    ids = json.loads((DATA / "split.json").read_text("utf8"))["val"]["turn_ids"]
    ids = sorted(
        ids, key=lambda i: hashlib.sha256(f"20261108:rtf:{i}".encode()).hexdigest()
    )[:100]
    offs = json.loads((DATA / "offsets_val.json").read_text("utf8"))
    au = np.load(DATA / "audio_val.npy", mmap_mode="r")
    get = lambda i: (
        np.asarray(au[offs[i][0] : offs[i][0] + offs[i][1]], np.float32) / 32768.0
    )  # noqa: E731
    asr = VerbatimASR(model, device="cpu", compute_type="int8")
    transcribe_chunked(asr, get(ids[0]), variant)  # warm-up
    sec = aud = 0.0
    for i in ids:
        x = get(i)
        ctranslate2.set_random_seed(13)
        tic = time.perf_counter()
        transcribe_chunked(asr, x, variant)
        sec += time.perf_counter() - tic
        aud += len(x) / 16000
    sess = np.concatenate([get(i) for i in ids])[: 120 * 16000]
    tic = time.perf_counter()
    transcribe_chunked(asr, sess, variant)
    sess_s = time.perf_counter() - tic
    res = json.loads(OUT.read_text("utf8")) if OUT.exists() else {}
    res[a.arm] = {
        "n_turns": len(ids),
        "audio_s": aud,
        "rtf": sec / aud,
        "session_120s_decode_s": sess_s,
        "session_audio_s": len(sess) / 16000,
        "omp": os.environ["OMP_NUM_THREADS"],
        "model": model,
    }
    OUT.write_text(json.dumps(res, indent=1), "utf8")
    print(json.dumps(res[a.arm]))


if __name__ == "__main__":
    main()
