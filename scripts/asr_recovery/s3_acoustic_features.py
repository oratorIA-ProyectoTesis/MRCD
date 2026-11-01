"""O3 feature extraction for every pool unit (folds 1-4): DTW self-similarity, ASR- and VAD-segmented. Usage: python s3_acoustic_features.py [NPROC]"""

import _pin  # noqa: F401  (MUST be first)
import json
import pickle
import sys
import time
from multiprocessing import Pool

import soundfile as sf

from mrcd.asr_recovery import acoustic as A
from mrcd.asr_recovery.common import ASR, OUT, WAV, load_pool


def work(uid):
    audio, sr = sf.read(WAV / f"{uid}.wav", dtype="float32")
    assert sr == 16000 and audio.ndim == 1
    words = json.loads((ASR / f"{uid}.json").read_text("utf8"))["words"]
    return uid, A.acoustic_features(audio, words)


def main(nproc):
    pool = load_pool()
    ids = [u["unit_id"] for u in pool]
    t0 = time.time()
    with Pool(nproc) as p:
        res = dict(p.imap_unordered(work, ids, chunksize=20))
    assert set(res) == set(ids)
    d = OUT / "features"
    d.mkdir(exist_ok=True)
    (d / "acoustic.pkl").write_bytes(pickle.dumps({"names_asr": A.NAMES_ASR, "names_vad": A.NAMES_VAD, "feats": res}))
    print("done", len(res), f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 6)
