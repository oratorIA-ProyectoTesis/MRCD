"""Plan A cache: P1 words and P1 engine unit features for every turn of DIHANA folds 1-4 (resumable, atomic writes).

    python scripts/plan_a/build_cache.py SHARD NSHARDS        # decode + features for the turns of this shard
    python scripts/plan_a/build_cache.py verify N             # re-decode N turns that already have a frozen P1 decode
    python scripts/plan_a/build_cache.py p0 SHARD NSHARDS     # P0 unit features from audio + the frozen P0 words

Run from the worktree root with PYTHONPATH=<worktree>, MRCD_REPO_ROOT=<main checkout> (read-only inputs: pool cache,
turn WAVs, frozen P1 decodes), OMP_NUM_THREADS=4, HF_HUB_OFFLINE=1. Fold-5 rows are filtered out after parsing turns.csv and never used (``load_pool``asserts folds 1-4 only). A turn whose two cache files exist is skipped, so a killed shard loses at most the turn in progress.

Decoding reuses the frozen ASR-recovery code path exactly (``asr_recovery.decode.transcribe_chunked(..., "P1")``,
CPU int8, ctranslate2 seed 13 before each turn). Features are ``DisfluencyEngine.unit_features(audio, p1_words)``, the
path the service uses for the unit pipelines. Nothing here computes a metric.
"""

from __future__ import annotations

import os

os.environ.setdefault(
    "OMP_NUM_THREADS", "4"
)  # before numpy (the ASR text depends on it)

import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

WT = Path(__file__).resolve().parents[2]
CACHE = WT / "results" / "plan_a_20261108" / "cache"
ASR_DIR = CACHE / "asr_p1"
FEAT_DIR = CACHE / "feat_p1"
#: P0 features are recomputed from audio with the same service path: the stored campaign shards
#: (results/mejoras_20261030/features) also hold fold-5 rows, so they are never loaded by this plan.
FEAT_P0_DIR = CACHE / "feat_p0"


def _atomic_write(p: Path, data: bytes) -> None:
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(p)


def _engine():
    from mrcd.engine import DisfluencyEngine

    return DisfluencyEngine.load(
        WT / "models" / "ckpt_audio_text.pt",
        device="cpu",
        asr_profile="p1_disfluent_prompt",
    )


def _decode(asr, audio: np.ndarray) -> tuple[list[dict], float]:
    import ctranslate2

    from mrcd.asr_recovery.decode import transcribe_chunked

    ctranslate2.set_random_seed(13)
    tic = time.perf_counter()
    words = transcribe_chunked(asr, audio, "P1")
    return [w.to_dict() for w in words], time.perf_counter() - tic


def p1_words(uid: str) -> list[dict] | None:
    """P1 words of a turn: the frozen ASR-recovery decode if it exists, else this plan's decode, else None."""
    from mrcd.asr_recovery import common as RC

    for d in (RC.OUT / "asr" / "P1", ASR_DIR):
        p = d / f"{uid}.json"
        if p.exists():
            return json.loads(p.read_text("utf8"))["words"]
    return None


def save_features(uid: str, f: dict, out_dir: Path = FEAT_DIR) -> None:
    import io

    buf = io.BytesIO()
    np.savez(
        buf,
        probs=np.asarray(f["probs"], np.float32),
        pooled=np.asarray(f["pooled"], np.float32),
        pause=np.frombuffer(json.dumps(f["pause"]).encode("utf8"), np.uint8),
        n_samples=np.asarray([int(f["n_samples"])]),
    )
    _atomic_write(out_dir / f"{uid}.npz", buf.getvalue())


def load_features(uid: str, out_dir: Path = FEAT_DIR) -> dict:
    z = np.load(out_dir / f"{uid}.npz")
    return {
        "probs": z["probs"].astype(float),
        "pooled": z["pooled"].astype(float),
        "pause": json.loads(bytes(z["pause"]).decode("utf8")),
        "n_samples": int(z["n_samples"][0]),
    }


def run_shard(shard: int, nshard: int) -> None:
    from mrcd.asr_recovery import common as RC

    ASR_DIR.mkdir(parents=True, exist_ok=True)
    FEAT_DIR.mkdir(parents=True, exist_ok=True)
    pool = RC.load_pool()  # folds 1-4 only, audited counts
    ids = [u["unit_id"] for u in pool]
    mine = [u for i, u in enumerate(ids) if i % nshard == shard]
    todo = [u for u in mine if not (FEAT_DIR / f"{u}.npz").exists()]
    print(f"shard {shard}/{nshard}: {len(mine)} turns, {len(todo)} to do", flush=True)
    if not todo:
        print("DONE", shard, flush=True)
        return
    eng = _engine()
    t0 = time.perf_counter()
    for k, uid in enumerate(todo):
        audio, sr = sf.read(RC.WAV / f"{uid}.wav", dtype="float32")
        assert sr == 16000 and audio.ndim == 1
        words = p1_words(uid)
        if words is None:
            words, sec = _decode(eng._asr, audio)
            rec = {
                "unit_id": uid,
                "variant": "P1",
                "n_samples": int(len(audio)),
                "omp_num_threads": os.environ["OMP_NUM_THREADS"],
                "sec": sec,
                "words": words,
            }
            _atomic_write(
                ASR_DIR / f"{uid}.json",
                json.dumps(rec, ensure_ascii=False).encode("utf8"),
            )
        save_features(uid, eng.unit_features(audio, words))
        if k % 25 == 0:
            print(
                f"shard {shard} {k + 1}/{len(todo)} {time.perf_counter() - t0:.0f}s",
                flush=True,
            )
    print("DONE", shard, flush=True)


def p0_words(uid: str) -> list[dict]:
    from mrcd.asr_recovery import common as RC

    return json.loads((RC.ASR / f"{uid}.json").read_text("utf8"))["words"]


def run_p0_shard(shard: int, nshard: int) -> None:
    from mrcd.asr_recovery import common as RC

    FEAT_P0_DIR.mkdir(parents=True, exist_ok=True)
    pool = RC.load_pool()
    mine = [u["unit_id"] for i, u in enumerate(pool) if i % nshard == shard]
    todo = [u for u in mine if not (FEAT_P0_DIR / f"{u}.npz").exists()]
    print(f"p0 shard {shard}/{nshard}: {len(todo)} to do", flush=True)
    if todo:
        eng = _engine()
        t0 = time.perf_counter()
        for k, uid in enumerate(todo):
            audio, sr = sf.read(RC.WAV / f"{uid}.wav", dtype="float32")
            assert sr == 16000 and audio.ndim == 1
            save_features(uid, eng.unit_features(audio, p0_words(uid)), FEAT_P0_DIR)
            if k % 100 == 0:
                print(f"p0 shard {shard} {k + 1}/{len(todo)} {time.perf_counter() - t0:.0f}s", flush=True)
    print("DONE", shard, flush=True)


def verify(n: int) -> None:
    """Decode-only fidelity check: n turns of the frozen P1 frame re-decoded here must give the same text."""
    from mrcd.asr_recovery import common as RC

    frozen = sorted((RC.OUT / "asr" / "P1").glob("*.json"))
    rng = np.random.default_rng(13)
    pick = [frozen[i] for i in rng.choice(len(frozen), size=n, replace=False)]
    eng = _engine()
    same = 0
    for p in pick:
        ref = json.loads(p.read_text("utf8"))
        audio, _ = sf.read(RC.WAV / f"{ref['unit_id']}.wav", dtype="float32")
        words, _ = _decode(eng._asr, audio)
        a = " ".join(w["text"] for w in words)
        b = " ".join(w["text"] for w in ref["words"])
        same += int(a == b)
    out = {"n": n, "same_text": same}
    (CACHE / "verify_decode.json").write_text(json.dumps(out), "utf8")
    print(json.dumps(out), flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "p0":
        run_p0_shard(int(sys.argv[2]), int(sys.argv[3]))
    elif sys.argv[1] == "verify":
        CACHE.mkdir(parents=True, exist_ok=True)
        verify(int(sys.argv[2]))
    else:
        run_shard(int(sys.argv[1]), int(sys.argv[2]))
