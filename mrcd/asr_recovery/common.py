"""Paths, pool loading (folds 1-4 only), outer partition and task units shared by every script."""

from __future__ import annotations
import os

import hashlib
import json
import pickle
from pathlib import Path

import numpy as np

from mrcd.context_eval.partition import assert_no_crossing, group_folds
from mrcd.context_eval.pool import assert_no_historical, build_pool

REPO = Path(r"C:\zArnes-tesis\mrcd")
OUT = REPO / "results" / "asr_recovery_20261031"
SEED = 13
CORPUS = Path(
    (os.environ.get("MRCD_CORPORA_ROOT", r"<CORPORA_ROOT>") + r"\01_DIHANA\extraido\dihana")
)
TURNS = REPO / "results/baselines_20261030/dihana/turns.csv"
ASR = REPO / "results/mejoras_20261030/asr_cache/dihana"
WAV = REPO / "results/baselines_20261030/dihana/wav16k"
PARENT = REPO / "results/ejecucion_material4_20261030/experimentos"
TASKS = ("rep_literal", "revision")
LITERAL_SUBTYPES = {"A_literal_repeat"}  # same restriction as the parent decomposition
EXPECTED = {
    "turns": 5056,
    "speakers": 180,
    "rep_literal": (214, 4775, 67),  # pos, neg, indeterminate
    "revision": (161, 4895),
}


def sha(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def sha_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf8")).hexdigest()


def check_pool_counts(pool: list[dict]) -> None:
    """Abort if the pool is not the one the protocol was written for."""
    assert_no_historical(u["fold"] for u in pool)
    if len(pool) != EXPECTED["turns"]:
        raise AssertionError(f"pool has {len(pool)} turns, expected {EXPECTED['turns']}")
    if len({u["speaker"] for u in pool}) != EXPECTED["speakers"]:
        raise AssertionError("unexpected speaker count")
    rl = [u["ref"]["rep_literal"] for u in pool]
    got = (sum(v == 1 for v in rl), sum(v == 0 for v in rl), sum(v is None for v in rl))
    if got != EXPECTED["rep_literal"]:
        raise AssertionError(f"rep_literal counts {got}")
    rv = [u["ref"]["revision"] for u in pool]
    if (sum(v == 1 for v in rv), sum(v == 0 for v in rv)) != EXPECTED["revision"]:
        raise AssertionError("revision counts differ")


def load_pool() -> list[dict]:
    cache = OUT / "_cache" / "pool.pkl"
    if cache.exists():
        pool = pickle.loads(cache.read_bytes())
    else:
        pool = build_pool(CORPUS, TURNS, ASR)
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_bytes(pickle.dumps(pool))
    check_pool_counts(pool)
    return pool


def outer_folds(pool: list[dict]) -> np.ndarray:
    """EXACTLY the parent partition (speaker sha256 order, K=5, seed 13)."""
    spk = [u["speaker"] for u in pool]
    f = group_folds(spk, 5, SEED)
    assert_no_crossing(spk, f, "speaker")
    assert_no_crossing([u["dialogue"] for u in pool], f, "dialogue")
    return f


def task_units(pool: list[dict], task: str):
    idx, y = [], []
    for i, u in enumerate(pool):
        v = u["ref"][task]
        if v is None:
            continue
        idx.append(i)
        y.append(int(v))
    return np.array(idx), np.array(y)

WHISPER = Path.home() / ".cache/huggingface/hub/models--Systran--faster-whisper-small/snapshots/536b0662742c02347bc0e980a01041f333bce120"


def run_config(inputs: dict | None = None, extra: dict | None = None) -> dict:
    """Reproducibility block shared by every run: versions, effective OMP, code / protocol hashes, inputs."""
    import os
    import platform
    import sys
    import time
    from importlib import metadata

    def ver(name):
        try:
            return metadata.version(name)
        except Exception:
            return None

    code = {
        str(p.relative_to(REPO)): sha(p)
        for p in sorted((REPO / "mrcd" / "asr_recovery").glob("*.py")) + sorted((REPO / "scripts" / "asr_recovery").glob("*.py"))
    }
    cfg = {
        "seed": SEED,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "OMP_NUM_THREADS_effective": os.environ.get("OMP_NUM_THREADS"),
        "numpy_imported_before_pin": "_pin" in sys.modules and bool(getattr(sys.modules["_pin"], "NUMPY_ALREADY_IMPORTED", False)),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "versions": {n: ver(n) for n in ("numpy", "scipy", "scikit-learn", "librosa", "numba", "soundfile", "faster-whisper", "ctranslate2")},
        "protocol_sha256": json.loads((OUT / "PROTOCOL.sha256.json").read_text("utf8"))["sha256"],
        "code_sha256": code,
        "inputs_sha256": inputs or {},
    }
    cfg.update(extra or {})
    return cfg
