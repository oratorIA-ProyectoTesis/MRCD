"""DIHANA folds 1-4 inputs shared by Plan C (scripts/plan_c) and Plan B (scripts/verbatim_ft). EXPLORATORY.

Read-only access to local, git-ignored inputs that live in the main checkout (``MRCD_DATA_REPO``, default
``<mrcd_repo>``) and in the corpus repository (``MRCD_CORPORA_ROOT``). Fold 5 is never indexed: every loader
filters on ``mrcd.context_eval.pool.ALLOWED_FOLDS`` before touching a file, and ``assert_folds`` aborts otherwise.

Audio is decoded on the fly from the original ``.ul`` files with the codec confirmed by
``scripts/baselines/dihana_decode.py`` (mu-law 8 kHz -> ``resample_poly(2, 1)`` -> PCM16 round trip), and its sha256 is
checked against ``wav16k_manifest.json``, so the samples are bit-identical to the WAVs the earlier campaigns decoded.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import pickle
from pathlib import Path

import numpy as np

from mrcd.context_eval.pool import ALLOWED_FOLDS, assert_no_historical

DATA_REPO = Path(os.environ.get("MRCD_DATA_REPO") or Path(__file__).resolve().parents[2].parent / "mrcd")
CORPORA = Path(os.environ.get("MRCD_CORPORA_ROOT") or DATA_REPO / "data" / "corpora")
DIHANA = CORPORA / "01_DIHANA" / "extraido" / "dihana"
TURNS_CSV = DATA_REPO / "results/baselines_20261030/dihana/turns.csv"
WAV_MANIFEST = DATA_REPO / "results/baselines_20261030/dihana/wav16k_manifest.json"
SUBSET = DATA_REPO / "results/asr_recovery_20261031/subset.json"
POOL_PKL = DATA_REPO / "results/asr_recovery_20261031/_cache/pool.pkl"
P0_CACHE = DATA_REPO / "results/mejoras_20261030/asr_cache/dihana"
P1_DIR = DATA_REPO / "results/asr_recovery_20261031/asr/P1"
NONLEX_TAGS = ("f:a", "f:e", "f:m", "f:sin_identificar")
SR = 16000


def assert_folds(folds) -> None:
    folds = [int(f) for f in folds]
    assert_no_historical(folds)
    bad = sorted({f for f in folds if f not in ALLOWED_FOLDS})
    if bad:
        raise ValueError(f"folds outside {ALLOWED_FOLDS}: {bad}")


def assert_split_manifest(manifest: dict, fold_of: dict[str, int]) -> None:
    """Guard for the cached Plan B split: train/dev/val speaker sets pairwise disjoint and every turn in folds 1-4.

    ``fold_of`` maps turn_id -> fold (``load_turns`` rows). A turn missing from it counts as outside folds 1-4."""
    names = ("train", "dev", "val")
    missing = [n for n in names if n not in manifest]
    if missing:
        raise ValueError(f"split manifest lacks {missing}")
    spk = {n: set(manifest[n]["speakers"]) for n in names}
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            both = spk[a] & spk[b]
            if both:
                raise ValueError(f"speakers shared by {a} and {b}: {sorted(both)[:5]}")
    folds = []
    for n in names:
        for t in manifest[n]["turn_ids"]:
            if t not in fold_of:
                raise ValueError(f"turn {t} of split {n!r} is not in folds {ALLOWED_FOLDS}")
            folds.append(fold_of[t])
    assert_folds(folds)


def load_turns() -> dict[str, dict]:
    """turn_id -> turns.csv row, folds 1-4 only (fold-5 rows are skipped before any use)."""
    out = {}
    with TURNS_CSV.open(encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if int(r["fold"]) in ALLOWED_FOLDS:
                out[r["turn_id"]] = r
    assert_folds(r["fold"] for r in out.values())
    return out


def n_nonlex_fillers(row: dict) -> int:
    c = json.loads(row["raw_tag_counts"] or "{}")
    return int(sum(c.get(k, 0) for k in NONLEX_TAGS))


def load_pool() -> list[dict]:
    """The frozen 5056-turn pool of the ASR-recovery experiment (folds 1-4; marks, text A, P0 text)."""
    pool = pickle.loads(POOL_PKL.read_bytes())
    assert_folds(u["fold"] for u in pool)
    if len(pool) != 5056:
        raise AssertionError(f"pool has {len(pool)} turns, expected 5056")
    return pool


def load_frame() -> list[dict]:
    fr = json.loads(SUBSET.read_text("utf8"))["frame"]
    assert_folds(s["fold"] for s in fr)
    return fr


_MANIFEST: dict | None = None


def _manifest() -> dict:
    global _MANIFEST
    if _MANIFEST is None:
        _MANIFEST = {m["turn_id"]: m for m in json.loads(WAV_MANIFEST.read_text())}
    return _MANIFEST


def ulaw_decode(b: np.ndarray) -> np.ndarray:
    """Same as scripts/baselines/dihana_decode.py (G.711 mu-law -> int16)."""
    u = (~b.astype(np.uint8)).astype(np.int32)
    sign = u & 0x80
    exp = (u >> 4) & 7
    man = u & 15
    s = (((man << 3) + 0x84) << exp) - 0x84
    return np.where(sign != 0, -s, s).astype(np.int16)


def load_audio(row: dict, verify: bool = True) -> np.ndarray:
    """16 kHz float32 audio of one folds-1-4 turn, bit-identical to the historical wav16k file."""
    import soundfile as sf
    from scipy.signal import resample_poly

    assert_folds([row["fold"]])
    raw = (DIHANA / row["ul_rel"]).read_bytes()
    x = resample_poly(
        ulaw_decode(np.frombuffer(raw, np.uint8)).astype(np.float32) / 32768.0, 2, 1
    ).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, x, SR, subtype="PCM_16", format="WAV")
    data = buf.getvalue()
    if verify:
        m = _manifest().get(row["turn_id"])
        if m is None or hashlib.sha256(data).hexdigest() != m["wav_sha256"]:
            raise AssertionError(
                f"reconstructed audio of {row['turn_id']} does not match wav16k_manifest.json"
            )
    y, sr = sf.read(io.BytesIO(data), dtype="float32")
    assert sr == SR and y.ndim == 1
    return y


def duration_s(turn_id: str) -> float:
    return float(_manifest()[turn_id]["dur_s"])


def p0_text(turn_id: str) -> str:
    d = json.loads((P0_CACHE / f"{turn_id}.json").read_text("utf8"))
    return " ".join(w["text"] for w in d["words"])
