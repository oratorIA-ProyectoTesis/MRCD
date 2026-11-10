"""Shared inputs of Plan F (``results/plan_f_20261108/PROTOCOL_EXPLORATORY_F.md``). EXPLORATORY, DIHANA folds 1-4 only.

Everything here is read-only access to git-ignored local inputs:

* ``MRCD_REPO_ROOT`` (default ``<worktree>/../../mrcd``): the main checkout, with the frozen pool, the 16 kHz turn WAVs, the
  frozen P0 and P1 decodes and the O-campaign predictions;
* ``MRCD_PLAN_A_CACHE`` / ``MRCD_PLAN_B_CACHE`` (default: the sibling plan A / plan B worktrees): Plan A's P1 decodes and
  P0 engine features, Plan B's P1 and FT_best decodes (only used for identity checks and reuse proofs).

The validation turns are RECOMPUTED here with the Plan B rule (``scripts/verbatim_ft/vtext.make_split``); the counts must be
36 speakers / 983 turns. Fold 5, NOCANDO, AMERESCO, EPIC and CHM150 are never read (``load_pool`` asserts folds 1-4).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

WT = Path(__file__).resolve().parents[2]
MAIN = Path(os.environ.get("MRCD_REPO_ROOT") or WT.parents[1] / "mrcd")
os.environ.setdefault("MRCD_REPO_ROOT", str(MAIN))
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, str(WT))
sys.path.insert(0, str(WT / "scripts" / "verbatim_ft"))

OUT = WT / "results" / "plan_f_20261108"
CACHE = OUT / "cache"
PROTOCOL = OUT / "PROTOCOL_EXPLORATORY_F.md"
PLAN_A_CACHE = Path(
    os.environ.get("MRCD_PLAN_A_CACHE")
    or WT.parent / "k-planA" / "results" / "plan_a_20261108" / "cache"
)
PLAN_B_CACHE = Path(
    os.environ.get("MRCD_PLAN_B_CACHE")
    or WT.parent / "l-planBC" / "results" / "plan_b_20261108" / "cache"
)
EXPECTED_VAL = {"speakers": 36, "turns": 983}
SR = 16000


def protocol_sha() -> str:
    import hashlib
    import json

    h = hashlib.sha256(PROTOCOL.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    frozen = json.loads((OUT / "PROTOCOL_EXPLORATORY_F.sha256.json").read_text("utf8"))[
        "sha256_lf"
    ]
    if h != frozen:
        raise SystemExit(
            "PROTOCOL_EXPLORATORY_F.md changed after freezing: write a dated annex instead"
        )
    return h


def val_pool() -> list[dict]:
    """The pool rows of the 36 Plan B validation speakers (sorted by unit id). Asserts the Plan B counts and folds 1-4."""
    import collections

    import vtext

    from mrcd.asr_recovery import common as RC

    pool = RC.load_pool()  # folds 1-4 only, audited counts
    by_spk = collections.defaultdict(list)
    for u in pool:
        by_spk[u["speaker"]].append(u)
    split = vtext.make_split(
        {s: any(u["ref"]["rep_literal"] == 1 for u in us) for s, us in by_spk.items()}
    )
    rows = sorted(
        (u for s in split["val"] for u in by_spk[s]), key=lambda u: u["unit_id"]
    )
    got = {"speakers": len(split["val"]), "turns": len(rows)}
    if got != EXPECTED_VAL:
        raise AssertionError(f"validation set {got} != Plan B {EXPECTED_VAL}")
    train_dev = split["train"] | split["dev"]
    assert not (set(u["speaker"] for u in rows) & train_dev)
    return rows


def wav_path(uid: str) -> Path:
    from mrcd.asr_recovery import common as RC

    return RC.WAV / f"{uid}.wav"


def p0_words(uid: str) -> list[dict]:
    import json

    from mrcd.asr_recovery import common as RC

    return json.loads((RC.ASR / f"{uid}.json").read_text("utf8"))["words"]


def p1_words(uid: str, source: str = "fresh") -> list[dict]:
    """P1 words. ``fresh`` (primary): this plan's own decode (``decode_p1.py``). ``reused`` (sensitivity): the frozen ASR-recovery
    decode if it exists, else Plan A's decode (both transcribe_chunked(.., 'P1'))."""
    import json

    from mrcd.asr_recovery import common as RC

    dirs = (
        (CACHE / "asr_p1_fresh",)
        if source == "fresh"
        else (RC.OUT / "asr" / "P1", PLAN_A_CACHE / "asr_p1")
    )
    for d in dirs:
        p = d / f"{uid}.json"
        if p.exists():
            return json.loads(p.read_text("utf8"))["words"]
    raise FileNotFoundError(f"no P1 decode for a validation turn ({uid[:3]}...)")


def ft_words(uid: str) -> list[dict]:
    import json

    return json.loads((CACHE / "asr_ft" / f"{uid}.json").read_text("utf8"))["words"]


def load_features(uid: str, directory: Path) -> dict:
    import json

    import numpy as np

    z = np.load(directory / f"{uid}.npz")
    return {
        "probs": z["probs"].astype(float),
        "pooled": z["pooled"].astype(float),
        "pause": json.loads(bytes(z["pause"]).decode("utf8")),
        "n_samples": int(z["n_samples"][0]),
    }
