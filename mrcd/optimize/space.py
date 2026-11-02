"""Explicit, finite search space and the seeded sampler. The protocol (O_campaign/PROTOCOL.md) freezes this file."""

from __future__ import annotations

import hashlib
import itertools
import json
from dataclasses import asdict, dataclass

import numpy as np

CLASSES = ("filler_word", "prolongation", "repetition")
BUDGET = 60  # max configurations evaluated per outer fold and class
PATIENCE = 20  # stop after this many consecutive configurations without a gain > EPS
EPS = 0.002
SEED = 13

PROFILES = {
    "filler_word": ("p0",),  # only P0 transcripts are cached for the 5056-turn pool
    "prolongation": ("p0",),
    "repetition": ("p0", "p1", "p0_noloop", "p1_noloop"),  # cached on the 1025-turn frame
}
# feature block sets (names are keys of mrcd.optimize.blocks.BLOCKS)
BLOCK_SETS = {
    "filler_word": (
        ("rule", "v0"),
        ("rule", "v0", "ctxfill"),
        ("rule", "v0", "ctxfill", "sig"),
        ("rule", "v0", "ctxfill", "sig", "pause"),
        ("rule", "v0", "ctxfill", "emb"),
        ("rule", "v0", "ctxfill", "sig", "pause", "emb"),
        ("rule", "ctxfill", "sig"),
    ),
    "prolongation": (
        ("v0",),
        ("v0", "sig"),
        ("v0", "sig", "lenx"),
        ("v0", "sig", "lenx", "pause"),
        ("v0", "emb"),
        ("v0", "sig", "lenx", "pause", "emb"),
        ("sig", "lenx", "pause"),
    ),
    "repetition": (
        ("reptxt", "rule"),
        ("reptxt", "rule", "loop"),
        ("reptxt", "rule", "loop", "sig"),
        ("reptxt", "rule", "loop", "sig", "v0"),
        ("reptxt", "rule", "loop", "sig", "acoustic"),
        ("reptxt", "rule", "loop", "sig", "v0", "acoustic"),
        ("reptxt", "rule", "v0"),
    ),
}
ACOUSTIC_PROFILES = ("p0",)  # DTW features were computed on the P0 words only
MODELS = (
    ("lr", (("C", 0.03),)),
    ("lr", (("C", 0.3),)),
    ("lr", (("C", 3.0),)),
    ("hgb", (("max_depth", 2), ("max_iter", 100))),
    ("hgb", (("max_depth", 2), ("max_iter", 200))),
    ("hgb", (("max_depth", 3), ("max_iter", 100))),
    ("hgb", (("max_depth", 3), ("max_iter", 200))),
)
FUSIONS = {"filler_word": ("none", "or_rule"), "prolongation": ("none",), "repetition": ("none", "or_rule")}


@dataclass(frozen=True)
class Config:
    cls: str
    profile: str
    blocks: tuple
    model: str
    hp: tuple
    fusion: str

    def key(self) -> str:
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:12]

    def to_dict(self) -> dict:
        return {**asdict(self), "blocks": list(self.blocks), "hp": dict(self.hp), "key": self.key()}

    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        return cls(d["cls"], d["profile"], tuple(d["blocks"]), d["model"], tuple(sorted(d["hp"].items())), d["fusion"])

    def label(self) -> str:
        h = ",".join(f"{k}={v}" for k, v in self.hp)
        return f"{self.profile}|{'+'.join(self.blocks)}|{self.model}({h})|{self.fusion}"


def full_space(cls: str) -> list[Config]:
    out = []
    for prof, bs, (m, hp), fu in itertools.product(PROFILES[cls], BLOCK_SETS[cls], MODELS, FUSIONS[cls]):
        if "acoustic" in bs and prof not in ACOUSTIC_PROFILES:
            continue
        if fu == "or_rule" and "rule" not in bs:
            continue
        out.append(Config(cls, prof, tuple(bs), m, tuple(sorted(hp)), fu))
    return out


def sample_order(cls: str, seed: int, budget: int = BUDGET) -> list[Config]:
    """Seeded permutation of the whole space, truncated to the budget. Deterministic."""
    sp = full_space(cls)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(sp))[:budget]
    return [sp[i] for i in idx]


def describe() -> dict:
    return {c: {"size": len(full_space(c)), "profiles": list(PROFILES[c]), "block_sets": len(BLOCK_SETS[c])} for c in CLASSES}


def space_hash() -> str:
    h = hashlib.sha256()
    for c in CLASSES:
        for cf in full_space(c):
            h.update(cf.key().encode())
    return h.hexdigest()
