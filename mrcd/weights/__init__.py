"""Engine checkpoint resolution: explicit path > ``$MRCD_CHECKPOINT`` > default location, sha256-verified.

PUBLIC REPOSITORY: the trained checkpoint is NOT distributed (see ``models/README.md``; available on request for
non-commercial research). The packaged location next to this file therefore stays empty here, and the default falls
back to ``models/ckpt_audio_text.pt`` in a clone, where the file is placed by hand. Private builds that do ship the
~300 KB file as package data (``mrcd/weights/ckpt_audio_text.pt``) keep working: the packaged copy wins when present.

The checkpoint is a pickle (``torch.load(..., weights_only=False)``): unpickling runs code. The default is
therefore checked against :data:`CHECKPOINT_SHA256` BEFORE it is handed to torch; a modified file is refused.
An explicit path (argument or ``MRCD_CHECKPOINT``) is the caller's decision and is used as given; its sha256 is
still recorded in every result's provenance.

Standard library only: importing this module must stay cheap.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Mapping

CHECKPOINT_NAME = "ckpt_audio_text.pt"
#: sha256 of the published ``audio_text`` checkpoint (same bytes as ``models/ckpt_audio_text.pt``)
CHECKPOINT_SHA256 = "87f95108a765bfacbbc092ff1b7580cfc804fea84132a9813163d928d033b669"
PACKAGED_CHECKPOINT = Path(__file__).resolve().parent / CHECKPOINT_NAME
#: historical location in a clone of the repository (absent in an installed package)
REPO_CHECKPOINT = Path(__file__).resolve().parents[2] / "models" / CHECKPOINT_NAME
ENV_VAR = "MRCD_CHECKPOINT"
#: what the service and the engine load when no path is given: the packaged copy if this build ships one,
#: otherwise the clone location (``models/ckpt_audio_text.pt``, not distributed in the public repository)
DEFAULT_CHECKPOINT = PACKAGED_CHECKPOINT if PACKAGED_CHECKPOINT.is_file() else REPO_CHECKPOINT


class CheckpointError(RuntimeError):
    """The default checkpoint is missing or does not match its pinned sha256."""


def sha256_of(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def verify_packaged_checkpoint(
    path: Path = DEFAULT_CHECKPOINT, expected: str = CHECKPOINT_SHA256
) -> Path:
    if not path.is_file():
        raise CheckpointError(
            f"the engine checkpoint {path} is missing: it is not distributed with MRCD (available on request for "
            f"research, see models/README.md). Place it at models/{CHECKPOINT_NAME} or pass --ckpt / {ENV_VAR}"
        )
    actual = sha256_of(path)
    if actual != expected:
        raise CheckpointError(
            f"the engine checkpoint {path} has sha256 {actual}, expected {expected}; "
            "the file was modified or truncated; obtain it again from the authors."
        )
    return path


def resolve_checkpoint(
    path: str | Path | None = None, *, environ: Mapping[str, str] | None = None
) -> Path:
    """Checkpoint to load: explicit ``path`` > ``$MRCD_CHECKPOINT`` > default location (sha256 verified)."""
    if path is not None and str(path) != "":
        return Path(path)
    env = os.environ if environ is None else environ
    from_env = env.get(ENV_VAR, "").strip()
    if from_env:
        return Path(from_env)
    return verify_packaged_checkpoint()


__all__ = [
    "CHECKPOINT_NAME",
    "CHECKPOINT_SHA256",
    "PACKAGED_CHECKPOINT",
    "DEFAULT_CHECKPOINT",
    "REPO_CHECKPOINT",
    "ENV_VAR",
    "CheckpointError",
    "resolve_checkpoint",
    "verify_packaged_checkpoint",
    "sha256_of",
]
