"""First-run download of the Whisper model used by the ASR, with a clear message (size, source, destination).

The engine loads Whisper through faster-whisper, which reads the Hugging Face cache. The command line calls
:func:`ensure_whisper_model` before it loads the engine, so the only download happens up front, visibly, and never
in the middle of a request. ``HF_HUB_OFFLINE=1`` is optional: with it, a missing model is a clear error instead of a
download (the Docker image sets it, because the model is baked into the image).
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Callable, TextIO

#: approximate download size of the faster-whisper conversions (model.bin + tokenizer + vocabulary), MB
APPROX_SIZE_MB = {
    "tiny": 75,
    "base": 145,
    "small": 486,
    "medium": 1530,
    "large-v3": 3090,
}
_TRUE = {"1", "true", "yes", "on"}


class WhisperModelMissing(RuntimeError):
    """The model is not cached and downloading is disabled (``HF_HUB_OFFLINE``)."""


#: files faster-whisper fetches (same list as ``faster_whisper.utils.download_model``)
_ALLOW_PATTERNS = [
    "config.json",
    "preprocessor_config.json",
    "model.bin",
    "tokenizer.json",
    "vocabulary.*",
]


def _download(size: str, local_files_only: bool) -> str:
    from faster_whisper.utils import download_model  # type: ignore

    if local_files_only or "/" in size:
        return download_model(size, local_files_only=local_files_only)
    # Same repository, files and cache as faster-whisper, but WITH the hub's progress bars
    # (faster-whisper disables them, which makes a ~0.5 GB first download look like a hang).
    import huggingface_hub
    from faster_whisper.utils import _MODELS  # type: ignore

    repo_id = _MODELS.get(size)
    if repo_id is None:
        return download_model(size)  # raises the library's own "invalid size" error
    return huggingface_hub.snapshot_download(repo_id, allow_patterns=_ALLOW_PATTERNS)


def cached_model_dir(
    size: str, *, downloader: Callable[[str, bool], str] = _download
) -> Path | None:
    """Folder of the cached model, or None when it is not in the local cache."""
    if Path(size).is_dir():
        return Path(size)
    try:
        return Path(downloader(size, True))
    except Exception:  # noqa: BLE001 - "not cached" surfaces as several hub exception types
        return None


def ensure_whisper_model(
    size: str = "small",
    *,
    stream: TextIO | None = None,
    environ: dict | None = None,
    downloader: Callable[[str, bool], str] = _download,
) -> Path:
    """Returns the model folder, downloading it once (with a message) when it is not cached yet."""
    out = sys.stderr if stream is None else stream
    env = os.environ if environ is None else environ
    found = cached_model_dir(size, downloader=downloader)
    if found is not None:
        return found
    if str(env.get("HF_HUB_OFFLINE", "")).strip().lower() in _TRUE:
        raise WhisperModelMissing(
            f"Whisper '{size}' is not in the local cache and HF_HUB_OFFLINE is set. "
            f"Run `mrcd download --whisper {size}` once without HF_HUB_OFFLINE, or unset it."
        )
    approx = APPROX_SIZE_MB.get(size)
    where = env.get("HF_HUB_CACHE") or env.get("HF_HOME") or "~/.cache/huggingface"
    print(
        f"[mrcd] First run: downloading Whisper '{size}' (faster-whisper conversion, "
        f"{'~' + str(approx) + ' MB' if approx else 'size unknown'}) from Hugging Face "
        f"into {where}. This happens once; later runs use the cache. "
        "Set HF_HUB_OFFLINE=1 afterwards to forbid any network access.",
        file=out,
        flush=True,
    )
    t0 = time.monotonic()
    path = Path(
        downloader(size, False)
    )  # huggingface_hub prints its own per-file progress bars
    print(
        f"[mrcd] Whisper '{size}' ready in {time.monotonic() - t0:.0f} s: {path}",
        file=out,
        flush=True,
    )
    return path
