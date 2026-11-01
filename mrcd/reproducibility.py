"""ASR reproducibility helpers (new module, 2026-10-30).

Finding that motivates this module (results/asr_discrepancia_20261030): the text produced by
faster-whisper / ctranslate2 int8 on CPU depends on the OpenMP thread count. Only
``OMP_NUM_THREADS=4`` reproduces the pilot transcripts; an unset variable (20 threads on the
development machine) changes the words of most clips. The model code never set it, so the
effective ASR was "whatever the host environment says".

``pin_asr_threads`` records / applies the variable, documents the choice, and lets the integrator opt out.

IMPORTANT (measured 2026-10-30, same script, same clip): assigning ``os.environ["OMP_NUM_THREADS"]="4"`` AFTER
``import numpy`` does NOT reproduce the OMP=4 transcript (the text equals an OMP-unset run) although ``os.environ`` shows "4".
Setting it BEFORE the first numpy import does. ``import mrcd`` imports numpy, therefore this function cannot make
the pin effective by itself when called after ``import mrcd``: the integrator must put the variable in the process
environment (shell / service unit) or assign it as the very first statement of the entry point. The returned
``effective`` flag says so, and ``scripts/improve/asr_cache.py`` additionally runs a transcript canary. ``asr_environment`` returns a small
dict that callers should store next to every result so that a transcript can be traced back to the
environment that produced it.

Opt-out
-------
* set ``MRCD_NO_OMP_PIN=1`` in the environment, or
* call ``pin_asr_threads(enabled=False)``.

An existing ``OMP_NUM_THREADS`` chosen by the integrator is respected (never silently replaced)
unless ``force=True``; the returned report says so, and the value is still recorded.
"""

from __future__ import annotations

import hashlib
import os
import platform
import sys
from importlib import metadata
from pathlib import Path

ENV_VAR = "OMP_NUM_THREADS"
OPT_OUT_ENV = "MRCD_NO_OMP_PIN"
DEFAULT_ASR_OMP_THREADS = 4
#: importing any of these BEFORE the variable is assigned makes the assignment ineffective for the ASR text
LATE_MODULES = ("numpy", "torch", "ctranslate2", "faster_whisper")

_TRUE = {"1", "true", "yes", "on"}


def pin_asr_threads(
    threads: int = DEFAULT_ASR_OMP_THREADS,
    *,
    enabled: bool = True,
    force: bool = False,
    environ: dict | None = None,
) -> dict:
    """Pin ``OMP_NUM_THREADS`` for reproducible ASR text. Call before importing ctranslate2.

    Returns a report ``{"action", "omp_num_threads", "requested", "late", "effective"}`` where ``action`` is one of
    ``"set"`` (we set it), ``"kept"`` (already had a value, left untouched), ``"overridden"``
    (``force=True`` replaced a different value) or ``"opted_out"``. ``late`` is True when numpy / torch /
    ctranslate2 were already imported. ``effective`` is False when we assigned the value late (it will NOT
    reproduce the pinned ASR text), True when the value equals ``threads`` and was not assigned late
    (a pre-existing value is assumed to come from the process environment), None when opted out.
    """
    env = os.environ if environ is None else environ
    if threads < 1:
        raise ValueError("threads must be >= 1")
    late = any(m in sys.modules for m in LATE_MODULES)
    current = env.get(ENV_VAR)
    if not enabled or str(env.get(OPT_OUT_ENV, "")).lower() in _TRUE:
        return {
            "action": "opted_out",
            "omp_num_threads": current,
            "requested": threads,
            "late": late,
            "effective": None,
        }
    if current is None:
        env[ENV_VAR] = str(threads)
        action = "set"
    elif current == str(threads):
        action = "kept"
    elif force:
        env[ENV_VAR] = str(threads)
        action = "overridden"
    else:
        action = "kept"
    assigned = action in ("set", "overridden")
    return {
        "action": action,
        "omp_num_threads": env.get(ENV_VAR),
        "requested": threads,
        "late": late,
        "effective": env.get(ENV_VAR) == str(threads) and not (assigned and late),
    }


def file_sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _version(dist: str) -> str | None:
    try:
        return metadata.version(dist)
    except metadata.PackageNotFoundError:
        return None


def asr_environment(
    model_dir: str | Path | None = None, *, environ: dict | None = None
) -> dict:
    """Describe the environment that determines the ASR text (no heavy imports).

    ``model_dir`` is the faster-whisper model folder; when given, the sha256 of ``model.bin`` is added.
    """
    env = os.environ if environ is None else environ
    out = {
        "omp_num_threads": env.get(ENV_VAR),
        "omp_pin_opted_out": str(env.get(OPT_OUT_ENV, "")).lower() in _TRUE,
        "cpu_count_logical": os.cpu_count(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "faster_whisper": _version("faster-whisper"),
        "ctranslate2": _version("ctranslate2"),
        "numpy": _version("numpy"),
    }
    if model_dir is not None:
        mb = Path(model_dir) / "model.bin"
        out["model_dir"] = str(model_dir)
        out["model_bin_sha256"] = file_sha256(mb) if mb.exists() else None
    out["environment_is_pinned"] = out["omp_num_threads"] == str(
        DEFAULT_ASR_OMP_THREADS
    )
    return out


# ----------------------------------------------------------------------------- engine integration
def omp_status(*, import_state: dict | None = None, environ: dict | None = None) -> dict:
    """Was ``OMP_NUM_THREADS`` in the environment before numpy was first imported by ``mrcd``?

    ``set_before_numpy``: True (present when ``import mrcd`` ran and numpy was not loaded yet),
    False (absent at that time: unset, or assigned later, which does NOT take effect for ctranslate2),
    None (numpy was already imported before ``mrcd``: cannot be verified from here).
    ``effective``: True only when the value is the pinned one (4) AND ``set_before_numpy`` is True;
    None when the value is 4 but unverifiable; False otherwise.
    """
    if import_state is None:
        import mrcd

        import_state = mrcd._IMPORT_STATE
    env = os.environ if environ is None else environ
    now = env.get(ENV_VAR)
    at_import = import_state["omp_num_threads_at_import"]
    pre = import_state["numpy_imported_before_mrcd"]
    set_before = None if (pre and at_import is not None) else (at_import is not None)
    changed_late = now != at_import
    if now != str(DEFAULT_ASR_OMP_THREADS) or set_before is False or changed_late:
        effective: bool | None = False
    else:
        effective = None if set_before is None else True
    return {
        "omp_num_threads": now,
        "omp_num_threads_at_mrcd_import": at_import,
        "numpy_imported_before_mrcd": pre,
        "set_before_numpy": set_before,
        "assigned_after_import": changed_late,
        "effective": effective,
    }


def omp_warning(status: dict) -> str | None:
    """Warning text when the ASR thread configuration is not the pinned one, else None.

    Cases: unset at import; assigned after numpy (ineffective); set before numpy but different from
    the pinned value (4). A verified pinned value, or one that cannot be verified (numpy imported
    before ``mrcd``), gives None.
    """
    at_import = status["omp_num_threads_at_mrcd_import"]
    now = status["omp_num_threads"]
    pinned = str(DEFAULT_ASR_OMP_THREADS)
    if at_import is None:
        why = (
            "OMP_NUM_THREADS is not set"
            if now is None
            else "OMP_NUM_THREADS was assigned after numpy was imported, which has no effect on ctranslate2"
        )
    elif at_import != pinned:
        why = f"OMP_NUM_THREADS={at_import} differs from the pinned value {pinned}"
    elif now != at_import:
        why = "OMP_NUM_THREADS was changed after numpy was imported, which has no effect on ctranslate2"
    else:
        return None
    return (
        f"{why}: the ASR text (and therefore the decisions) depends on the OpenMP thread count; "
        f"only OMP_NUM_THREADS={pinned} reproduces the pinned configuration. Set it in the "
        "process environment before the first numpy import (entry points can use scripts/_pin_omp.py; "
        f"opt out with {OPT_OUT_ENV}=1). See mrcd/reproducibility.py."
    )


def engine_environment(asr=None) -> dict:
    """Effective environment to store next to a result (no heavy imports besides torch's thread count)."""
    env = asr_environment()
    env.update(omp_status())
    try:
        import torch

        torch_threads = torch.get_num_threads()
    except Exception:  # pragma: no cover
        torch_threads = None
    env["cpu_threads"] = {
        "torch": torch_threads,
        # VerbatimASR builds WhisperModel without cpu_threads (library default); only OMP matters (measured).
        "asr_cpu_threads_arg": 0 if asr is not None and hasattr(asr, "compute_type") else None,
    }
    if asr is not None:
        env["asr"] = {
            k: getattr(asr, k, None) for k in ("model_size", "device", "compute_type")
        }
    return env
