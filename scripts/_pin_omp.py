"""Pin OMP_NUM_THREADS=4 for reproducible ASR text. Standard library ONLY on purpose.

Import this FIRST in an entry point, before numpy / torch / ctranslate2 / mrcd:

    import _pin_omp  # noqa: F401            (script run as a file)
    # or, when scripts/ is a package:  from scripts import _pin_omp  # noqa: F401

Why it is not inside ``mrcd``: ``import mrcd`` imports numpy, and the variable only reaches the OpenMP
runtime when it is in the environment BEFORE numpy is first imported (measured, see
results/asr_discrepancia_20261030 and mrcd/reproducibility.py). A pin implemented inside the package
would run too late.

Behaviour (never silent):
* variable already set  -> kept as is (the caller's choice wins), recorded in ``REPORT``;
* variable unset        -> set to 4, recorded;
* opt-out               -> ``MRCD_NO_OMP_PIN=1`` in the environment or ``--no-omp-pin`` in argv: untouched;
* numpy already imported -> the assignment would have no effect: nothing is assigned and a warning is
  emitted (``effective`` False).
Importing a library module never changes the environment; only entry points that import this do.
"""

from __future__ import annotations

import os
import sys
import warnings

THREADS = 4
_TRUE = {"1", "true", "yes", "on"}


def pin(threads: int = THREADS, *, argv: list[str] | None = None) -> dict:
    argv = sys.argv if argv is None else argv
    before = os.environ.get("OMP_NUM_THREADS")
    late = "numpy" in sys.modules
    if os.environ.get("MRCD_NO_OMP_PIN", "").lower() in _TRUE or "--no-omp-pin" in argv:
        return {"action": "opted_out", "value": before, "late": late, "effective": None}
    if before is not None:
        return {
            "action": "kept",
            "value": before,
            "late": late,
            "effective": before == str(threads),
        }
    if late:
        warnings.warn(
            "OMP_NUM_THREADS is not set and numpy was imported before the pin: setting it now would "
            "NOT affect the ASR text. Put it in the process environment, or import _pin_omp first. "
            "Opt out of this check with MRCD_NO_OMP_PIN=1.",
            RuntimeWarning,
            stacklevel=2,
        )
        return {
            "action": "not_set_too_late",
            "value": None,
            "late": True,
            "effective": False,
        }
    os.environ["OMP_NUM_THREADS"] = str(threads)
    return {"action": "set", "value": str(threads), "late": False, "effective": True}


REPORT = pin()
