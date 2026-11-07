"""Console command ``mrcd`` (same subcommands as ``python -m mrcd.service``).

Why this tiny package lives OUTSIDE ``mrcd``
--------------------------------------------
The text that faster-whisper / ctranslate2 produces on CPU depends on the OpenMP thread count, and only
``OMP_NUM_THREADS=4`` reproduces the pinned configuration (see ``mrcd/reproducibility.py``). The variable reaches
the OpenMP runtime only when it is in the environment BEFORE numpy is first imported, and ``import mrcd`` imports
numpy. A console script that pointed into ``mrcd`` would import the package first and the pin would come too late.
This module imports nothing heavy, sets the variable when it is missing, and only then imports the service CLI.

Rules (same as ``scripts/_pin_omp.py``):
* ``OMP_NUM_THREADS`` already set -> kept (the caller's choice wins);
* unset -> set to 4;
* ``MRCD_NO_OMP_PIN=1`` -> untouched.

Importing ``mrcd`` as a library never changes the environment; only this entry point does.
"""

from __future__ import annotations

import os
import sys
from typing import MutableMapping

PINNED_OMP_THREADS = 4
_TRUE = {"1", "true", "yes", "on"}


def pin_omp(environ: MutableMapping[str, str] | None = None) -> str:
    """Returns ``"set"``, ``"kept"`` or ``"opted_out"``."""
    env = os.environ if environ is None else environ
    if env.get("MRCD_NO_OMP_PIN", "").strip().lower() in _TRUE:
        return "opted_out"
    if env.get("OMP_NUM_THREADS"):
        return "kept"
    env["OMP_NUM_THREADS"] = str(PINNED_OMP_THREADS)
    return "set"


def main(argv: list[str] | None = None) -> int:
    pin_omp()
    from mrcd.service.cli import main as cli_main

    return cli_main(sys.argv[1:] if argv is None else argv)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
