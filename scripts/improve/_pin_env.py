"""Import this FIRST in every entry point (before numpy / torch / ctranslate2 / mrcd).

Finding (2026-10-30): setting OMP_NUM_THREADS from Python AFTER `import numpy` does NOT reproduce the OMP=4 ASR text,
even though os.environ shows "4" (same script: variable set before numpy -> pilot transcript reproduced; set after -> text
of an OMP-unset run). `import mrcd` imports numpy, so `mrcd.reproducibility.pin_asr_threads()` called after `import mrcd`
is too late; the variable must be in the process environment before any numpy import.
"""
import os
os.environ["OMP_NUM_THREADS"] = "4"
