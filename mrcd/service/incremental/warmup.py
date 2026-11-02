"""Deterministic warm-up: one analysis of a fixed synthetic signal, result discarded.

It makes the process state (torch thread count after Silero is imported, ONNX/Whisper first-call costs) the same for
every later analysis, offline or incremental. No parameter of any model changes; it is the call the first request makes.
"""

from __future__ import annotations

import numpy as np

from mrcd.constants import SAMPLE_RATE


def synthetic_signal(seconds: float = 13.0) -> np.ndarray:
    """Fixed 13 s signal generated in code (seeded noise + harmonic tone); contains no recorded speech."""
    rng = np.random.default_rng(20261101)
    t = np.arange(int(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    x = (
        0.08 * np.sin(2 * np.pi * 140 * t)
        + 0.04 * np.sin(2 * np.pi * 280 * t)
        + 0.01 * rng.standard_normal(len(t))
    )
    return x.astype(np.float32)


def warm_up(analyzer, options_list) -> None:
    """Runs on the engine worker thread, once per preloaded (profile, head)."""
    for options in options_list:
        analyzer.analyze_array(synthetic_signal(), SAMPLE_RATE, options)
