"""Back-of-the-envelope sample-size helpers for the Via B corpus (planning only, no decision depends on them)."""

from __future__ import annotations

import math
from statistics import NormalDist


def expected_positives(
    rate_per_min: float, speech_min: float, retention: float = 1.0
) -> float:
    """Expected number of reference events: rate x minutes x retention (retention < 1 models lower rates in
    prepared presentations than in the corpora the rate was measured on)."""
    return rate_per_min * speech_min * retention


def minutes_needed(
    rate_per_min: float, target_positives: int = 40, retention: float = 1.0
) -> float | None:
    eff = rate_per_min * retention
    return None if eff <= 0 else target_positives / eff


def wilson_halfwidth(p: float, n: int, level: float = 0.95) -> float:
    """Half-width of the Wilson interval for a proportion (recall/precision with n positives)."""
    z = NormalDist().inv_cdf(0.5 + level / 2)
    denom = 1 + z * z / n
    return z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom


def detectable_paired_difference(
    n_pos: int, discordant_rate: float, alpha: float = 0.05, power: float = 0.80
) -> float:
    """Smallest difference in recall between two systems on the same n positives detectable with the given power,
    using the normal approximation of a paired (McNemar-type) contrast: delta = (z_a + z_b) sqrt(d / n), where d is
    the share of positives on which exactly one system succeeds. Speaker clustering inflates this (design effect not
    included), so it is a LOWER bound on the detectable difference."""
    nd = NormalDist()
    return (nd.inv_cdf(1 - alpha / 2) + nd.inv_cdf(power)) * math.sqrt(
        discordant_rate / n_pos
    )
