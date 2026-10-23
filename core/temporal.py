"""Pure, registered temporal ownership and evidence contract for future v2 adapters.

Ownership partitions real media; requested evidence may extend beyond it. No legacy
feature extractor or checkpoint is implicitly upgraded by importing this module.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from numbers import Integral

TEMPORAL_CONTRACT = "cell500-centered3000-v2"
STRIDE_MS = 500
CONTEXT_MS = 3000
_EVIDENCE_FIELDS = (
    "requested_start_ms", "requested_end_ms", "observed_start_ms",
    "observed_end_ms", "pad_left_ms", "pad_right_ms",
)


def integer_ms(value, name: str) -> int:
    """Reject booleans, floats (including NaN), and guessed fractional milliseconds."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer in milliseconds")
    return int(value)


def _configuration(stride_ms=STRIDE_MS, context_ms=CONTEXT_MS):
    if integer_ms(stride_ms, "stride_ms") != STRIDE_MS:
        raise ValueError(f"{TEMPORAL_CONTRACT} requires stride_ms={STRIDE_MS}")
    if integer_ms(context_ms, "context_ms") != CONTEXT_MS:
        raise ValueError(f"{TEMPORAL_CONTRACT} requires context_ms={CONTEXT_MS}")


def _duration(duration_ms):
    duration = integer_ms(duration_ms, "duration_ms")
    if duration < 0:
        raise ValueError("duration_ms must be nonnegative")
    return duration


def decision_cells(duration_ms, stride_ms=STRIDE_MS) -> list[dict]:
    """Cover [0, actual duration) exactly, including short media and a partial tail."""
    _configuration(stride_ms)
    duration = _duration(duration_ms)
    return [
        {
            "temporal_contract": TEMPORAL_CONTRACT,
            "media_duration_ms": duration,
            "index": index,
            "owned_start_ms": start,
            "owned_end_ms": min(start + STRIDE_MS, duration),
            # Do not shift a partial tail's nominal center towards the start.
            "nominal_center_ms": start + STRIDE_MS // 2,
        }
        for index, start in enumerate(range(0, duration, STRIDE_MS))
    ]


def _evidence(cell, duration):
    center = cell["nominal_center_ms"]
    start, end = center - CONTEXT_MS // 2, center + CONTEXT_MS // 2
    observed_start, observed_end = max(0, start), min(duration, end)
    return dict(
        requested_start_ms=start, requested_end_ms=end,
        observed_start_ms=observed_start, observed_end_ms=observed_end,
        pad_left_ms=observed_start - start, pad_right_ms=end - observed_end,
    )


def validate_temporal_grid(cells, duration_ms, stride_ms=STRIDE_MS,
                           context_ms=CONTEXT_MS) -> int:
    """Validate complete ownership/order and, when present, complete evidence bounds."""
    _configuration(stride_ms, context_ms)
    duration = _duration(duration_ms)
    if not isinstance(cells, Sequence) or isinstance(cells, (str, bytes)):
        raise ValueError("cells must be an ordered sequence")
    expected = decision_cells(duration)
    if len(cells) != len(expected):
        raise ValueError("cell count must exactly cover media_duration_ms including the tail")
    for index, (cell, owned) in enumerate(zip(cells, expected)):
        if not isinstance(cell, Mapping):
            raise ValueError(f"cell {index} must be a mapping")
        for field, value in owned.items():
            actual = cell.get(field)
            if field != "temporal_contract":
                actual = integer_ms(actual, f"cell {index}.{field}")
            if actual != value:
                raise ValueError(f"cell {index}.{field} disagrees with the registered grid")
        if any(field in cell for field in _EVIDENCE_FIELDS):
            for field, value in _evidence(owned, duration).items():
                if integer_ms(cell.get(field), f"cell {index}.{field}") != value:
                    raise ValueError(f"cell {index}.{field} disagrees with real evidence support")
    return len(cells)


def evidence_windows(cells, context_ms=CONTEXT_MS) -> list[dict]:
    """Return copies with nominal 3s requests and real-media-clipped observation."""
    _configuration(context_ms=context_ms)
    if not isinstance(cells, Sequence) or isinstance(cells, (str, bytes)):
        raise ValueError("cells must be an ordered sequence")
    if not cells:
        return []  # The only valid empty grid represents duration zero.
    if not isinstance(cells[0], Mapping):
        raise ValueError("cell 0 must be a mapping")
    duration = _duration(cells[0].get("media_duration_ms"))
    validate_temporal_grid(cells, duration)
    return [dict(cell, **_evidence(cell, duration)) for cell in cells]
