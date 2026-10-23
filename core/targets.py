"""Pure human-interval -> owned-cell targets, not a legacy NPZ/training adapter.

Every supervised head has an explicit eligibility mask. A proposed label is not
human evidence: positives require type_known=True; context requires a separate,
explicit boolean is_disfluent on an event decision.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy

import numpy as np

from core.constants import BACKGROUND, TAXONOMY, TAXONOMY_VERSION
from core.temporal import (
    TEMPORAL_CONTRACT, decision_cells, evidence_windows, integer_ms,
    validate_temporal_grid,
)

TARGET_CONTRACT = "mrcd-target-projection-v2"
_CONTEXT_LABELS = ["functional", "disfluent"]
_BOOL_ARRAYS = (
    "multi_hot", "type_mask", "fully_reviewed", "type_ambiguity",
    "softmax_mask", "context_multi_hot", "context_conflict", "context_mask",
)
_INT_ARRAYS = ("softmax_target", "context_target")


def _interval(start, end, duration, name):
    start, end = integer_ms(start, f"{name}.start"), integer_ms(end, f"{name}.end")
    if not 0 <= start < end <= duration:
        raise ValueError(f"{name} must satisfy 0 <= start < end <= media_duration_ms")
    return start, end


def _coverage(regions, duration):
    if not isinstance(regions, Sequence) or isinstance(regions, (str, bytes)):
        raise ValueError("reviewed_coverage must be a sequence of [start_ms, end_ms]")
    ordered = []
    for index, region in enumerate(regions):
        if not isinstance(region, (list, tuple)) or len(region) != 2:
            raise ValueError(f"reviewed_coverage {index} must have exactly two bounds")
        ordered.append(_interval(*region, duration, f"reviewed_coverage {index}"))
    union = []
    for start, end in sorted(ordered):
        if union and start <= union[-1][1]:
            union[-1][1] = max(union[-1][1], end)
        else:
            union.append([start, end])
    return union


def _events(events, duration):
    if not isinstance(events, Sequence) or isinstance(events, (str, bytes)):
        raise ValueError("source_events must be an ordered sequence")
    parsed = []
    for index, event in enumerate(events):
        if not isinstance(event, Mapping):
            raise ValueError(f"source event {index} must be a mapping")
        start, end = _interval(event.get("start_ms"), event.get("end_ms"), duration,
                               f"source event {index}")
        label = event.get("label")
        if label is not None and label not in TAXONOMY:
            raise ValueError(f"source event {index} has a noncanonical type label")
        known = event.get("type_known", False)
        if type(known) is not bool:
            raise ValueError(f"source event {index}.type_known must be a boolean")
        decision = event.get("decision", "event")
        if decision not in ("event", "uncertain", "not_evaluable"):
            raise ValueError(f"source event {index} has an unsupported decision")
        context = event.get("is_disfluent")
        if context is not None and type(context) is not bool:
            raise ValueError(f"source event {index}.is_disfluent must be boolean or null")
        if known and (label is None or decision == "not_evaluable"):
            raise ValueError(f"source event {index} contradicts explicit known type evidence")
        parsed.append((start, end, TAXONOMY.index(label) if known else None,
                       context if decision == "event" else None))
    return parsed


def _arrays(cells, parsed, coverage):
    n, c = len(cells), len(TAXONOMY)
    multi_hot = np.zeros((n, c), dtype=bool)
    fully_reviewed = np.zeros(n, dtype=bool)
    unknown_type = np.zeros(n, dtype=bool)
    unknown_context = np.zeros((n, c), dtype=bool)
    context_multi_hot = np.zeros((n, c, 2), dtype=bool)
    event_indices = []
    for row, cell in enumerate(cells):
        start, end = cell["owned_start_ms"], cell["owned_end_ms"]
        fully_reviewed[row] = any(left <= start and end <= right for left, right in coverage)
        indices = []
        for index, (left, right, label, context) in enumerate(parsed):
            if max(start, left) >= min(end, right):
                continue  # Half-open endpoint contact is not positive overlap.
            indices.append(index)
            if label is None:
                unknown_type[row] = True
            else:
                multi_hot[row, label] = True
                if context is None:
                    unknown_context[row, label] = True
                else:
                    context_multi_hot[row, label, int(context)] = True
        event_indices.append(indices)
    # Known positives survive holes; unknown other labels never become negatives.
    fully_reviewed &= ~unknown_type
    type_mask = multi_hot | fully_reviewed[:, None]
    counts = multi_hot.sum(axis=1)
    type_ambiguity = (counts > 1) | unknown_type
    softmax_mask = fully_reviewed & ~type_ambiguity
    softmax_target = np.full(n, -1, dtype=np.int64)
    softmax_target[softmax_mask & (counts == 0)] = 0
    for row in np.flatnonzero(softmax_mask & (counts == 1)):
        softmax_target[row] = int(np.flatnonzero(multi_hot[row])[0]) + 1
    context_counts = context_multi_hot.sum(axis=2)
    context_conflict = context_counts > 1
    context_mask = (multi_hot & fully_reviewed[:, None] & (context_counts == 1)
                    & ~unknown_context)
    context_target = np.full((n, c), -1, dtype=np.int8)
    rows, columns = np.nonzero(context_mask)
    context_target[rows, columns] = context_multi_hot[rows, columns, 1].astype(np.int8)
    return dict(
        multi_hot=multi_hot, type_mask=type_mask, fully_reviewed=fully_reviewed,
        type_ambiguity=type_ambiguity, softmax_target=softmax_target,
        softmax_mask=softmax_mask, context_multi_hot=context_multi_hot,
        context_conflict=context_conflict, context_target=context_target,
        context_mask=context_mask, event_indices=event_indices,
    )


def project_targets(duration_ms, events, reviewed_coverage, *, cells=None) -> dict:
    """Project original human evidence without clipping it or creating unknown negatives.

    Binary type targets are multi_hot, contextual targets are conditional on type,
    and the optional single softmax is masked when mutually incompatible. Arrays
    retain partial positive evidence even when the corresponding head is masked.
    """
    grid = decision_cells(duration_ms) if cells is None else deepcopy(cells)
    validate_temporal_grid(grid, duration_ms)
    duration = integer_ms(duration_ms, "duration_ms")
    parsed = _events(events, duration)
    coverage = _coverage(reviewed_coverage, duration)
    return dict(
        target_contract=TARGET_CONTRACT, temporal_contract=TEMPORAL_CONTRACT,
        taxonomy_version=TAXONOMY_VERSION, media_duration_ms=duration,
        type_labels=list(TAXONOMY), softmax_labels=[BACKGROUND, *TAXONOMY],
        context_labels=list(_CONTEXT_LABELS), requires_masked_training=True,
        cells=evidence_windows(grid), source_events=deepcopy(list(events)),
        reviewed_coverage=coverage, **_arrays(grid, parsed, coverage),
    )


def validate_targets(projected) -> int:
    """Verify schema, aligned masks, and exact derivation from retained human evidence.

    This deliberately rejects arrays whose shape looks correct but which silently
    invent negatives or erase positives. It does not certify human annotation truth.
    """
    if not isinstance(projected, Mapping):
        raise ValueError("targets must be a mapping")
    required = (
        "media_duration_ms", "cells", "source_events", "reviewed_coverage",
        "event_indices", *_BOOL_ARRAYS, *_INT_ARRAYS,
    )
    missing = [field for field in required if field not in projected]
    if missing:
        raise ValueError(f"missing target fields: {', '.join(missing)}")
    expected = project_targets(projected["media_duration_ms"], projected["source_events"],
                               projected["reviewed_coverage"], cells=projected["cells"])
    for field in ("target_contract", "temporal_contract", "taxonomy_version", "type_labels",
                  "softmax_labels", "context_labels", "requires_masked_training",
                  "reviewed_coverage", "event_indices", "cells"):
        if projected.get(field) != expected[field]:
            raise ValueError(f"{field} disagrees with the registered target contract")
    if projected["requires_masked_training"] is not True:
        raise ValueError("requires_masked_training must explicitly be true")
    for field in (*_BOOL_ARRAYS, *_INT_ARRAYS):
        actual = projected[field]
        dtype = np.bool_ if field in _BOOL_ARRAYS else np.integer
        if (not isinstance(actual, np.ndarray) or not np.issubdtype(actual.dtype, dtype)
                or actual.shape != expected[field].shape
                or not np.array_equal(actual, expected[field])):
            raise ValueError(f"{field} has invalid shape, dtype, eligibility, or evidence values")
    return len(projected["cells"])
