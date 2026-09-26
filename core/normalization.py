"""One mask-aware normalization policy shared by training and inference."""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from core.contracts import NUMERIC_FEATURES

MASK_OF = {"ling": "lmask", "kin": "kmask"}


def fit_stats(data: Mapping, indices=None) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Fit only on selected training rows and observed (not padded) tokens."""
    result = {}
    for key in NUMERIC_FEATURES:
        if key not in data:
            continue
        x = np.asarray(data[key], np.float32)
        x = x if indices is None else x[indices]
        rows = x.reshape(-1, x.shape[-1])
        mask_name = MASK_OF.get(key)
        if mask_name and mask_name in data:
            mask = np.asarray(data[mask_name], bool)
            mask = mask if indices is None else mask[indices]
            if key == "kin" and "has_video" in data:
                available = np.asarray(data["has_video"], bool)
                available = available if indices is None else available[indices]
                mask = mask & available[:, None]
            rows = rows[mask.reshape(-1)]
        rows = rows[np.isfinite(rows).all(axis=1)]
        if len(rows):
            mu, sd = rows.mean(0), rows.std(0) + 1e-6
        else:
            mu, sd = np.zeros(x.shape[-1], np.float32), np.ones(x.shape[-1], np.float32)
        result[key] = mu.astype(np.float32), sd.astype(np.float32)
    return result


def apply_stats(data: Mapping, stats: Mapping) -> dict:
    """Normalize features, keeping padded/missing positions exactly zero."""
    result = dict(data)
    for key, (mu, sd) in stats.items():
        if key not in data:
            continue
        x = (np.asarray(data[key], np.float32) - np.asarray(mu)) / np.asarray(sd)
        mask_name = MASK_OF.get(key)
        if mask_name and mask_name in data:
            mask = np.asarray(data[mask_name], bool)
            if key == "kin" and "has_video" in data:
                mask = mask & np.asarray(data["has_video"], bool)[:, None]
            x = np.where(mask[..., None], x, 0)
        result[key] = x.astype(np.float32)
    return result
