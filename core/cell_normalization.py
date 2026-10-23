"""V2-only per-measure statistics; never reinterpret legacy normalization."""
import numpy as np
from core.cell_contracts import NUMERIC, effective_masks, input_evaluable, validate_cell_features


def validate_cell_stats(stats):
    widths = dict(ac=5, pros=8, ling=12, kin=12)
    if not isinstance(stats, dict) or set(stats) != set(widths):
        raise ValueError("v2 statistics must include exactly all four feature groups")
    for key, width in widths.items():
        if not isinstance(stats[key], dict) or set(stats[key]) != {"mean", "std", "count"}:
            raise ValueError(f"invalid statistics fields for {key}")
        mean, std, count = (np.asarray(stats[key][name]) for name in ("mean", "std", "count"))
        if (any(value.shape != (width,) for value in (mean, std, count))
                or not np.issubdtype(mean.dtype, np.number) or not np.issubdtype(std.dtype, np.number)
                or not np.issubdtype(count.dtype, np.integer) or np.any(count < 0)
                or not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0)
                or np.any(mean[count == 0] != 0) or np.any(std[count == 0] != 1)):
            raise ValueError(f"invalid v2 statistics for {key}")


def fit_cell_stats(data, targets, indices=None, mode="trimodal"):
    n = validate_cell_features(data)
    selected = np.zeros(n, bool)
    selected[np.arange(n) if indices is None else np.asarray(indices, int)] = True
    selected &= input_evaluable(data, mode)
    selected &= np.asarray(targets["type_mask"]).any(1) | np.asarray(targets["context_mask"]).any(1)
    masks = effective_masks(data, mode)
    stats = {}
    for key in NUMERIC:
        width = data[key].shape[-1]
        x, mask = data[key][selected].reshape(-1, width), masks[key][selected].reshape(-1, width)
        mean, std, count = np.zeros(width), np.ones(width), np.zeros(width, np.int64)
        for column in range(width):
            observed = x[mask[:, column], column]  # Select BEFORE subtraction/squaring.
            if len(observed):
                mean[column], std[column], count[column] = observed.mean(), max(observed.std(), 1e-6), len(observed)
        stats[key] = dict(mean=mean.tolist(), std=std.tolist(), count=count.tolist())
    # NumPy arrays for local callers; the checkpoint serializer stores plain lists.
    return {key: {field: np.asarray(value) for field, value in fields.items()} for key, fields in stats.items()}


def apply_cell_stats(data, stats, mode="trimodal", modalities=None):
    validate_cell_features(data)
    validate_cell_stats(stats)
    masks = effective_masks(data, mode, modalities)
    result = dict(data)
    for key in NUMERIC:
        fields = stats[key]
        mean, std, count = (np.asarray(fields[name]) for name in ("mean", "std", "count"))
        if (mean.shape != (data[key].shape[-1],) or std.shape != mean.shape or count.shape != mean.shape
                or not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0) or np.any(count < 0)):
            raise ValueError(f"invalid v2 statistics for {key}")
        mask = masks[key] & (count > 0)
        safe = np.where(mask, data[key], mean)  # NaN * 0 is still NaN: never multiply.
        result[key] = np.where(mask, (safe - mean) / std, 0).astype(np.float32)
        result[key + "_mask"] = mask
    return result
