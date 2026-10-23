"""Assemble owned cells from persisted real signal timestamps, without ASR/network."""
from copy import deepcopy
import numpy as np
from core.cell_contracts import SHAPES, cell_feature_contract, validate_cell_features
from core.extractors.linguistic import Word, normalize, word_features
from core.constants import FUNCTION_WORDS
from core.temporal import decision_cells, evidence_windows


def _track(raw, times_key, values_key, width):
    times = np.asarray(raw[times_key], dtype=np.float64) * 1000
    values = np.asarray(raw[values_key], dtype=np.float64)
    if (times.ndim != 1 or values.shape != (len(times), width)
            or not np.isfinite(times).all() or np.any(np.diff(times) <= 0)):
        raise ValueError(f"invalid persisted {times_key}/{values_key} track")
    return times, values


def _slope(times, values):
    x = times / 1000
    return float(np.dot(x - x.mean(), values - values.mean()) / np.square(x - x.mean()).sum())


def _prosody(times, frames, f0, cell, center):
    out, mask = np.zeros(8, np.float32), np.zeros(8, bool)
    lo, hi = cell["observed_start_ms"], cell["observed_end_ms"]
    own_lo, own_hi = cell["owned_start_ms"], cell["owned_end_ms"]
    valid = (times >= lo) & (times < hi)
    t, x, hz = times[valid], frames[valid], f0[valid]
    vad_valid = np.isfinite(x[:, 3])
    silent = vad_valid & (x[:, 3] <= .5)
    runs, start = [], None
    for i in range(len(t) + 1):
        continuing = i < len(t) and silent[i]
        if start is not None and (not continuing or t[i] - t[i - 1] > 10.001):
            runs.append((start, i))
            start = None
        if continuing and start is None:
            start = i
    candidates = [(max(0, min(own_hi, t[b - 1] + 10) - max(own_lo, t[a])), a, b) for a, b in runs]
    candidates = [r for r in candidates if r[0] > 0]
    if not candidates:
        owned = (t >= own_lo) & (t < own_hi)
        expected = np.arange(own_lo, own_hi, 10)
        if (len(t[owned]) == len(expected) and np.allclose(t[owned], expected, atol=1e-4)
                and vad_valid[owned].all()):
            mask[0] = True  # Observed no silence, not unavailable silence.
        return out, mask
    _, a, b = sorted(candidates, key=lambda item: (-item[0], t[item[1]]))[0]
    start_ms, end_ms = t[a], min(t[b - 1] + 10, hi)
    left = a > 0 and vad_valid[a - 1] and x[a - 1, 3] > .5 and t[a] - t[a - 1] <= 10.001
    right = b < len(t) and vad_valid[b] and x[b, 3] > .5 and t[b] - t[b - 1] <= 10.001
    if left and right:
        out[0], mask[0] = (end_ms - start_ms) / 1000, True
    if left:
        out[1], mask[1] = (start_ms - own_lo) / 1000, True
    voiced = np.isfinite(hz) & (hz > 0)
    pre = (t >= start_ms - 300) & (t < start_ms) & voiced
    post = (t >= end_ms) & (t < end_ms + 300) & voiced
    if left and pre.sum() >= 3:
        pitch = 12 * np.log2(hz[pre] / 100)
        out[2:5] = [_slope(t[pre], pitch), np.ptp(pitch), pitch.mean() - center]
        mask[2:5] = True
    if left and right and pre.sum() >= 3 and post.sum() >= 3:
        out[5], mask[5] = 12 * np.log2(hz[post] / 100).mean() - 12 * np.log2(hz[pre] / 100).mean(), True
    rms_pre = (t >= start_ms - 300) & (t < start_ms) & np.isfinite(x[:, 0])
    if left and rms_pre.sum() >= 3:
        out[6], mask[6] = _slope(t[rms_pre], x[rms_pre, 0]), True
    speech = vad_valid & (x[:, 3] > .5) & np.isfinite(x[:, 0])
    if left and rms_pre.sum() >= 3 and speech.sum() >= 5:
        out[7], mask[7] = x[rms_pre, 0].mean() - np.median(x[speech, 0]), True
    return out, mask


def _words(words, duration):
    result = []
    for index, raw in enumerate(words):
        start = raw.get("start_ms", raw.get("start", 0) * 1000)
        end = raw.get("end_ms", raw.get("end", 0) * 1000)
        if not np.isfinite([start, end]).all() or not 0 <= start < end <= duration:
            raise ValueError("word interval outside actual media")
        text = str(raw["text"])
        probability = raw.get("prob")
        prob_known = probability is not None and np.isfinite(probability)
        if prob_known and not 0 <= probability <= 1:
            raise ValueError("word probability outside [0,1]")
        result.append((start, end, index, Word(text, normalize(text), start / 1000, end / 1000,
                       float(probability) if prob_known else 0., raw.get("punct_after", "")), prob_known))
    return sorted(result, key=lambda w: (w[0], w[1], w[2]))


def build_cell_features(raw, words, duration_ms):
    cells = evidence_windows(decision_cells(duration_ms))
    n = len(cells)
    data = {key: np.zeros((n, *shape), dtype=np.float32) for key, shape in SHAPES.items()}
    for key in SHAPES:
        if key.endswith("_mask") or key in ("ac_support", "kin_support", "ac_target", "lmask", "ltarget", "ktarget", "has_video"):
            data[key] = np.zeros((n, *SHAPES[key]), bool)
    data["word_indices"] = np.full((n, 40), -1, np.int64)
    data["real_frame_count"] = np.zeros(n, np.int64)
    data["word_source_count"] = np.full(n, len(words), np.int64)
    for key in ("media_duration_ms", "owned_start_ms", "owned_end_ms", "nominal_center_ms"):
        data[key] = np.zeros(n, np.int64)
    for key in ("ac_pos", "kpos", "lpos", "word_intervals"):
        data[key] = np.zeros((n, *SHAPES[key]), np.float64)
    times, frames = _track(raw, "ac_times", "ac_frames", 5)
    f0 = np.asarray(raw["f0"], float)
    if f0.shape != times.shape:
        raise ValueError("persisted raw f0 must align with ac_times")
    real = (times >= 0) & (times < duration_ms)
    times, frames, f0 = times[real], frames[real], f0[real]
    voiced = np.isfinite(f0) & (f0 > 0)
    pitch = np.zeros(len(times))
    pitch[voiced] = 12 * np.log2(f0[voiced] / 100)
    center = float(np.median(pitch[voiced])) if voiced.any() else 0.
    observed = np.isfinite(frames)
    observed[:, 1] = voiced
    frames = np.where(observed, frames, 0.)
    frames[:, 1] = pitch
    adjacent = np.zeros(len(times), bool)
    adjacent[1:] = voiced[1:] & voiced[:-1] & np.isclose(np.diff(times), 10, atol=1e-4)
    frames[:, 4] = np.r_[0., np.diff(pitch)] if len(pitch) else np.zeros(0)
    observed[:, 4] = adjacent
    # Prosody must still see unavailable VAD/RMS as missing, not sanitized silence.
    pros_frames = np.where(observed, frames, np.nan)
    kt, kv = _track(raw, "k_times", "k_vectors", 12)
    detected = np.asarray(raw["k_detected"])
    if detected.dtype != bool or detected.shape != kt.shape:
        raise ValueError("persisted k_detected must be an aligned real-detection mask")
    real = (kt >= 0) & (kt < duration_ms) & detected
    kt, kv = kt[real], kv[real]
    candidates = _words(words, duration_ms)
    for row, cell in enumerate(cells):
        req, end, c = cell["requested_start_ms"], cell["requested_end_ms"], cell["nominal_center_ms"]
        own_lo, own_hi = cell["owned_start_ms"], cell["owned_end_ms"]
        for key in ("media_duration_ms", "owned_start_ms", "owned_end_ms", "nominal_center_ms"):
            data[key][row] = cell[key]
        indices = np.flatnonzero((times >= req) & (times < end))
        slots = np.floor((times[indices] - req) / 10 + 1e-6).astype(int)
        if len(np.unique(slots)) != len(slots):
            raise ValueError("multiple persisted acoustic samples occupy one 10ms slot")
        data["ac"][row, slots] = np.where(observed[indices], frames[indices], 0)
        data["ac_mask"][row, slots] = observed[indices]
        data["ac_mask"][row, slots, 4] &= times[np.maximum(indices - 1, 0)] >= req
        data["ac_support"][row, slots] = True
        data["ac_target"][row, slots] = (times[indices] >= own_lo) & (times[indices] < own_hi)
        data["ac_pos"][row, slots] = (times[indices] - c) / 1000
        data["pros"][row], data["pros_mask"][row] = _prosody(times, pros_frames, f0, cell, center)
        indices = np.flatnonzero((kt >= req) & (kt < end))
        slots = np.floor((kt[indices] - req) / 100 + 1e-6).astype(int)
        if len(np.unique(slots)) != len(slots):
            raise ValueError("multiple real visual frames occupy one 100ms slot")
        data["kin"][row, slots] = np.where(np.isfinite(kv[indices]), kv[indices], 0)
        data["kin_mask"][row, slots] = np.isfinite(kv[indices])
        data["kin_support"][row, slots] = True
        data["kpos"][row, slots] = (kt[indices] - c) / 1000
        data["ktarget"][row, slots] = (kt[indices] >= own_lo) & (kt[indices] < own_hi)
        data["real_frame_count"][row] = len(indices)
        data["has_video"][row] = len(indices) >= 24
        context = [w for w in candidates if c - 5000 <= (w[0] + w[1]) / 2 < c + 5000
                   or max(own_lo, w[0]) < min(own_hi, w[1])]
        owned = [i for i, w in enumerate(context) if max(own_lo, w[0]) < min(own_hi, w[1])]
        if len(owned) > 40:
            raise ValueError(f"target_word_overflow: cell {row} has {len(owned)} owned words")
        remaining = sorted(set(range(len(context))) - set(owned),
                           key=lambda i: (abs((context[i][0] + context[i][1]) / 2 - c), *context[i][:3]))
        selected = sorted(owned + remaining[:40 - len(owned)])
        lightweight = word_features([w[3] for w in context])
        for slot, index in enumerate(selected):
            start, stop, original_index, word, prob_known = context[index]
            target = max(own_lo, start) < min(own_hi, stop)
            data["ling"][row, slot, :10] = lightweight[index]
            data["ling"][row, slot, 10] = 0 if target else (-1 if stop <= own_lo else 1)
            data["ling"][row, slot, 11] = float(word.norm in FUNCTION_WORDS and index + 1 < len(context)
                                                    and context[index + 1][0] - stop >= 250)
            data["ling_mask"][row, slot] = True
            data["ling_mask"][row, slot, 7] = prob_known
            if index == 0:
                data["ling_mask"][row, slot, [3, 8]] = False
                if not word.text.endswith("-"):
                    data["ling_mask"][row, slot, 4] = False
            if index + 1 == len(context):
                data["ling_mask"][row, slot, [9, 11]] = False
            data["lmask"][row, slot] = True
            data["ltarget"][row, slot] = target
            data["lpos"][row, slot] = ((start + stop) / 2 - c) / 1000
            data["word_indices"][row, slot] = original_index
            data["word_intervals"][row, slot] = [start, stop]
    data["feature_contract"] = cell_feature_contract()
    validate_cell_features(data)
    return data
