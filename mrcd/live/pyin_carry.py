"""pYIN with a resumable Viterbi forward pass (used by ``live_cfg_v3``).

Why this exists
---------------
``librosa.pyin`` decodes the whole signal with a global Viterbi pass whose initial state distribution is
uniform. The incremental tracker re-decodes ``[commit - L, now]`` on every tick, so each decode starts from a
uniform prior at an arbitrary frame. The observation probabilities are bit-identical regardless of where
the slice starts (verified, ``scripts/live_parity/diagnose_seed23.py``), but in long low-energy stretches
(room noise) the voiced and unvoiced hypotheses keep a constant score gap, so the optimal path does NOT
coalesce and the decoded voicing of a later frame depends on where the decode started, no matter how large
the left margin is. Seed 23 of the 10.5 min stress session flips 13 frames from a start 2.5 s into a noise
stretch; starting at 292 s or earlier gives the offline answer, starting at 293 s or later does not, and
the right end of the slice is irrelevant.

Fix: keep the exact forward scores (``delta``) of the last committed frame and start the next decode from
them instead of from the uniform prior. The forward recursion is a deterministic function of the
observations seen so far, so ``delta`` carried across ticks is bit-equal to the one a single whole-file pass
computes, and the back-pointers of every later frame are the same. What stays inexact is only the
dependence on the FUTURE (the path is backtracked from the end of the data available at commit time).
"""

from __future__ import annotations

import types

import numba
import numpy as np


@numba.njit(cache=True)
def _forward(
    log_prob, log_trans, pred_ptr, pred_idx, delta0, has_prior, log_p_init
):  # pragma: no cover
    """Same arithmetic as ``librosa.sequence._viterbi`` plus an optional resume point.

    ``log_prob``: (T, S). If ``has_prior`` the first row is NOT used for scoring: ``delta0`` is the value of the
    previous frame and row 0 is the first new frame. Returns (state (T,), value (T, S), ptr (T, S)).
    """
    n_steps, n_states = log_prob.shape
    state = np.zeros(n_steps, dtype=np.uint16)
    value = np.zeros((n_steps, n_states), dtype=np.float64)
    ptr = np.zeros((n_steps, n_states), dtype=np.uint16)
    prev = np.zeros(n_states, dtype=np.float64)
    if has_prior:
        prev[:] = delta0
        t0 = 0
    else:
        value[0] = log_prob[0] + log_p_init
        prev[:] = value[0]
        t0 = 1
    for t in range(t0, n_steps):
        for j in range(n_states):
            best_cost = -np.inf
            for q in range(pred_ptr[j], pred_ptr[j + 1]):
                k = pred_idx[q]
                cost = prev[k] + log_trans[k, j]
                if cost > best_cost:
                    ptr[t, j] = k
                    best_cost = cost
            value[t, j] = log_prob[t, j] + best_cost
        prev[:] = value[t]
    state[n_steps - 1] = np.argmax(value[n_steps - 1])
    for t in range(n_steps - 2, -1, -1):
        state[t] = ptr[t + 1, state[t + 1]]
    return state, value, ptr


class _SequenceProxy:
    """Stands in for ``librosa.sequence`` inside a PRIVATE copy of ``librosa.pyin``: every attribute is delegated to the
    real module except ``viterbi``, which records its arguments. The real module is never mutated, so concurrent
    ``librosa.pyin`` calls in other threads (``extract_acoustic``...) are unaffected."""

    def __init__(self, cap: dict):
        self._cap = cap

    def __getattr__(self, name):
        from librosa import sequence

        return getattr(sequence, name)

    def viterbi(self, prob, transition, p_init=None, transition_min_prob=None, **kw):
        self._cap.update(
            prob=prob, transition=transition, p_init=p_init, tmin=transition_min_prob
        )
        return np.zeros((prob.shape[0], prob.shape[-1]), np.uint16)


def _capturing_pyin(cap: dict):
    """A copy of ``librosa.pyin`` (same code object, same defaults) whose module-level ``sequence`` is a proxy."""
    from librosa.core import pitch

    src = pitch.pyin
    g = dict(vars(pitch))  # snapshot; the proxy replaces only the name ``sequence``
    g["sequence"] = _SequenceProxy(cap)
    fn = types.FunctionType(src.__code__, g, src.__name__, src.__defaults__, src.__closure__)
    fn.__kwdefaults__ = dict(src.__kwdefaults__ or {})
    return fn


def pyin_resumable(
    y, *, sr, fmin, fmax, frame_length, hop_length, carry=None, carry_index=None
):
    """``librosa.pyin`` equivalent that can resume the Viterbi pass.

    ``carry``: forward scores (S,) of frame ``carry_index - 1`` of the slice ``y`` (they were produced by an earlier
    call), or None for the uniform prior. Returns ``(f0, voiced_prob, value_rows)`` where frames with index
    < ``carry_index`` get NaN f0 (not decoded) and ``value_rows`` is the forward matrix of the decoded frames
    (row r is frame ``carry_index + r``). ``y`` must be 1-D (mono); other shapes raise ``ValueError``.
    """
    y = np.asarray(y)
    if y.ndim != 1:  # librosa.pyin accepts multichannel; this decoder is mono only (fail clearly, not in numba)
        raise ValueError(f"pyin_resumable supports mono audio only: expected a 1-D array, got shape {y.shape}")
    cap: dict = {}
    _, _, vprob = _capturing_pyin(cap)(
        y,
        fmin=fmin,
        fmax=fmax,
        sr=sr,
        frame_length=frame_length,
        hop_length=hop_length,
        center=True,
    )
    if "prob" not in cap:  # librosa changed its internals: fail loudly instead of decoding garbage
        raise RuntimeError("librosa.pyin did not call sequence.viterbi; pyin_carry needs updating")
    prob = cap["prob"][0]  # (S, T)
    n_states = prob.shape[0]
    n_bins = n_states // 2
    eps = np.finfo(prob.dtype).tiny
    log_trans = np.log(cap["transition"] + eps)
    log_prob = np.log(prob + eps).T.copy()  # (T, S)
    log_p_init = np.log(cap["p_init"] + eps)
    tmin = cap["tmin"]
    thr = np.log(tmin + eps) if tmin else -np.inf
    preds = [
        np.flatnonzero(log_trans[:, j] >= thr) for j in range(n_states)
    ]  # ascending, as librosa
    pred_ptr = np.concatenate([[0], np.cumsum([len(p) for p in preds])]).astype(
        np.int64
    )
    pred_idx = np.concatenate(preds).astype(np.int64)
    if carry is None:
        state, value, _ = _forward(
            log_prob,
            log_trans,
            pred_ptr,
            pred_idx,
            np.zeros(n_states),
            False,
            log_p_init,
        )
        first = 0
    else:
        first = int(carry_index)
        state, value, _ = _forward(
            log_prob[first:],
            log_trans,
            pred_ptr,
            pred_idx,
            np.asarray(carry, np.float64),
            True,
            log_p_init,
        )
    freqs = fmin * 2 ** (np.arange(n_bins) / (12 * int(np.ceil(1.0 / 0.1))))
    f0 = np.full(prob.shape[1], np.nan)
    st = state.astype(np.int64)
    voiced = st < n_bins
    seg = freqs[st % n_bins]
    seg[~voiced] = np.nan
    f0[first:] = seg
    return f0, vprob, value
