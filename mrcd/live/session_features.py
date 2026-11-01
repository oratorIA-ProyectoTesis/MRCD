"""Session-accumulated feature extraction for the live path (new module).

Why this module exists
----------------------
``window_builder.build_latest_contextual_window`` re-extracts features from the 13 s ring buffer on
every tick. That is NOT equivalent to the offline whole-file extraction: Silero VAD is a stateful
LSTM that starts from zero state at the slice start (the slice is cut mid-speech), and the speaker F0
centre is the median of the slice instead of the median of the recording (measured in
``results/paridad_20261030/PARIDAD.md``: VAD frames differ in 24/160 windows, centre differs up to
2.1 semitones, 10/160 windows change argmax).

This tracker keeps the *session* context instead, and reproduces the offline definitions:

* **VAD**: Silero is fed the session from sample 0 in 512-sample chunks (the same chunking as
  ``get_speech_timestamps``), so every probability equals the offline one. Segmentation uses
  ``get_speech_timestamps_from_probs`` (same parameters and 0.1 s rounding as
  ``extractors.acoustic``). Closed segments are frozen once the probabilities have been silent for
  ``_QUIET_CHUNKS`` chunks, which bounds the cost per tick without changing the result.
* **RMS / pYIN**: frame ``i`` of a slice that starts on a hop boundary is global frame ``lo + i``.
  Frames are recomputed over ``[commit - left_margin, now]`` and only frames at least
  ``right_margin_s`` before ``now`` are committed. RMS is local, but pYIN decodes with a Viterbi
  pass whose result at a frame depends on frames on BOTH sides. 2-D study over all 98 AMERESCO clips
  (+ an 8-clip, 4.5 min concatenation), criterion = committed frames that differ from whole-file
  extraction (results/paridad_live_20261030/margin_study_*.json): (L=1,R=3) 39 frames in 2 clips
  (max 12.3 st); (2,3) 7 frames in 1 clip; (3,3), (4,3), (3,4), (5,3) 0 frames. Defaults are
  L=3 s, R=3 s: the smallest tested pair with 0 differing frames. This is an empirical result on
  those 99 signals, NOT a proof: the Viterbi dependence is unbounded in principle. A window is ready
  ``max(5, 3 + right_margin)`` = 6 s after its start (unchanged by L).
* **Words**: session-absolute times, no slice offset (this removes the ``lpos`` off-by-one artefact
  of ``word.start - offset_s``).
* **F0 centre**: the offline centre is the median over the WHOLE file, which a live system cannot
  know. ``f0_center="causal"`` uses the median of the voiced frames committed so far (explicitly NOT
  equivalent to offline; the divergence is bounded and measured). A float fixes the centre (used to
  prove that everything else is equivalent: with the offline centre the tensors match).

Causality limits that remain by construction, not bugs: (1) the centre above, (2) the word-level
features of the last ASR word use the NEXT word (gap, bigram) that does not exist yet, (3) the
ASR text itself (not handled here).

The audio is assumed to be float32 in [-1, 1] (a microphone). The offline extractor rescales a whole
file whose peak exceeds 1.5 (assumed int16); a stream has no peak known in advance, so it is not
applied here.
"""

from __future__ import annotations

import bisect
import math

import numpy as np

from mrcd.constants import ACOUSTIC_HOP_S, CONTEXT_S, SAMPLE_RATE, STRIDE_S, WINDOW_S

_HOP = int(SAMPLE_RATE * ACOUSTIC_HOP_S)  # 160 samples
_CHUNK = 512  # Silero window at 16 kHz
_QUIET_PROB = 0.35  # == default neg_threshold (0.5 - 0.15)
_QUIET_CHUNKS = (
    32  # ~1 s of probabilities below _QUIET_PROB makes the state machine idle
)
#: seconds of lookahead the linguistic context of a window needs (CONTEXT_S minus the 5 s look-back)
CONTEXT_AHEAD_S = 5.0

# Same Silero parameters as mrcd.extractors.acoustic._silero_vad
_VAD_KW = dict(
    min_silence_duration_ms=150, min_speech_duration_ms=120, speech_pad_ms=30
)


class WindowNotReady(RuntimeError):
    """The requested window needs frames that are not committed yet."""


class IncrementalSegmenter:
    """Speech segmentation over a growing list of Silero probabilities.

    Equivalent to calling ``get_speech_timestamps_from_probs`` on all probabilities, but only the
    part after the last *quiet* point is re-segmented. A quiet point (>= ``_QUIET_CHUNKS`` chunks below
    the negative threshold) leaves the state machine idle, and neighbouring segments are more than
    ``2 * speech_pad`` apart, so the segments before it can never change.
    """

    def __init__(self) -> None:
        self.probs: list[float] = []
        self._cut = 0  # index of the first probability not yet frozen
        self._frozen: list[
            tuple[int, int]
        ] = []  # sample ranges (already padded), before _cut

    def extend(self, new_probs) -> None:
        self.probs.extend(float(p) for p in new_probs)

    def segments_seconds(self, total_samples: int | None = None) -> list[tuple[float, float]]:
        """Segments so far. ``total_samples`` (session close only) is the true audio length: the
        offline path clamps the last segment to ``len(audio)``, not to ``len(probs) * 512``."""
        from silero_vad.utils_vad import get_speech_timestamps_from_probs

        base = self._cut * _CHUNK
        extra = {} if total_samples is None else {"audio_length_samples": total_samples - base}
        tail = get_speech_timestamps_from_probs(
            self.probs[self._cut :], sampling_rate=SAMPLE_RATE, return_seconds=False, **_VAD_KW, **extra
        )
        live = [(s["start"] + base, s["end"] + base) for s in tail]
        allsegs = [*self._frozen, *live]
        self._freeze(live)
        total_s = (len(self.probs) * _CHUNK if total_samples is None else total_samples) / SAMPLE_RATE
        return [
            (max(round(a / SAMPLE_RATE, 1), 0), min(round(b / SAMPLE_RATE, 1), total_s))
            for a, b in allsegs
        ]

    def _freeze(self, live: list[tuple[int, int]]) -> None:
        """Freeze ``live`` when the last ``_QUIET_CHUNKS`` probabilities are all quiet.

        Quiet stretch starts at chunk ``q0``. A segment still open there closes within ~6 chunks
        (150 ms of silence), so from ``q0 + 8`` the offline state machine is idle and no earlier
        segment can change (the next one starts >= 1 s later, far beyond ``2 * speech_pad``).
        """
        n = len(self.probs)
        if n - _QUIET_CHUNKS < self._cut:
            return
        if not all(p < _QUIET_PROB for p in self.probs[n - _QUIET_CHUNKS :]):
            return
        self._frozen.extend(live)
        self._cut = n - _QUIET_CHUNKS + 8


class SessionFeatureTracker:
    """Accumulates the session and produces windows with the offline feature definitions."""

    def __init__(
        self,
        *,
        f0_center: str | float = "causal",
        left_margin_s: float = 3.0,
        right_margin_s: float = 3.0,
        vad_model=None,
        viterbi_carry: bool = False,
    ) -> None:
        if not (f0_center == "causal" or isinstance(f0_center, (int, float))):
            raise ValueError("f0_center must be 'causal' or a float (semitones)")
        if left_margin_s < 3.0 or right_margin_s < 3.0:
            import warnings

            warnings.warn(
                "margins below 3 s left / 3 s right were NOT frame-exact on all 98 AMERESCO clips "
                "(see margin_study_*.json); expect small F0 differences versus offline",
                RuntimeWarning,
                stacklevel=2,
            )
        if left_margin_s < 0.5 or right_margin_s < 1.0:
            raise ValueError(
                "margins below 0.5 s (left) / 1.0 s (right) break equality with whole-file "
                "extraction (RMS edge effects); 3 s on the right is the measured safe default (pYIN Viterbi)"
            )
        self.f0_center_mode = f0_center
        # viterbi_carry=True (live_cfg_v3): resume the pYIN Viterbi pass from the exact forward scores of the last
        # committed frame instead of a uniform prior at the slice start; see mrcd/live/pyin_carry.py
        self._carry_on = bool(viterbi_carry)
        self._carry = None  # forward scores (S,) of frame n_commit - 1
        self._last_f0 = np.float32("nan")  # f0 (Hz) of the last committed frame (for df0 of the next one)
        self._left = int(round(left_margin_s / ACOUSTIC_HOP_S))
        self._right = int(round(right_margin_s / ACOUSTIC_HOP_S))
        self._vad_model = vad_model
        self._seg = IncrementalSegmenter()
        self._vad_tail = np.zeros(0, np.float32)  # samples not yet forming a full chunk
        self._audio = np.zeros(0, np.float32)  # kept tail of the session
        self._audio_origin = (
            0  # absolute sample index of self._audio[0] (multiple of _HOP)
        )
        self._n_samples = 0
        self._n_commit = 0  # frames [0, n_commit) are final
        self._blocks: list[np.ndarray] = []
        self._frames_cache: np.ndarray | None = None
        self._voiced: list[float] = []
        self._closed = False

    # ------------------------------------------------------------------ input
    @property
    def duration_s(self) -> float:
        return self._n_samples / SAMPLE_RATE

    @property
    def committed_s(self) -> float:
        return self._n_commit * ACOUSTIC_HOP_S

    @property
    def closed(self) -> bool:
        """True after :meth:`finalize`: the session is over and no more audio is accepted."""
        return self._closed

    def append(self, audio: np.ndarray, *, commit: bool = True) -> None:
        if self._closed:
            raise RuntimeError("tracker is finalized; create a new SessionFeatureTracker for a new session")
        chunk = np.asarray(audio, dtype=np.float32).reshape(-1)
        if not len(chunk):
            return
        if not np.isfinite(chunk).all():
            raise ValueError("audio must be finite")
        self._audio = np.concatenate([self._audio, chunk])
        self._n_samples += len(chunk)
        self._feed_vad(chunk)
        if commit:
            self.update()

    def _model(self):
        if self._vad_model is None:
            from silero_vad import load_silero_vad

            self._vad_model = load_silero_vad(onnx=True)
            self._vad_model.reset_states()
        return self._vad_model

    def _feed_vad(self, chunk: np.ndarray) -> None:
        import torch

        buf = np.concatenate([self._vad_tail, chunk])
        n = len(buf) // _CHUNK
        model = self._model()
        probs = [
            float(
                model(
                    torch.from_numpy(buf[i * _CHUNK : (i + 1) * _CHUNK].copy()),
                    SAMPLE_RATE,
                ).item()
            )
            for i in range(n)
        ]
        self._vad_tail = buf[n * _CHUNK :]
        self._seg.extend(probs)

    # --------------------------------------------------------------- frames
    def finalize(self) -> None:
        """Close the session: commit EVERY remaining frame (no right margin) and accept no more audio.

        Future context used: none (the stream ended), so the tail frames are computed exactly as the
        offline extraction computes them (the last partial Silero chunk is zero-padded like
        ``get_speech_timestamps`` does and the last segment is clamped to the true audio length).
        Idempotent. Frames committed before this call are never rewritten.
        """
        if self._closed:
            return
        if len(self._vad_tail):  # offline pads the final partial chunk with zeros
            self._feed_vad(np.zeros(_CHUNK - len(self._vad_tail), np.float32))
        self.update(final=True)
        self._closed = True

    def update(self, *, final: bool = False) -> None:
        """Commit every frame that is at least ``right_margin_s`` old (call once per tick).

        ``final=True`` (used by :meth:`finalize`) commits up to the last frame."""
        import librosa

        from mrcd.extractors.acoustic import AcousticTrack

        total_frames = 1 + self._n_samples // _HOP
        commit_end = total_frames - (0 if final else self._right)
        if commit_end <= self._n_commit:
            return
        lo = max(self._n_commit - self._left, 0)
        a0 = lo * _HOP - self._audio_origin
        seg_audio = self._audio[a0:]
        rms = librosa.feature.rms(
            y=seg_audio, frame_length=4 * _HOP, hop_length=_HOP, center=True
        )[0]
        if self._carry_on:
            from mrcd.live.pyin_carry import pyin_resumable

            first = self._n_commit - lo if self._carry is not None else 0
            f0, vprob, value = pyin_resumable(
                seg_audio,
                fmin=60.0,
                fmax=400.0,
                sr=SAMPLE_RATE,
                frame_length=1024,
                hop_length=_HOP,
                carry=self._carry,
                carry_index=first,
            )
            if first > 0:
                f0[first - 1] = self._last_f0  # df0 of the first new frame needs its predecessor
        else:
            f0, _, vprob = librosa.pyin(
                seg_audio,
                fmin=60.0,
                fmax=400.0,
                sr=SAMPLE_RATE,
                frame_length=1024,
                hop_length=_HOP,
                center=True,
            )
        n = commit_end - lo
        times = np.arange(lo, commit_end) * ACOUSTIC_HOP_S
        vad = np.zeros(n, dtype=np.int8)
        for a, b in self._seg.segments_seconds(self._n_samples if final else None):
            if b < times[0] or a > times[-1]:
                continue
            vad[(times >= a) & (times < b)] = 1
        track = AcousticTrack(
            times,
            rms[:n].astype(np.float32),
            f0[:n].astype(np.float32),
            np.nan_to_num(vprob[:n]).astype(np.float32),
            vad,
        )
        rows = track.frame_matrix()[self._n_commit - lo :]
        self._blocks.append(rows)
        self._frames_cache = None
        self._voiced.extend(rows[:, 1][rows[:, 1] != 0.0].tolist())
        if self._carry_on:
            self._carry = value[commit_end - 1 - lo - first].copy()
            self._last_f0 = np.float32(f0[n - 1])
        self._n_commit = commit_end
        # drop audio that neither the next recompute nor the VAD tail needs
        keep_from = max((self._n_commit - self._left) * _HOP, 0)
        drop = keep_from - self._audio_origin
        if drop > 0:
            self._audio = self._audio[drop:]
            self._audio_origin = keep_from

    def frame_matrix(self) -> np.ndarray:
        """Committed frames (T, 5), uncentred, global time = row * 10 ms."""
        if self._frames_cache is None:
            self._frames_cache = (
                np.concatenate(self._blocks)
                if self._blocks
                else np.zeros((0, 5), np.float32)
            )
            self._blocks = [self._frames_cache] if len(self._frames_cache) else []
        return self._frames_cache

    def f0_center_st(self) -> float:
        """Median voiced F0 in semitones (0.0 when < 10 voiced frames, as offline)."""
        if self.f0_center_mode != "causal":
            return float(self.f0_center_mode)
        return float(np.median(self._voiced)) if len(self._voiced) >= 10 else 0.0

    # --------------------------------------------------------------- windows
    def latest_start(self) -> float | None:
        """Newest 0.5 s-grid window start whose 10 s linguistic context is complete, or None."""
        horizon = min(self.duration_s - CONTEXT_AHEAD_S, self.committed_s - WINDOW_S)
        s = math.floor(horizon / STRIDE_S + 1e-9) * STRIDE_S
        if s < 0:
            return None
        return float(s)

    def window(
        self,
        start_s: float,
        words,
        *,
        context: bool = True,
        include_pros: bool = True,
        ling_dim: int | None = None,
    ) -> dict[str, np.ndarray]:
        """Tensors for the window starting at ``start_s`` (session seconds)."""
        from mrcd.dataset import acoustic_windows, center_f0, linguistic_windows
        from mrcd.extractors.linguistic import word_features
        from mrcd.extractors.prosody import prosody_windows

        if start_s + WINDOW_S > self.committed_s + 1e-9:
            raise WindowNotReady(
                f"window [{start_s:.2f}, {start_s + WINDOW_S:.2f}] needs frames up to "
                f"{start_s + WINDOW_S:.2f} s but only {self.committed_s:.2f} s are committed"
            )
        frames = self.frame_matrix()
        a = int(round(start_s / ACOUSTIC_HOP_S))
        seg = frames[a : a + int(round(WINDOW_S / ACOUSTIC_HOP_S))]
        c = self.f0_center_st()
        zero = np.zeros(1)
        out: dict[str, np.ndarray] = {}
        local = _local_words(words, start_s, context)
        feats = word_features(local)
        ling, lpos, lmask = linguistic_windows(
            [w.to_dict() for w in local],
            feats,
            np.asarray([start_s], np.float64),
            context=context,
        )
        if ling_dim is not None:
            if ling_dim > ling.shape[-1]:
                raise ValueError(
                    f"checkpoint expects ling={ling_dim}, extractor gives {ling.shape[-1]}"
                )
            ling = ling[:, :, :ling_dim]
        out["ac"] = acoustic_windows(center_f0(seg, c), zero)
        out["ling"], out["lpos"], out["lmask"] = ling, lpos, lmask
        if include_pros:
            out["pros"] = prosody_windows(seg, zero, c)
        return out


def _local_words(words, start_s: float, context: bool):
    """Words that can influence the window, plus one neighbour on each side.

    ``word_features`` of word ``i`` only looks at ``i-1`` and ``i+1``, so trimming to the context
    range plus one neighbour leaves every in-range feature unchanged and keeps the cost O(context).
    """
    if not words:
        return []
    pre = 5.0 if context else 0.0
    span = CONTEXT_S if context else WINDOW_S
    lo, hi = start_s - pre, start_s - pre + span
    centers = [(w.start + w.end) / 2 for w in words]
    if any(centers[i] > centers[i + 1] for i in range(len(centers) - 1)):
        # ASR timestamps are not monotone (seen in real clips). The offline path keeps the given order
        # and so must we: reordering would change the neighbour features. Use the whole list.
        return list(words)
    i0 = max(bisect.bisect_left(centers, lo) - 1, 0)
    i1 = min(bisect.bisect_left(centers, hi) + 1, len(words))
    return list(words[i0:i1])
