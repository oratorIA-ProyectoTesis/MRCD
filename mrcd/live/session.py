"""Deterministic core of the live path: tracker + grid scheduling + provisional / final rows.

``scripts/live_demo.LiveEngine`` owns the threads, the microphone and the ASR; this module owns
everything that decides WHAT is scored and WHEN, so it can be tested without a microphone, a model
download or wall-clock time.

Named configurations (the previous behaviour stays reproducible, nothing is silently replaced)
---------------------------------------------------------------------------------------------
``live_cfg_v1``  behaviour of commit 80a921a: margins L=3 s / R=3 s, rows without ``status``, no
                 end-of-session pass (the last ~6-8 s of every session are never scored).
``live_cfg_v2``  v1 + right margin 4 s (a window is ready ``max(5, 3 + 4) = 7`` s after its start), explicit
                 end-of-session pass (``finalize``), ``status`` on every row, ``emitted_at_s`` (stream time at
                 emission); see results/ejecucion_material4_20261030/paridad/PARIDAD_LIVE_V2.md.

Provisional vs final
--------------------
* A row emitted by :meth:`LiveScorer.step` is ``status="provisional"``: its F0 centre is the causal
  one at that tick (median of the voiced frames committed so far; centre 0 until 10 voiced frames)
  and its ASR words are the ones known at that tick. Provisional rows are NEVER edited or removed.
* Future context a window needs: the window ends at ``start + 3 s``; pYIN commits frames only
  ``right_margin_s`` after they were recorded; the linguistic context needs 5 s after the window start.
  Latency of a window = ``max(5, 3 + right_margin_s)`` s after its start, i.e. ``right_margin_s`` s
  after its end, plus the ASR time of that tick, plus the 0.5 s tick.
* :meth:`LiveScorer.finalize` (called when the session stops) closes the tracker (every remaining
  frame is committed with no future context), scores the tail that provisional scoring never reaches
  (``status="final"``; optionally the terminal window ``end-3 s`` when it does not fall on the grid)
  and, if requested (``rescore_at_close``), appends one ``status="final"`` row per earlier window with
  ``revises=<timestamp>`` computed with the whole-session F0 centre and the final words. Original
  rows stay as they were: a consumer that wants the best estimate takes the last row per timestamp.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable

import numpy as np

from mrcd.constants import ACOUSTIC_HOP_S, SAMPLE_RATE, STRIDE_S, WINDOW_S
from mrcd.live.session_features import SessionFeatureTracker
from mrcd.live.window_builder import window_text


@dataclass(frozen=True)
class LiveConfig:
    name: str
    left_margin_s: float = 3.0
    right_margin_s: float = 3.0
    #: end-of-session pass on stop (score the tail, optional terminal window)
    finalize_on_stop: bool = False
    #: add ``status`` / ``emitted_at_s`` / ``f0_center_st`` / ``terminal`` to every row
    mark_status: bool = False
    #: score the terminal window ``end-3 s`` at the end of the session (only with finalize_on_stop)
    terminal_window: bool = False
    #: append ``status="final"`` rows that revise every earlier window (only with finalize_on_stop)
    rescore_at_close: bool = False
    #: when the consumer lags, windows older than ``latest - lag_lookback_s`` are skipped (and recorded)
    lag_lookback_s: float = 3.0
    #: replace microphone blocks flagged by the driver with zeros so the timeline keeps its length
    fill_dropped_audio: bool = False
    #: resume the pYIN Viterbi pass from the exact forward scores of the last committed frame (removes the left
    #: start-dependence that no margin removes; see mrcd/live/pyin_carry.py)
    viterbi_carry: bool = False

    def replace(self, **kw) -> "LiveConfig":
        return replace(self, **kw)


LIVE_CFG_V1 = LiveConfig(name="live_cfg_v1")
LIVE_CFG_V2 = LiveConfig(
    name="live_cfg_v2",
    # R = 4 s: with R = 3 s one frame of a 10.5 min real+synthetic session (seed 3, 0.25-1 s ticks) differed from
    # the whole-file extraction (3.16 st, margin_study_v2_long10_5_seed3.json); L = 3, R = 4 gave 0 differing
    # frames on 101 local signals x 3 partitions, the 8 min session and four 10-10.5 min sessions (empirical).
    left_margin_s=3.0,
    right_margin_s=4.0,
    finalize_on_stop=True,
    mark_status=True,
    terminal_window=True,
    rescore_at_close=True,
    fill_dropped_audio=True,
)
#: v2 + exact Viterbi resume. Same margins and scheduling as v2 on purpose: the only change is the carried forward
#: scores, so v2 -> v3 differences are attributable to that single variable.
LIVE_CFG_V3 = LIVE_CFG_V2.replace(name="live_cfg_v3", viterbi_carry=True)
CONFIGS = {c.name: c for c in (LIVE_CFG_V1, LIVE_CFG_V2, LIVE_CFG_V3)}


def get_config(cfg: "str | LiveConfig") -> LiveConfig:
    if isinstance(cfg, LiveConfig):
        return cfg
    try:
        return CONFIGS[cfg]
    except KeyError:
        raise ValueError(
            f"unknown live config {cfg!r}; known: {sorted(CONFIGS)}"
        ) from None


class LiveScorer:
    """Feeds audio to a :class:`SessionFeatureTracker` and scores the 0.5 s grid windows.

    ``score(tensors) -> probabilities`` receives the dict of one window (batch of 1) and returns a
    1-D array aligned with ``labels``.
    """

    def __init__(
        self,
        score: Callable[[dict], np.ndarray],
        labels: list[str],
        *,
        config: "str | LiveConfig" = LIVE_CFG_V2,
        f0_center: "str | float" = "causal",
        context: bool = True,
        include_pros: bool = True,
        ling_dim: int | None = None,
        vad_model=None,
        asr_mark: dict | None = None,
    ) -> None:
        #: extra keys added to EVERY row (e.g. ``StreamingASR.asr_mark()`` for an experimental ASR profile);
        #: ``None`` (the p0 default) leaves the row schema untouched.
        self.asr_mark = dict(asr_mark) if asr_mark else None
        self.config = get_config(config)
        self.score, self.labels = score, list(labels)
        self.context, self.include_pros, self.ling_dim = context, include_pros, ling_dim
        self.tracker = SessionFeatureTracker(
            f0_center=f0_center,
            left_margin_s=self.config.left_margin_s,
            right_margin_s=self.config.right_margin_s,
            vad_model=vad_model,
            viterbi_carry=self.config.viterbi_carry,
        )
        self.rows: list[dict] = []
        self.gaps: list[dict] = []
        self._next: float | None = None  # next grid start not yet scored
        self._emitted: list[float] = []  # timestamps of provisional rows (for rescore)
        self.finalized = False

    # ----------------------------------------------------------------- input
    @property
    def duration_s(self) -> float:
        return self.tracker.duration_s

    def push(self, audio: np.ndarray, lost: int = 0) -> None:
        """Feed new audio; ``lost`` samples (evicted from the ring) are zero-filled and recorded."""
        if self.finalized:
            raise RuntimeError("LiveScorer is finalized")
        if lost:
            self.gaps.append(
                {
                    "kind": "audio_lost",
                    "at_s": self.tracker.duration_s,
                    "lost_s": lost / SAMPLE_RATE,
                }
            )
            self.tracker.append(np.zeros(lost, np.float32), commit=False)
        self.tracker.append(audio)

    # ---------------------------------------------------------------- scoring
    def _row(
        self,
        ts: float,
        words,
        *,
        status: str,
        terminal: bool = False,
        revises: float | None = None,
    ) -> dict:
        tensors = self.tracker.window(
            ts,
            words,
            context=self.context,
            include_pros=self.include_pros,
            ling_dim=self.ling_dim,
        )
        p = np.asarray(self.score(tensors), dtype=np.float64).reshape(-1)
        idx = int(p.argmax())
        row = {
            "timestamp": ts,
            "texto_ventana": window_text(words, ts),
            "clase": self.labels[idx],
            "confianza": float(p[idx]),
            "todas_las_probabilidades": {
                label: float(v) for label, v in zip(self.labels, p)
            },
        }
        if self.config.mark_status:
            row["status"] = status
            row["emitted_at_s"] = self.tracker.duration_s
            row["f0_center_st"] = self.tracker.f0_center_st()
            row["terminal"] = terminal
            if revises is not None:
                row["revises"] = revises
        if self.asr_mark:
            row.update(self.asr_mark)
        return row

    def step(self, words, *, asr_floor_s: float = 0.0) -> list[dict] | None:
        """Score every grid window that became ready. ``None`` while warming up.

        ``asr_floor_s``: oldest session time for which ``words`` still has the 5 s of left context
        (the ASR re-transcribes only the recent ring); older windows are skipped and recorded.
        """
        latest = self.tracker.latest_start()
        if latest is None:
            return None
        oldest = max(latest - self.config.lag_lookback_s, asr_floor_s, 0.0)
        first = latest if self._next is None else max(self._next, oldest)
        expected = 0.0 if self._next is None else self._next
        skipped = int(round((first - expected) / STRIDE_S))
        if skipped > 0 and first <= latest + 1e-9:
            self.gaps.append(
                {
                    "kind": "windows_skipped",
                    "from_s": expected,
                    "to_s": first,
                    "n_windows": skipped,
                    "reason": "consumer_lag_or_asr_floor",
                    "at_s": self.tracker.duration_s,
                }
            )
        out = []
        for k in range(int(round((latest - first) / STRIDE_S)) + 1):
            ts = first + k * STRIDE_S
            row = self._row(ts, words, status="provisional")
            self.rows.append(row)
            self._emitted.append(ts)
            out.append(row)
        self._next = max(latest + STRIDE_S, self._next or 0.0)
        return out

    def finalize(self, words, *, asr_floor_s: float = 0.0) -> list[dict]:
        """Close the session and score what provisional scoring never reaches. Idempotent.

        Only meaningful with ``config.finalize_on_stop``; with ``live_cfg_v1`` it just marks the
        scorer closed and returns ``[]`` (the old behaviour: the tail is dropped).
        """
        if self.finalized:
            return []
        self.finalized = True
        if not self.config.finalize_on_stop:
            return []
        tr = self.tracker
        tr.finalize()
        end = tr.duration_s
        out: list[dict] = []
        if end < WINDOW_S:
            return out
        last_grid = float(np.floor((end - WINDOW_S) / STRIDE_S + 1e-9) * STRIDE_S)
        # With rescoring the final pass also fills every grid window that never got a provisional row (skipped at
        # the start or under consumer lag); the words of the whole session are available at this point.
        first = (
            0.0
            if self.config.rescore_at_close
            else max(self._next if self._next is not None else 0.0, asr_floor_s, 0.0)
        )
        first = float(np.ceil(first / STRIDE_S - 1e-9) * STRIDE_S)  # keep the grid
        done = {int(round(t / STRIDE_S)) for t in self._emitted}
        n = (
            int(round((last_grid - first) / STRIDE_S)) + 1
            if last_grid >= first - 1e-9
            else 0
        )
        for k in range(n):
            ts = first + k * STRIDE_S
            if int(round(ts / STRIDE_S)) in done:
                continue
            row = self._row(ts, words, status="final")
            self.rows.append(row)
            out.append(row)
        if (
            self.config.terminal_window
            and (end - WINDOW_S) - last_grid > ACOUSTIC_HOP_S
        ):
            row = self._row(end - WINDOW_S, words, status="final", terminal=True)
            self.rows.append(row)
            out.append(row)
        if self.config.rescore_at_close:
            for ts in list(self._emitted):
                row = self._row(ts, words, status="final", revises=ts)
                self.rows.append(row)
                out.append(row)
        return out
