"""Verbatimness metrics shared by Plan C and Plan B (EXPLORATORY, DIHANA folds 1-4).

* WER: ``mrcd.asr_recovery.stats.wer`` against text A (human transcript, marks stripped), pooled (ratio of sums).
* survival_rep: ``turn_survives("repetition", rep_marks, text, {"A_literal_repeat"})`` on ``rep_literal == 1`` turns
  (definition of results/asr_recovery_20261031/EXPERIMENTO_ASR.md).
* filler_token_recall: sum(min(hyp, ref)) / sum(ref) over turns with >= 1 non-lexical ``<f>`` tag; ref = number of
  those tags, hyp = filled-pause tags of a CrisperWhisper output + non-lexical filler words of the scoring text.
* filler_turn_survival: share of those turns whose output has >= 1 filled pause (survival-models protocol 4.1).

Uncertainty: cluster (speaker) percentile bootstrap; the same resample indices are used for every arm, so paired
differences are paired.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from mrcd.asr_recovery.stats import wer
from mrcd.context_eval.reference import _norm_tokens
from mrcd.context_eval.survival import turn_survives
from mrcd.context_eval.transcript_reference import is_nonlexical_filler

LITERAL = {"A_literal_repeat"}
TAG_RE = re.compile(r"\[([^\[\]]*)\]")
FILLED_PAUSE_TAGS = frozenset(
    {"uh", "um", "eh", "em", "er", "ah", "ahh", "hm", "hmm", "mm", "mmm"}
)


def strip_tags_cw(raw: str) -> str:
    return " ".join(TAG_RE.sub(" ", raw or "").split())


def filled_tags(raw: str | None) -> int:
    return sum(
        m.group(1).strip().lower() in FILLED_PAUSE_TAGS
        for m in TAG_RE.finditer(raw or "")
    )


def tokens(text: str) -> list[str]:
    return _norm_tokens(text)


def nonlex_filler_words(text: str) -> int:
    t = tokens(text)
    return sum(
        is_nonlexical_filler(w, t[i + 1] if i + 1 < len(t) else None)
        for i, w in enumerate(t)
    )


@dataclass(frozen=True)
class UnitRef:
    unit_id: str
    speaker: str
    text_a: str
    rep_literal: int | None
    rep_marks: tuple
    n_fill: int

    @classmethod
    def from_pool(cls, u: dict, n_fill: int) -> "UnitRef":
        return cls(
            u["unit_id"],
            u["speaker"],
            u["text_A"],
            u["ref"]["rep_literal"],
            tuple(u["ref"]["rep_marks"]),
            n_fill,
        )


def unit_scores(
    refs: list[UnitRef],
    texts: list[str],
    raw: list[str] | None = None,
    verbatim: list[str] | None = None,
) -> dict[str, np.ndarray]:
    """Per-unit arrays; ``texts`` are scoring texts (tags removed), ``raw`` the raw outputs (for filled-pause tags).

    With ``verbatim`` (Plan B targets with reconstructed fillers) two more metrics are computed: WER vs the verbatim
    target and the false-filler rate (filled pauses per 100 reference tokens on turns without a ``<f>`` tag)."""
    raw = raw or [None] * len(texts)
    e, n, surv, fh, fr, ft = [], [], [], [], [], []
    for r, t, rw in zip(refs, texts, raw):
        a, b = wer(r.text_a, t)
        e.append(a)
        n.append(b)
        surv.append(
            float(turn_survives("repetition", list(r.rep_marks), t, LITERAL))
            if r.rep_literal == 1
            else np.nan
        )
        h = filled_tags(rw) + nonlex_filler_words(t)
        fh.append(min(h, r.n_fill))
        fr.append(r.n_fill)
        ft.append(float(h > 0) if r.n_fill > 0 else np.nan)
    out = dict(wer_e=e, wer_n=n, surv=surv, fill_hit=fh, fill_ref=fr, fill_turn=ft)
    if verbatim is not None:
        ve, vn, ffh, ffn = [], [], [], []
        for r, t, rw, v in zip(refs, texts, raw, verbatim):
            a, b = wer(v, t)
            ve.append(a)
            vn.append(b)
            free = r.n_fill == 0
            ffh.append(filled_tags(rw) + nonlex_filler_words(t) if free else 0)
            ffn.append(len(tokens(r.text_a)) if free else 0)
        out.update(wer_v_e=ve, wer_v_n=vn, ff_hit=ffh, ff_ref=ffn)
    return {k: np.array(v, float) for k, v in out.items()}


def boot_indices(n_or_groups, B: int, seed: int) -> list[np.ndarray]:
    """Cluster bootstrap: an int means one cluster per unit; a sequence gives the cluster of each unit."""
    groups = (
        list(range(n_or_groups)) if isinstance(n_or_groups, int) else list(n_or_groups)
    )
    uniq = sorted(set(groups), key=str)
    members = {g: np.array([i for i, x in enumerate(groups) if x == g]) for g in uniq}
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(B):
        pick = rng.integers(0, len(uniq), len(uniq))
        out.append(np.concatenate([members[uniq[k]] for k in pick]))
    return out


def _stats(s: dict[str, np.ndarray], idx: np.ndarray | None = None) -> dict[str, float]:
    if idx is not None:
        s = {k: v[idx] for k, v in s.items()}
    m = ~np.isnan(s["surv"])
    f = s["fill_ref"] > 0
    with np.errstate(invalid="ignore", divide="ignore"):
        extra = {}
        if "wer_v_e" in s:
            extra = {
                "wer_verbatim": s["wer_v_e"].sum() / s["wer_v_n"].sum(),
                "false_filler_per100": 100 * s["ff_hit"].sum() / s["ff_ref"].sum(),
            }
        return extra | {
            "wer": s["wer_e"].sum() / s["wer_n"].sum(),
            "survival_rep": s["surv"][m].mean() if m.any() else np.nan,
            "filler_token_recall": s["fill_hit"][f].sum() / s["fill_ref"][f].sum()
            if f.any()
            else np.nan,
            "filler_turn_survival": s["fill_turn"][f].mean() if f.any() else np.nan,
        }


def _ci(vals) -> list[float]:
    v = np.array([x for x in vals if not np.isnan(x)])
    return (
        [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
        if len(v)
        else [np.nan, np.nan]
    )


def summarise(s: dict[str, np.ndarray], idx: list[np.ndarray]) -> dict:
    point = _stats(s)
    boots = [_stats(s, i) for i in idx]
    m = ~np.isnan(s["surv"])
    f = s["fill_ref"] > 0
    out = {
        k: {"value": float(point[k]), "ci95": _ci(b[k] for b in boots)} for k in point
    }
    out["wer"]["n_ref_tokens"] = int(s["wer_n"].sum())
    out["survival_rep"].update(n=int(m.sum()), k=int(np.nansum(s["surv"])))
    out["filler_token_recall"].update(
        n_ref=int(s["fill_ref"][f].sum()),
        hits=int(s["fill_hit"][f].sum()),
        n_turns=int(f.sum()),
    )
    out["filler_turn_survival"]["n"] = int(f.sum())
    return out


def paired(
    a: dict[str, np.ndarray], b: dict[str, np.ndarray], idx: list[np.ndarray]
) -> dict:
    """a - b on identical units, same resamples."""
    pa, pb = _stats(a), _stats(b)
    out = {}
    for k in pa:
        d = [x[k] - y[k] for x, y in ((_stats(a, i), _stats(b, i)) for i in idx)]
        out[k] = {"delta": float(pa[k] - pb[k]), "ci95": _ci(d)}
    return out
