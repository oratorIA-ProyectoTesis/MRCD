"""Decision profiles of the service (contract rev 1.4).

* ``v0``       - the engine's historical decision for every class (window argmax events). Reproduces every document the
                 service produced before rev 1.4 (events, scores, summary). Always available, explicit opt-in.
* ``o2026_12`` - EXPLORATORY, opt-in (plan A, 2026-11-08). ONE ASR pass with ``p1_disfluent_prompt`` feeds every class:
                 ``filler_word`` (subclass ``filled_pause``) and ``prolongation`` come from unit pipelines re-fitted on P1 text
                 and P1 engine features (same blocks / fusion as o2026_11, LR only); ``repetition`` is the confirmatory
                 lexical rule as candidate generator followed by a demotion-only contextual layer; lexical fillers are a
                 second ``filler_word`` subclass (``lexical_filler``). See ``mrcd/service/contextual.py`` and
                 ``results/plan_a_20261108/PROTOCOL_EXPLORATORY_A.md``.
* ``o2026_11`` - PRODUCT DEFAULT of the service. ``filler_word`` and ``prolongation`` come from the per-class pipelines
                 of the O-campaign (2026-11-01) applied to pause-delimited ASR segments; ``repetition`` and every other class
                 keep the engine decision (the campaign did not beat the one-line rule / S3 route for repetition).

The library (``DisfluencyEngine.analyze``) is NOT affected: it always returns the V0 events. Only the service chooses a profile.

Honest limits (also emitted as a warning in every document): exploratory result (DIHANA folds 1-4, nested CV by speaker),
configuration selection unstable across folds, the V0 comparator is a weak baseline, scores are model outputs and NOT
calibrated probabilities (``calibration_claim: none``), and the pipelines were trained on DIHANA *turns* while the service
applies them to ASR segments split at pauses >= 0.6 s (the same unit convention as the S3 text route), a domain shift that
was not measured outside DIHANA (EPIC filler: no demonstrated gain; AMERESCO prolongation: uninformative).

Artifacts are plain JSON (no pickle). Every file is read once, its sha256 is compared with the value PINNED IN THIS SOURCE
FILE and only then parsed from those same bytes; a changed file is refused (``ProfileArtifactError``).
"""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from mrcd.optimize import safe_pipeline as SAFE
from mrcd.service import s3_features as S3F
from mrcd.service.contract import (  # noqa: F401  (re-exported)
    SCORE_KIND_RULE,
    SCORE_KIND_TEXT_ROUTE,
    SCORE_KIND_UNIT_PIPELINE,
)

PROFILE_V0 = "v0"
PROFILE_O = "o2026_11"
#: o2026_11 + repetition from a second ASR pass with the P1 prompt (contract rev 1.5, see ``repetition_events``)
PROFILE_OB = "o2026_11b"
#: single P1 pass for every class + candidate/demotion repetition + filler subclasses (plan A, opt-in)
PROFILE_12 = "o2026_12"
DECISION_PROFILES = (PROFILE_V0, PROFILE_O, PROFILE_OB, PROFILE_12)
DEFAULT_DECISION_PROFILE = (
    PROFILE_O  # service / app default; the library default stays V0
)
O_CLASSES = ("filler_word", "prolongation")
#: profiles whose filler_word / prolongation come from the O-campaign unit pipelines
UNIT_PIPELINE_PROFILES = (PROFILE_O, PROFILE_OB, PROFILE_12)
#: ASR profile of the second pass used by ``o2026_11b`` for repetition only (and the single pass of ``o2026_12``)
P1_ASR_PROFILE = "p1_disfluent_prompt"
#: repetition routes on P1 segment text: the one-line lexical rule, or the S3 text route (ctx_LR)
REPETITION_ROUTES = ("rule_p1", "rep_ctx_p1")
#: route used by ``o2026_11b`` (fixed by the frozen protocol of results/mejora_servicio_20261102)
OB_REPETITION_ROUTE = "rule_p1"


def replaced_classes(profile: str) -> tuple[str, ...]:
    """Classes whose engine (V0) events the profile drops and replaces with its own."""
    if profile == PROFILE_O:
        return O_CLASSES
    if profile in (PROFILE_OB, PROFILE_12):
        return (*O_CLASSES, "repetition")
    return ()


def required_asr_profile(profile: str) -> str | None:
    """The ASR profile a decision profile owns (``o2026_12`` -> P1), or None when any requested profile is used."""
    return P1_ASR_PROFILE if profile == PROFILE_12 else None


def trained_asr_profile(profile: str) -> str:
    """ASR profile whose text the profile's unit pipelines were fitted on."""
    return P1_ASR_PROFILE if profile == PROFILE_12 else "p0"


def uses_p1_repetition(profile: str) -> bool:
    return profile == PROFILE_OB


PROFILE_DIR = Path(__file__).resolve().parent / "profiles" / PROFILE_O
PROFILE_12_DIR = Path(__file__).resolve().parent / "profiles" / PROFILE_12
#: sha256 of the exported pipelines, pinned here (a second manifest next to the files would not protect against an
#: attacker who can edit both). Regenerate with ``scripts/o_integration/export_json_pipelines.py`` and update on purpose.
PINNED_SHA256 = {
    "filler_word.json": "2a07b50c156bd8e064f6703b38c78dd85b1b853a6ad4a5e0e3be354a80858c1e",
    "prolongation.json": "d6230ed2e8b930e1cd930c89b1d74769310f7068c65c3070f75faee83ab7a96a",
}
#: sha256 of the campaign pickles the JSON files were exported from (provenance only; the pickles are never read here)
SOURCE_PICKLE_SHA256 = {
    "filler_word": "787828317c87b188469f09bec03d6fd3c633e8b1af2a7d2dff84725936064c91",
    "prolongation": "ba2f253648717a0671cf7f15d833bd466327b8f07db34ad43f471975b38f3f21",
}
#: sha256 of the o2026_12 pipelines (exported by ``scripts/plan_a/evaluate.py final``), pinned here on purpose
PINNED_SHA256_12 = {
    "filler_word.json": "b974eb906673cdd8a545801468248709f36dd525a41327c4d946c2c2eb05875d",
    "prolongation.json": "59c8295cb539b02f33b2e713fff48bf44ec5020fba12fedc80051a754143a802",
}
_PINS = {PROFILE_O: PINNED_SHA256, PROFILE_12: PINNED_SHA256_12}
MARGIN_S = (
    0.15  # context kept on each side of a segment clip (ASR word times are approximate)
)
MODALITIES = ["acoustic", "linguistic"]
VALIDATION_TEXT = (
    "exploratory unit-level decision (O-campaign pipeline) on an ASR segment split at pauses; "
    "interval = segment span; trained on DIHANA turns, not validated outside DIHANA"
)


class ProfileArtifactError(ValueError):
    """A profile artifact is missing, altered or malformed."""


def sha256_hex(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def load_o_pipelines(directory: Path | None = None, profile: str = PROFILE_O) -> dict:
    """Verified pipelines of ``o2026_11`` (or ``o2026_12``): ``{cls: ClassPipeline}``. Not cached (see ``o_pipelines``)."""
    pins = _PINS[profile]
    d = (  # module attributes read at call time (tests relocate them)
        Path(directory)
        if directory is not None
        else (PROFILE_12_DIR if profile == PROFILE_12 else PROFILE_DIR)
    )
    out = {}
    for cls in O_CLASSES:
        name = f"{cls}.json"
        try:
            blob = (d / name).read_bytes()
        except OSError as exc:
            raise ProfileArtifactError(
                f"{name}: artifact missing ({type(exc).__name__})"
            ) from exc
        if sha256_hex(blob) != pins[name]:
            raise ProfileArtifactError(f"{name}: sha256 mismatch (artifact altered)")
        try:
            out[cls] = SAFE.load_pipeline(json.loads(blob.decode("utf8")))
        except (ValueError, KeyError, TypeError) as exc:
            raise ProfileArtifactError(
                f"{name}: malformed artifact ({type(exc).__name__})"
            ) from exc
    return out


@lru_cache(maxsize=1)
def o_pipelines() -> dict:
    return load_o_pipelines()


@lru_cache(maxsize=1)
def o12_pipelines() -> dict:
    return load_o_pipelines(profile=PROFILE_12)


def pipelines_for(profile: str) -> dict:
    """Verified unit pipelines used by a decision profile (o2026_11 / o2026_11b share the o2026_11 files)."""
    return o12_pipelines() if profile == PROFILE_12 else o_pipelines()


def profile_meta(
    profile: str,
    *,
    applied: bool | None = True,
    asr_profile: str = "p0",
    stats: dict | None = None,
) -> dict:
    """The ``provenance.decision_profile`` object of a document. ``asr_profile`` (the requested ASR profile) is only
    written into the per-class ``asr_profile_used`` of ``o2026_11b`` (rev 1.5); other profiles are unchanged."""
    if profile == PROFILE_V0:
        return {
            "name": PROFILE_V0,
            "status": "historical",
            "classes": {
                c: {"route": "engine_v0_window_events"}
                for c in ("filler_word", "prolongation", "repetition")
            },
            "calibration_claim": "none",
            "pipelines_verified": None,
            "note": "engine decision of every class (window argmax events); identical to contract revisions <= 1.3",
        }
    if profile == PROFILE_12:
        return _profile_12_meta(applied, stats)
    classes = {
        c: {
            "route": "o_campaign_unit_pipeline",
            "unit": f"ASR segment split at pauses >= {S3F.GAP_S} s (max {S3F.MAX_WORDS} words)",
            "artifact": f"{c}.json",
            "artifact_sha256": PINNED_SHA256[f"{c}.json"],
            "source_pickle_sha256": SOURCE_PICKLE_SHA256[c],
            "pipeline_version": "o-campaign/1",
            "asr_profile_trained_on": "p0",
        }
        for c in O_CLASSES
    }
    if profile == PROFILE_OB:
        for c in O_CLASSES:
            classes[c]["asr_profile_used"] = asr_profile
        classes["repetition"] = {
            "route": f"{OB_REPETITION_ROUTE}_on_p1_segments",
            "unit": f"P1 ASR segment split at pauses >= {S3F.GAP_S} s (max {S3F.MAX_WORDS} words)",
            "asr_profile_used": P1_ASR_PROFILE,
            "rule": "immediate exact repetition of 1-4 tokens (textfeat.rule_rep)"
            if OB_REPETITION_ROUTE == "rule_p1"
            else "S3 ctx_LR text route rep_ctx_p1",
        }
        return {
            "name": PROFILE_OB,
            "status": "exploratory",
            "classes": classes,
            "calibration_claim": "none",
            "pipelines_verified": None if applied is None else bool(applied),
            "note": "o2026_11 for filler_word and prolongation (P0 text); repetition from a second ASR pass with the "
            "P1 prompt. The library default stays v0.",
        }
    classes["repetition"] = {"route": "engine_v0_window_events"}
    return {
        "name": PROFILE_O,
        "status": "exploratory",
        "classes": classes,
        "calibration_claim": "none",
        "pipelines_verified": None if applied is None else bool(applied),
        "note": "product default of the service; the library default stays v0. Repetition keeps the engine decision.",
    }


def _profile_12_meta(applied: bool | None, stats: dict | None) -> dict:
    from mrcd.service import contextual as CX

    stats = stats or {}
    classes = {
        c: {
            "route": "unit_pipeline_refit_p1",
            "unit": f"ASR segment split at pauses >= {S3F.GAP_S} s (max {S3F.MAX_WORDS} words)",
            "artifact": f"{c}.json",
            "artifact_sha256": PINNED_SHA256_12[f"{c}.json"],
            "pipeline_version": "o-campaign/1",
            "asr_profile_trained_on": P1_ASR_PROFILE,
            "asr_profile_used": P1_ASR_PROFILE,
        }
        for c in O_CLASSES
    }
    classes["filler_word"]["subclasses"] = {
        CX.FILLED_PAUSE: "unit pipeline (trained on DIHANA non-lexical f marks)",
        CX.LEXICAL_FILLER: "lexical items of the existing Spanish filler lexicon with the auto-labeler evidence rule "
        "(adjacent gap >= 0.15 s, duration >= 0.35 s or comma/ellipsis); otherwise uncertain, not emitted",
    }
    classes["filler_word"]["lexical_candidates"] = stats.get("lexical_filler")
    classes["repetition"] = {
        "route": "rule_candidates_contextual_demotion",
        "asr_profile_used": P1_ASR_PROFILE,
        "generator": "confirmatory lexical rule: immediate exact repetition of 1-4 norm_tokens, non-lexical "
        "fillers removed, interval = second copy",
        "demotion_rules": {
            "loop": "inside an ASR hallucination loop (asr_profiles._scan_loops) -> uncertain",
            "gap": "silence between copies > 1.5 s (H.repetition_max_gap_s) -> uncertain",
            "seam": "copies decoded by different 8 s / 2 s ASR chunks -> uncertain",
            "polarity": "one-token repeat of a polarity item (textfeat.POLARITY) -> functional",
        },
        "emitted_states": [CX.DISFLUENT],
        "candidates": stats.get("repetition"),
    }
    return {
        "name": PROFILE_12,
        "status": "exploratory",
        "classes": classes,
        "calibration_claim": "none",
        "pipelines_verified": None if applied is None else bool(applied),
        "note": "single ASR pass with p1_disfluent_prompt for every class; exploratory (DIHANA folds 1-4, plan A "
        "protocol results/plan_a_20261108). Only the disfluent state is emitted as an event; demoted candidates are "
        "counted here. The functional state has no human reference. The library default stays v0.",
    }


def warning_for(profile: str) -> dict | None:
    if profile == PROFILE_12:
        return {
            "code": "decision_profile_exploratory",
            "message": "Decision profile o2026_12 (opt-in, exploratory, DIHANA folds 1-4 only): one ASR pass with the P1 "
            "prompt (p1_disfluent_prompt) feeds every class. filler_word (subclasses filled_pause and lexical_filler) "
            "and prolongation pipelines were re-fitted on P1 text; repetition is a lexical rule whose candidates may "
            "only be demoted (to uncertain or functional) by fixed contextual rules. Scores are not calibrated "
            "probabilities; the functional state has no human reference. Use decision_profile=o2026_11 or v0 to "
            "reproduce earlier outputs.",
        }
    if profile == PROFILE_OB:
        return {
            "code": "decision_profile_exploratory",
            "message": "Decision profile o2026_11b: filler_word and prolongation as in o2026_11 (exploratory pipelines on "
            "DIHANA folds 1-4, scores not calibrated, units are ASR segments split at pauses). Repetition comes from a "
            "second ASR pass with the P1 prompt (p1_disfluent_prompt) and "
            + (
                "the immediate-repetition lexical rule"
                if OB_REPETITION_ROUTE == "rule_p1"
                else "the S3 text route rep_ctx_p1"
            )
            + " on pause-split segments; evaluated on DIHANA folds 1-4 only. Use decision_profile=o2026_11 or v0 to "
            "reproduce earlier outputs.",
        }
    if profile != PROFILE_O:
        return None
    return {
        "code": "decision_profile_exploratory",
        "message": "Decision profile o2026_11 (service default): filler_word and prolongation come from exploratory "
        "pipelines selected by nested cross-validation on DIHANA folds 1-4 (configuration unstable across folds, weak V0 "
        "baseline, no external gain demonstrated). Scores are model outputs, not calibrated probabilities. Units are ASR "
        "segments split at pauses, whereas training used DIHANA turns. Repetition keeps the engine decision. "
        "Use decision_profile=v0 to reproduce the historical output.",
    }


# ----------------------------------------------------------------------------------------------- unit records
def _clip_range(seg: list[dict], n: int, sr: int) -> tuple[int, int]:
    a = max(0, int(round((seg[0]["start"] - MARGIN_S) * sr)))
    b = min(n, int(round((seg[-1]["end"] + MARGIN_S) * sr)))
    return a, max(b, a + 1)


def unit_records(
    engine, audio: np.ndarray, sample_rate: int, words: list[dict], key: str = "p0"
) -> tuple[list[dict], list[dict]]:
    """One record (``optimize.data`` layout) per pause-delimited ASR segment, plus its ``{start, end}`` span.

    Each segment becomes a standalone clip (like a DIHANA turn): words are re-timed to the clip, the engine builds the
    window probabilities / pooled vector / pause descriptors for that clip (``DisfluencyEngine.unit_features``)."""
    recs, spans = [], []
    for seg in S3F.segments_from_words(words):
        a, b = _clip_range(seg, len(audio), sample_rate)
        t0 = a / sample_rate
        rel = []
        for w in seg:
            w2 = dict(w)
            w2["start"] = max(0.0, float(w["start"]) - t0)
            w2["end"] = max(w2["start"], float(w["end"]) - t0)
            rel.append(w2)
        f = engine.unit_features(audio[a:b], rel)
        recs.append(
            {
                "probs": f["probs"],
                "pooled": f["pooled"],
                "pause": f["pause"],
                "n_samples": f["n_samples"],
                "words": {key: rel},
                "text": {key: " ".join(w["text"] for w in seg)},
                "acoustic": None,
            }
        )
        spans.append({"start": float(seg[0]["start"]), "end": float(seg[-1]["end"])})
    return recs, spans


def profile_events(
    pipelines: dict,
    recs: list[dict],
    spans: list[dict],
    key: str = "p0",
    subclass: dict | None = None,
) -> list[dict]:
    """Positive units of each pipelined class as raw event dicts (analyzer layout, plus ``_decision``).
    ``subclass`` (o2026_12): ``{cls: subclass}`` adds a ``context`` (state disfluent + subclass) to those events."""
    out = []
    for cls in O_CLASSES:
        if not recs:
            break
        pipe = pipelines[cls]
        pred = pipe.predict(recs)
        for i, (r, sp) in enumerate(zip(recs, spans)):
            if not int(pred["decision"][i]):
                continue
            score = float(np.clip(pred["score"][i], 0.0, 1.0))
            out.append(
                {
                    "class": cls,
                    "start_s": round(sp["start"], 3),
                    "end_s": round(sp["end"], 3),
                    "score": score,
                    "n_windows": int(len(r["probs"])),
                    "text": r["text"][key],
                    **(
                        {
                            "context": {
                                "state": "disfluent",
                                "subclass": subclass[cls],
                                "basis": "unit_pipeline",
                            }
                        }
                        if subclass and cls in subclass
                        else {}
                    ),
                    "decision": {
                        "state": str(pred["state"][i]),
                        "estimate": round(
                            float(np.clip(pred["calibrated"][i], 0.0, 1.0)), 6
                        ),
                        "lo": round(float(pipe.band["lo"]), 6),
                        "hi": round(float(pipe.band["hi"]), 6),
                        "basis": "o_campaign_unit_pipeline_inner_band_uncalibrated",
                        "calibration_claim": "none",
                    },
                }
            )
    return out


def apply_profile(
    engine, audio: np.ndarray, sample_rate: int, words: list[dict] | None
) -> list[dict]:
    """Events of the O pipelines for one recording. Raises ``ProfileArtifactError`` if an artifact was altered."""
    pipes = o_pipelines()
    recs, spans = unit_records(engine, audio, sample_rate, words or [])
    return profile_events(pipes, recs, spans)


def apply_profile_12(
    engine, audio: np.ndarray, sample_rate: int, words: list[dict] | None
) -> tuple[list[dict], dict]:
    """Events and candidate statistics of ``o2026_12`` for one recording (the words ARE the single P1 pass)."""
    from mrcd.service import contextual as CX

    words = list(words or [])
    pipes = o12_pipelines()
    recs, spans = unit_records(engine, audio, sample_rate, words, key="p1")
    events = profile_events(
        pipes, recs, spans, key="p1", subclass={"filler_word": CX.FILLED_PAUSE}
    )
    duration = len(audio) / float(sample_rate)
    cands = CX.classify_repetitions(words, duration)
    for ev in CX.merge_disfluent(cands):
        events.append(
            {
                "class": "repetition",
                "start_s": round(float(ev["start_s"]), 3),
                "end_s": round(float(ev["end_s"]), 3),
                "score": 1.0,
                "score_kind": SCORE_KIND_RULE,
                "n_windows": 1,
                "modalities_used": ["linguistic"],
                "validation": "lexical rule candidate (immediate exact repetition of 1-4 tokens) kept as disfluent by "
                "the contextual demotion rules; P1 ASR text; evaluated on DIHANA folds 1-4 only",
                "text": None,
                "basis": "rule_candidate_contextual",
                "decision": None,
                "context": {
                    "state": CX.DISFLUENT,
                    "subclass": None,
                    "basis": "rule_candidate_contextual",
                },
            }
        )
    lex = CX.classify_lexical_fillers(words)
    for c in lex:
        if c["state"] != CX.DISFLUENT:
            continue
        events.append(
            {
                "class": "filler_word",
                "start_s": round(c["start_s"], 3),
                "end_s": round(c["end_s"], 3),
                "score": 1.0,
                "score_kind": SCORE_KIND_RULE,
                "n_windows": 1,
                "modalities_used": ["linguistic"],
                "validation": "lexical filler item with hesitation evidence (adjacent pause, lengthening or comma); "
                "exploratory, compared only with DIHANA d:rellenar marks (folds 1-4)",
                "text": c["token"],
                "basis": "lexical_filler_rule",
                "decision": None,
                "context": {
                    "state": CX.DISFLUENT,
                    "subclass": CX.LEXICAL_FILLER,
                    "basis": "lexical_filler_rule",
                },
            }
        )
    stats = {
        "repetition": CX.demotion_stats(cands),
        "lexical_filler": CX.lexical_stats(lex),
    }
    return events, stats


# ----------------------------------------------------------------------------------------------- o2026_11b repetition
def rule_spans(seg: list[dict]) -> list[tuple[int, int]]:
    """Word-index spans ``(first, last)`` of every immediate exact repetition of 1-4 tokens in a segment (tokens as in
    ``textfeat.rule_rep``: letter runs of the lower-cased text), overlapping spans merged."""
    from mrcd.context_eval.textfeat import _LETTERS

    toks, owner = [], []
    for i, w in enumerate(seg):
        for t in _LETTERS.findall(str(w["text"]).lower()):
            toks.append(t)
            owner.append(i)
    hits = []
    for n in range(1, 5):
        for i in range(len(toks) - 2 * n + 1):
            if toks[i : i + n] == toks[i + n : i + 2 * n]:
                hits.append((owner[i], owner[i + 2 * n - 1]))
    merged: list[list[int]] = []
    for a, b in sorted(hits):
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return [(a, b) for a, b in merged]


def repetition_events(
    words_p1: list[dict] | None, route: str | None = None
) -> list[dict]:
    """Repetition events of ``o2026_11b``: P1 ASR words split at pauses (same segmentation as the o2026_11 units and the
    S3 text route).

    ``rule_p1``: the one-line rule (immediate exact repetition of 1-4 tokens, ``textfeat.rule_rep``); a segment fires
    exactly when ``rule_rep(segment text)`` does, and one event is emitted per (merged) repetition, its interval being the
    repeated words. The score is an indicator (1.0 = the rule fired), not a probability.
    ``rep_ctx_p1``: the S3 text route, one decision per segment (logistic score, its own threshold), interval = segment."""
    route = route or OB_REPETITION_ROUTE
    if route not in REPETITION_ROUTES:
        raise ValueError(f"unknown repetition route {route!r}")
    segs = S3F.segments_from_words(words_p1 or [])
    if not segs:
        return []
    texts = [" ".join(w["text"] for w in s) for s in segs]
    if route == "rule_p1":
        out = []
        for seg in segs:
            for a, b in rule_spans(seg):
                out.append(
                    {
                        "class": "repetition",
                        "start_s": round(float(seg[a]["start"]), 3),
                        "end_s": round(float(seg[b]["end"]), 3),
                        "score": 1.0,
                        "score_kind": SCORE_KIND_RULE,
                        "n_windows": 1,
                        "modalities_used": ["linguistic"],
                        "validation": "lexical rule (immediate exact repetition of 1-4 tokens) on P1 ASR text split at "
                        "pauses; interval = the repeated words; evaluated on DIHANA folds 1-4 only",
                        "text": " ".join(w["text"] for w in seg[a : b + 1]),
                        "basis": "rule_p1_repetition",
                        "decision": None,
                    }
                )
        return out
    sc, d = S3F.load_route().decide(texts)
    scores, dec = [float(x) for x in sc], [int(x) for x in d]
    kind, basis = SCORE_KIND_TEXT_ROUTE, "text_route_repetition"
    validation = (
        "S3 text route rep_ctx_p1 (turn-level model) on a P1 ASR segment split at pauses; "
        "interval = segment span; evaluated on DIHANA folds 1-4 only"
    )
    out = []
    for s, t, score, d in zip(segs, texts, scores, dec):
        if not d:
            continue
        out.append(
            {
                "class": "repetition",
                "start_s": round(float(s[0]["start"]), 3),
                "end_s": round(float(s[-1]["end"]), 3),
                "score": float(np.clip(score, 0.0, 1.0)),
                "score_kind": kind,
                "n_windows": 1,
                "modalities_used": ["linguistic"],
                "validation": validation,
                "text": t,
                "basis": basis,
                "decision": None,
            }
        )
    return out
