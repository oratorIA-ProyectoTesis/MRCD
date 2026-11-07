"""Opt-in verbatim ASR support (profile ``cw2``) and the opt-in ``verbatim_or`` decision route (2026-11-07).

Nothing here is a default. The module is import-light on purpose (no torch/faster-whisper at import time).

Declared BEFORE measuring (``results/verbatim_profile_20261107/PROTOCOL.md``):

* Tag mapping. CrisperWhisper writes non-speech/filled-pause evidence as bracketed tags (``[UH]``). A tag whose
  lower-cased content is in ``FP_TAG_NORM`` is a filled pause and becomes the word in ``FP_TAG_NORM`` (a form the
  MRCD filler lexicon and ``rule_filler`` already know). Any other tag is markup: dropped from the words and
  reported in ``VerbatimASR.dropped_tags``. Tags are never repetition material: the repetition rule sees the text
  with every tag removed.
* Route ``verbatim_or``. For each class, ``decision = classifier OR verbatim_rule``:
  repetition: ``rule_rep(strip_tags(text))``; filler_word: ``rule_filler(strip_tags(text))`` OR at least one
  filled-pause tag. The classifier decision is the one MRCD already produces for the class (V0 for repetition;
  the service decision profile for filler_word).

Licence: CrisperWhisper 2.0 weights and outputs are non-commercial research only (see ``THIRD_PARTY_NOTICES.md``).
"""

from __future__ import annotations

import re

#: Filled-pause tags (lower case, no brackets) -> the lexicon word they stand for. Same set as the survival and
#: benchmark protocols (``FP_TAGS``); the target forms are in ``FILLER_LEXICON["es"]``.
FP_TAG_NORM = {
    "uh": "eh",
    "um": "em",
    "eh": "eh",
    "em": "em",
    "er": "eh",
    "ah": "ah",
    "ahh": "ah",
    "hm": "mm",
    "hmm": "mm",
    "mm": "mm",
    "mmm": "mm",
}
TAG_RE = re.compile(r"\[([^\[\]]*)\]")

ROUTE_NAME = "verbatim_or"
ROUTE_RULE = (
    "per class: classifier decision OR verbatim rule on the tag-stripped CrisperWhisper text "
    "(repetition: rule_rep; filler_word: rule_filler OR a filled-pause tag)"
)
ROUTE_CLASSES = ("repetition", "filler_word")


def strip_tags(raw: str) -> str:
    """Text with every bracketed tag removed (tags are markup, not words)."""
    return " ".join(TAG_RE.sub(" ", raw or "").split())


def has_filled_pause_tag(raw: str) -> bool:
    return any(
        m.group(1).strip().lower() in FP_TAG_NORM for m in TAG_RE.finditer(raw or "")
    )


def map_token(tok: str) -> tuple[str | None, str | None]:
    """One ASR token -> (word, dropped_tag). A filled-pause tag becomes its lexicon word; any other tag is dropped."""
    m = TAG_RE.fullmatch(tok.strip())
    if not m:
        return tok, None
    name = m.group(1).strip().lower()
    return (FP_TAG_NORM[name], None) if name in FP_TAG_NORM else (None, name)


def map_tags(raw: str) -> tuple[list[str], list[str]]:
    """Whole transcript -> (tokens with mapped tags, names of dropped tags)."""
    out, dropped = [], []
    for tok in (raw or "").split():
        w, d = map_token(tok)
        if w is not None:
            out.append(w)
        if d is not None:
            dropped.append(d)
    return out, dropped


def route_decisions(raw_text: str, classifier: dict) -> dict:
    """Opt-in ``verbatim_or`` fusion. ``classifier`` maps class -> 0/1 (the MRCD decision for that class)."""
    from mrcd.context_eval.textfeat import rule_rep
    from mrcd.optimize.blocks import rule_filler

    clean = strip_tags(raw_text)
    rep = int(bool(classifier.get("repetition", 0)) or bool(rule_rep(clean)))
    fill = int(
        bool(classifier.get("filler_word", 0))
        or bool(rule_filler(clean))
        or has_filled_pause_tag(raw_text)
    )
    return {"repetition": rep, "filler_word": fill}


# ------------------------------------------------------------------ backend
def resolve_device(device: str = "auto") -> tuple[str, str]:
    """-> (device, compute_type). GPU float16 when CUDA is usable, otherwise CPU float32 (documented fallback, slow)."""
    if device not in ("auto", "cuda", "cpu"):
        raise ValueError(f"device must be auto|cuda|cpu, got {device!r}")
    cuda = False
    if device != "cpu":
        try:
            import torch  # type: ignore

            cuda = bool(torch.cuda.is_available())
        except Exception:  # noqa: BLE001
            cuda = False
    if device == "cuda" and not cuda:
        from mrcd.asr_profiles import ASRDeviceUnavailable

        raise ASRDeviceUnavailable(
            "device='cuda' requested for cw2 but torch reports no CUDA device"
        )
    return ("cuda", "float16") if cuda else ("cpu", "float32")


def load_crisper(profile, device: str = "auto", compute_type: str | None = None):
    """Load the pinned CrisperWhisper snapshot -> (model, device, compute_type). Needs the optional ``crisperwhisper`` package.

    The snapshot is resolved at ``profile.model_revision`` (local cache first). ``backend="transformers"`` is the one the
    survival and benchmark campaigns used.
    """
    try:
        from crisperwhisper import CrisperWhisperModel  # type: ignore
        from huggingface_hub import snapshot_download  # type: ignore
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "profile 'cw2' needs the optional 'crisperwhisper' package (and torch); install it in a separate "
            "environment (see THIRD_PARTY_NOTICES.md for the licence)"
        ) from exc
    dev, ct = resolve_device(device)
    path = snapshot_download(profile.model_id, revision=profile.model_revision)
    ct = compute_type or ct
    return (
        CrisperWhisperModel(path, backend="transformers", device=dev, compute_type=ct),
        dev,
        ct,
    )
