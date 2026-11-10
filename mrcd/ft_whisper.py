"""Opt-in ``ft_small_v1`` ASR profile: the Plan B verbatim fine-tune of ``openai/whisper-small`` (CTranslate2 int8).

EXPERIMENTAL and never a default (``results/plan_f_20261108/PROTOCOL_EXPLORATORY_F.md``). The model is
``openai/whisper-small`` (MIT) fully fine-tuned on DIHANA folds 1-4 human transcripts only (CC BY 4.0; no CrisperWhisper
output was used, so it is not a Derivative Work of that non-commercial licence), best checkpoint by dev loss (step 200 of
the capped run, ``results/plan_b_20261108``), converted with ``ct2-transformers-converter --quantization int8``. The
weights are never shipped with MRCD: the converted directory is located through ``MRCD_FT_CT2_DIR``, or, when unset, by a
repository-relative convention (a ``_models_ft/ct2/capped_best_int8`` directory next to the repository checkout).

Decoding is the ordinary faster-whisper path of ``p0`` with the prompt removed (see ``asr_profiles.PROFILES``). The model
identity is the sha256 of its ``model.bin``, hashed lazily by ``VerbatimASR.model_info()`` and part of the ASR cache key.

Filled-pause tokens: the model writes DIHANA ``<f tipo=a|e|m|sin_identificar/>`` as ``ah | eh | mm | eh`` (Plan B target
convention, ``scripts/verbatim_ft/vtext.FILLER_WORD``). ``FT_FILLER_TOKENS`` documents that convention; recognition in the
decision profile ``o2026_13`` uses the engine's ``NONLEXICAL_FILLER_RE`` (a superset that also accepts ``ehh``, ``hmm``, ...).
"""

from __future__ import annotations

import os
from pathlib import Path

ENV_DIR = "MRCD_FT_CT2_DIR"
#: converted-model directory looked up next to the repository when the environment variable is not set
DEFAULT_DIR_PARTS = ("_models_ft", "ct2", "capped_best_int8")
BASE_MODEL_ID = "openai/whisper-small"
MODEL_ID = "openai/whisper-small fine-tuned on DIHANA folds 1-4 (Plan B step 200), CTranslate2 int8, local"
LICENSE_NOTE = (
    "Base model openai/whisper-small is MIT. Fine-tuned only on DIHANA human transcripts (CC BY 4.0, credit: DIHANA "
    "consortium); no CrisperWhisper output was used. DIHANA licence does not address data protection of the recorded "
    "voices: the weights are not published and are never shipped with MRCD."
)
#: filled-pause words the model was trained to write (DIHANA f tipo -> written token)
FT_FILLER_TOKENS = {"a": "ah", "e": "eh", "m": "mm", "sin_identificar": "eh"}
CONVERSION_COMMAND = (
    "ct2-transformers-converter --model <Plan B best checkpoint (step 200 of the capped run)> --output_dir <dir> "
    "--quantization int8 --copy_files tokenizer.json preprocessor_config.json"
)


def _is_ct2_dir(p: Path) -> bool:
    return (p / "model.bin").is_file() and (p / "tokenizer.json").is_file()


def default_dir_candidates() -> list[Path]:
    """Repository-relative candidates (the main checkout and the ``_wt/<worktree>`` layout), never an absolute literal."""
    repo = Path(__file__).resolve().parents[1]
    return [
        repo.parent.joinpath(*DEFAULT_DIR_PARTS),
        repo.parent.parent.joinpath(*DEFAULT_DIR_PARTS),
    ]


def resolve_model_dir(path: str | os.PathLike | None = None) -> Path:
    """The CTranslate2 directory of the fine-tuned model: ``path``, else ``$MRCD_FT_CT2_DIR``, else the repo-relative
    convention. Raises ``RuntimeError`` (never silently falls back to another model) when none is a CT2 Whisper dir."""
    given = path or os.environ.get(ENV_DIR)
    if given:
        p = Path(given)
        if not _is_ct2_dir(p):
            raise RuntimeError(
                f"{p} is not a CTranslate2 Whisper directory (model.bin and tokenizer.json expected); set {ENV_DIR} "
                f"to the converted fine-tune (created with: {CONVERSION_COMMAND})"
            )
        return p
    for c in default_dir_candidates():
        if _is_ct2_dir(c):
            return c
    raise RuntimeError(
        f"profile 'ft_small_v1' needs the local CTranslate2 conversion of the Plan B fine-tune: set {ENV_DIR} to its "
        f"directory (created with: {CONVERSION_COMMAND}); the default location "
        f"'{'/'.join(DEFAULT_DIR_PARTS)}' next to the repository was not found. Weights are never shipped with MRCD."
    )
