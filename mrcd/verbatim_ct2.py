"""Opt-in ``cw2_ct2`` backend: CrisperWhisper 2.0 large converted to CTranslate2, decoded with upstream ``ctranslate2``.

EXPERIMENTAL and never a default. Plan C (``results/plan_c_20261108``) measured, on 48 DIHANA folds 1-4 turns of at
most 30 s, that the float16 conversion decoded this way gives text identical to the Hugging Face ``crisperwhisper``
pipeline of profile ``cw2`` on every unit, at about half its GPU time (exploratory). What was measured, and is
reproduced here:

* decoder prompt ``[verbatim_1]..[verbatim_5]`` + ``<|startoftranscript|><|es|><|transcribe|><|notimestamps|>``
  (``crisperwhisper/prompt.py``), greedy, ``suppress_blank=False``, the model's ``generation_config.suppress_tokens``,
  ``max_length=448`` (prompt included, as in the Plan C protocol; about 438 new tokens);
* the waveform is zero-padded to 30 s BEFORE the log-mel (as the Hugging Face feature extractor does), not the
  features after it.

Not measured (stated, not hidden): audio longer than 30 s is cut into 30 s windows with 4 s overlap and region
de-duplication (MRCD's rule, not the package's continuation strategy); word times come from CTranslate2 cross-attention
alignment (``faster_whisper.WhisperModel.find_alignment``), and bracketed tags, which are not aligned, take the end time
of the preceding word. Licence: the converted weights are a Derivative Work under the nyra health Non-Commercial
Research License (see ``mrcd.asr_profiles.CW2_LICENSE_NOTE``); they are never distributed with MRCD.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

ENV_DIR = "MRCD_CW2_CT2_DIR"
SAMPLE_RATE = 16000
WINDOW_S = 30.0
OVERLAP_S = 4.0
MAX_NEW_TOKENS = 256  # informational; the decode is bounded by MAX_LENGTH
MAX_LENGTH = 448  # decoder length cap, prompt included, as measured (PROTOCOL_EXPLORATORY_C.md section 3)
#: Exact conversion of Plan C (ctranslate2 4.8.2, transformers 5.22.1); the source snapshot is the pinned cw2 revision.
CONVERSION_COMMAND = (
    "ct2-transformers-converter --model <snapshot of nyralabs/CrisperWhisper2.0_large at the pinned revision> "
    "--output_dir <dir> --quantization float16 --copy_files tokenizer.json preprocessor_config.json added_tokens.json "
    "generation_config.json special_tokens_map.json tokenizer_config.json vocab.json merges.txt normalizer.json LICENSE.md"
)
PROMPT_TAG_PREFIXES = ("[verbatim_", "[intended_")
PROMPT_MARKERS = ("<vtx>", "<evtx>", "<htx>", "<ehtx>", "<ctx>", "<ectx>")


def resolve_model_dir(path: str | os.PathLike | None = None) -> Path:
    p = path or os.environ.get(ENV_DIR)
    if not p:
        raise RuntimeError(
            f"profile 'cw2_ct2' needs a local CTranslate2 conversion of CrisperWhisper 2.0 large: set {ENV_DIR} to its "
            f"directory (created with: {CONVERSION_COMMAND}). Non-commercial research licence; never shipped with MRCD."
        )
    p = Path(p)
    if not (p / "model.bin").is_file() or not (p / "tokenizer.json").is_file():
        raise RuntimeError(
            f"{p} is not a CTranslate2 Whisper directory (model.bin and tokenizer.json expected)"
        )
    return p


def special_ids(vocab: dict[str, int]) -> set[int]:
    """Ids removed from the output text: Whisper control tokens, the mode tags and the prompt markers (same rule as
    ``crisperwhisper.transformers_engine.TransformersEngine._build_special_ids``). Bracketed tags such as ``[UH]`` stay."""
    out = set()
    for tok, i in vocab.items():
        if (
            (tok.startswith("<|") and tok.endswith("|>"))
            or tok.startswith(PROMPT_TAG_PREFIXES)
            or tok in PROMPT_MARKERS
        ):
            out.add(int(i))
    return out


def place_tags(
    kept: list[int], eot: int, words: list[dict]
) -> list[tuple[str | None, int | None, int]]:
    """Order of output items: ``(None, word_index, 0)`` for an aligned word, ``(None, None, tag_id)`` for a tag.

    ``kept`` is the decoded id sequence without special ids; ids >= ``eot`` are tags (not aligned). A tag goes after
    every word whose tokens were all emitted before it.
    """
    ends, c = [], 0
    for w in words:
        c += len(w["tokens"])
        ends.append(c)
    out, wi, seen = [], 0, 0
    for t in kept:
        if t >= eot:
            while wi < len(words) and ends[wi] <= seen:
                out.append((None, wi, 0))
                wi += 1
            out.append((None, None, int(t)))
        else:
            seen += 1
    while wi < len(words):
        out.append((None, wi, 0))
        wi += 1
    return out


class CT2Crisper:
    """Loaded decoder. ``transcribe(audio) -> [(token_text, start_s, end_s, prob)]`` (tags as their literal text)."""

    def __init__(self, model_dir: Path, device: str, compute_type: str):
        from faster_whisper import WhisperModel  # type: ignore
        from faster_whisper.tokenizer import Tokenizer  # type: ignore

        self.model_dir = Path(model_dir)
        self.fw = WhisperModel(
            str(self.model_dir), device=device, compute_type=compute_type
        )
        self.hf = self.fw.hf_tokenizer
        self.tok = Tokenizer(self.hf, True, task="transcribe", language="es")
        vocab = self.hf.get_vocab()
        tags = [vocab[f"[verbatim_{i}]"] for i in range(1, 6)]
        self.prompt = tags + list(self.tok.sot_sequence) + [self.tok.no_timestamps]
        gen = json.loads((self.model_dir / "generation_config.json").read_text("utf8"))
        self.suppress = [
            int(t) for t in gen.get("suppress_tokens") or [] if int(t) >= 0
        ]
        self.special = special_ids(vocab)
        self.eot = self.tok.eot
        self.max_length = MAX_LENGTH

    def features(self, audio: np.ndarray) -> np.ndarray:
        n = int(WINDOW_S * SAMPLE_RATE)
        a = np.asarray(audio, np.float32)[:n]
        mel = self.fw.feature_extractor(np.pad(a, (0, n - len(a))), padding=0)
        return mel[:, : self.fw.feature_extractor.nb_max_frames]

    def decode_window(self, audio: np.ndarray) -> list[tuple[str, float, float, float]]:
        enc = self.fw.encode(self.features(audio))
        res = self.fw.model.generate(
            enc,
            [self.prompt],
            max_length=self.max_length,
            beam_size=1,
            suppress_blank=False,
            suppress_tokens=self.suppress,
            return_scores=False,
            return_no_speech_prob=False,
        )
        kept = [int(t) for t in res[0].sequences_ids[0] if int(t) not in self.special]
        text_tokens = [t for t in kept if t < self.eot]
        n_frames = min(
            int(np.ceil(len(audio) / self.fw.feature_extractor.hop_length)),
            self.fw.feature_extractor.nb_max_frames,
        )
        words = (
            self.fw.find_alignment(self.tok, [text_tokens], enc, n_frames)[0]
            if text_tokens
            else []
        )
        out, last_end = [], 0.0
        for _, wi, tag in place_tags(kept, self.eot, words):
            if wi is not None:
                w = words[wi]
                out.append(
                    (
                        w["word"],
                        float(w["start"]),
                        float(w["end"]),
                        float(w["probability"]),
                    )
                )
                last_end = float(w["end"])
            else:
                out.append(
                    (
                        self.hf.decode([tag], skip_special_tokens=False),
                        last_end,
                        last_end,
                        1.0,
                    )
                )
        return out

    def transcribe(self, audio: np.ndarray) -> list[tuple[str, float, float, float]]:
        size, step = (
            int(WINDOW_S * SAMPLE_RATE),
            int((WINDOW_S - OVERLAP_S) * SAMPLE_RATE),
        )
        if len(audio) <= size:
            return self.decode_window(audio)
        out = []
        for k, start in enumerate(range(0, len(audio), step)):
            seg = audio[start : start + size]
            if len(seg) < SAMPLE_RATE:
                break
            off, new_from = start / SAMPLE_RATE, (0.0 if k == 0 else OVERLAP_S)
            for t, s, e, p in self.decode_window(seg):
                if (s + e) / 2 >= new_from:
                    out.append((t, s + off, e + off, p))
            if start + size >= len(audio):
                break
        return out


def load_crisper_ct2(
    profile, device: str = "auto", compute_type: str | None = None, model_dir=None
):
    """-> (CT2Crisper, device, compute_type). The profile demands CUDA (no silent CPU fallback)."""
    from mrcd.asr_profiles import ASRDeviceUnavailable

    want = profile.device or "cuda"
    if device not in ("auto", want):
        raise ValueError(
            f"profile {profile.name!r} requires device={want!r}, got device={device!r}"
        )
    try:
        import ctranslate2  # type: ignore

        n = ctranslate2.get_cuda_device_count()
    except Exception as exc:  # noqa: BLE001
        raise ASRDeviceUnavailable(
            f"profile {profile.name!r} requires CUDA ({type(exc).__name__}: {exc})"
        ) from exc
    if want == "cuda" and n < 1:
        raise ASRDeviceUnavailable(
            f"profile {profile.name!r} requires CUDA but ctranslate2 reports 0 CUDA devices"
        )
    ct = compute_type or profile.compute_type or "float16"
    return CT2Crisper(resolve_model_dir(model_dir), want, ct), want, ct
