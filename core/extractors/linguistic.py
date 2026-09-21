"""Extractor lingüístico: ASR verbatim (Faster-Whisper) + rasgos sintácticos.

- Faster-Whisper INT8, word_timestamps=True, condition_on_previous_text=False
  y initial_prompt cargado con muletillas (ver constants.WHISPER_INITIAL_PROMPT).
- Rasgos por palabra ligeros (sin red) y, opcionalmente, embeddings de
  RoBERTa-BNE (PlanTL-GOB-ES/roberta-base-bne) para la rama sintáctica.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import asdict, dataclass

import numpy as np

from core.constants import (EDIT_TERMS, FILLER_LEXICON, FUNCTION_WORDS,
                            WHISPER_INITIAL_PROMPT)


@dataclass
class Word:
    text: str          # tal cual lo emitió el ASR
    norm: str          # minúsculas, sin puntuación ni tildes
    start: float       # s
    end: float         # s
    prob: float
    punct_after: str   # puntuación que cerraba el token ('.', ',', '?', '...', '')

    def to_dict(self):
        return asdict(self)


def normalize(tok: str) -> str:
    tok = unicodedata.normalize("NFKD", tok.lower())
    tok = "".join(c for c in tok if not unicodedata.combining(c))
    return re.sub(r"[^a-zñ\- ]", "", tok).strip("- ").strip()


def _punct(tok: str) -> str:
    m = re.search(r"([.,;:?!…]+)\s*$", tok.strip())
    return m.group(1) if m else ""


# ------------------------------------------------------------------ ASR
class VerbatimASR:
    def __init__(self, model_size: str = "small", device: str = "auto",
                 compute_type: str | None = None, language: str = "es"):
        from faster_whisper import WhisperModel  # type: ignore

        if device == "auto":
            try:
                import ctranslate2  # type: ignore
                device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
            except Exception:
                device = "cpu"
        compute_type = compute_type or ("float16" if device == "cuda" else "int8")
        try:
            self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
            if device == "cuda":   # validar librerías CUDA (cuBLAS/cuDNN) con una inferencia mínima
                list(self.model.transcribe(np.zeros(16000, np.float32), language=language)[0])
        except Exception as exc:
            if device != "cuda":
                raise
            print(f"[asr] GPU no utilizable ({str(exc)[:120]}); usando CPU int8")
            device, compute_type = "cpu", "int8"
            self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
        self.device, self.compute_type, self.model_size = device, compute_type, model_size
        self.language = language

    def detect_language(self, audio: np.ndarray, seconds: float = 30.0) -> tuple[str, float]:
        """Idioma dominante en los primeros `seconds` s -> (código, probabilidad)."""
        seg = audio[: int(seconds * 16000)].astype(np.float32)
        _, info = self.model.transcribe(seg, language=None, beam_size=1, without_timestamps=True)
        return info.language, float(info.language_probability)

    def transcribe(self, audio: np.ndarray) -> list[Word]:
        segments, _ = self.model.transcribe(
            audio.astype(np.float32),
            language=self.language,
            word_timestamps=True,
            condition_on_previous_text=False,
            initial_prompt=WHISPER_INITIAL_PROMPT,
            suppress_tokens=[],          # no suprimir tokens "no informativos"
            vad_filter=False,            # el VAD lo hace la rama acústica
            beam_size=5,
            temperature=0.0,
        )
        words: list[Word] = []
        for seg in segments:
            for w in seg.words or []:
                words.append(Word(w.word.strip(), normalize(w.word), float(w.start),
                                  float(w.end), float(w.probability), _punct(w.word)))
        return [w for w in words if w.norm]


# ------------------------------------------------------ rasgos por palabra
WORD_FEAT_NAMES = (
    "is_filler", "is_function_word", "is_edit_term", "repeats_prev", "is_fragment",
    "clause_end", "log_duration", "asr_prob", "gap_before", "gap_after",
)
WORD_FEAT_DIM = len(WORD_FEAT_NAMES)


def lexicon_for(variety: str = "es-PE") -> set[str]:
    lex = set(FILLER_LEXICON["es"])
    lex |= FILLER_LEXICON.get(variety, set())
    return lex


def word_features(words: list[Word], variety: str = "es-PE") -> np.ndarray:
    """Matriz (N, 10) de rasgos ligeros por palabra (sin red neuronal)."""
    lex = lexicon_for(variety)
    feats = np.zeros((len(words), WORD_FEAT_DIM), dtype=np.float32)
    for i, w in enumerate(words):
        prev = words[i - 1] if i else None
        nxt = words[i + 1] if i + 1 < len(words) else None
        bigram = f"{w.norm} {nxt.norm}" if nxt else ""
        feats[i] = [
            float(w.norm in lex or bigram in lex),
            float(w.norm in FUNCTION_WORDS),
            float(w.norm in EDIT_TERMS or bigram in EDIT_TERMS),
            float(prev is not None and prev.norm == w.norm),
            float(w.text.endswith("-") or (prev is not None and prev.norm and w.norm.startswith(prev.norm) and prev.norm != w.norm)),
            float(w.punct_after in {".", "?", "!", "…", "...", ";", ":"}),
            np.log(max(w.end - w.start, 1e-2)),
            w.prob,
            (w.start - prev.end) if prev else 0.0,
            (nxt.start - w.end) if nxt else 0.0,
        ]
    return feats


class SyntacticEncoder:
    """Embeddings contextuales RoBERTa-BNE por palabra (media de subtokens)."""

    def __init__(self, model_name: str = "PlanTL-GOB-ES/roberta-base-bne", device: str = "cpu"):
        from transformers import AutoModel, AutoTokenizer  # type: ignore

        self.tok = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True)
        self.model = AutoModel.from_pretrained(model_name).to(device).eval()
        self.device = device

    def encode(self, words: list[Word], max_words: int = 400) -> np.ndarray:
        import torch

        out = []
        for s in range(0, len(words), max_words):
            chunk = [w.text for w in words[s:s + max_words]]
            enc = self.tok(chunk, is_split_into_words=True, return_tensors="pt", truncation=True)
            with torch.no_grad():
                h = self.model(**{k: v.to(self.device) for k, v in enc.items()}).last_hidden_state[0]
            ids = enc.word_ids()
            for wi in range(len(chunk)):
                idx = [j for j, x in enumerate(ids) if x == wi]
                out.append(h[idx].mean(0).cpu().numpy() if idx else np.zeros(h.shape[-1]))
        return np.asarray(out, dtype=np.float32)
