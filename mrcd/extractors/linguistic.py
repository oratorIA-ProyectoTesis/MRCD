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

from mrcd.constants import (
    EDIT_TERMS,
    FILLER_LEXICON,
    FUNCTION_WORDS,
    SAMPLE_RATE,
)
from mrcd.asr_profiles import ASRProfile, get_profile


@dataclass
class Word:
    text: str  # tal cual lo emitió el ASR
    norm: str  # minúsculas, sin puntuación ni tildes
    start: float  # s
    end: float  # s
    prob: float
    punct_after: str  # puntuación que cerraba el token ('.', ',', '?', '...', '')

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
def _decode_words(model, audio: np.ndarray, kwargs: dict) -> list[Word]:
    """One ``WhisperModel.transcribe`` call -> words (module level: test doubles call ``VerbatimASR.transcribe`` unbound)."""
    segments, _ = model.transcribe(audio.astype(np.float32), **kwargs)
    words: list[Word] = []
    for seg in segments:
        for w in seg.words or []:
            words.append(
                Word(
                    w.word.strip(),
                    normalize(w.word),
                    float(w.start),
                    float(w.end),
                    float(w.probability),
                    _punct(w.word),
                )
            )
    return [w for w in words if w.norm]


class VerbatimASR:
    def __init__(
        self,
        model_size: str = "small",
        device: str = "auto",
        compute_type: str | None = None,
        language: str = "es",
        profile: "str | ASRProfile | None" = None,
    ):
        # ``None`` -> "p0": the historical decoding parameters, unchanged (see mrcd/asr_profiles.py).
        self.profile = get_profile(profile)
        if self.profile.backend == "crisperwhisper":  # opt-in verbatim profile (cw2); never reached by p0/p1
            from mrcd import verbatim

            self.model, self.device, self.compute_type = verbatim.load_crisper(self.profile, device, compute_type)
            self.model_size, self.language = self.profile.model_id, language
            self._model_bin_sha256 = self._model_dir = None
            self._model_info_done, self.dropped_tags = True, []
            return
        if self.profile.backend == "crisperwhisper_ct2":  # opt-in cw2_ct2 (Plan C); never reached by other profiles
            from mrcd import verbatim_ct2

            self.model, self.device, self.compute_type = verbatim_ct2.load_crisper_ct2(self.profile, device, compute_type)
            self.model_size, self.language = self.profile.model_id, language
            self._model_dir = str(self.model.model_dir)
            self._model_bin_sha256 = None  # hashed lazily by ``model_info()`` (model.bin is ~3 GB)
            self._model_info_done, self.dropped_tags = False, []
            return
        from faster_whisper import WhisperModel  # type: ignore

        strict_device = self.profile.device
        if strict_device is not None:  # opt-in GPU profile: demand it, never fall back silently
            if device not in ("auto", strict_device):
                raise ValueError(
                    f"profile {self.profile.name!r} requires device={strict_device!r}, got device={device!r}"
                )
            device = strict_device
            compute_type = compute_type or self.profile.compute_type
            if device == "cuda":
                try:
                    import ctranslate2  # type: ignore

                    n_cuda = ctranslate2.get_cuda_device_count()
                except Exception as exc:  # noqa: BLE001
                    n_cuda, why = 0, f"{type(exc).__name__}: {exc}"
                else:
                    why = "ctranslate2 reports 0 CUDA devices"
                if n_cuda < 1:
                    from mrcd.asr_profiles import ASRDeviceUnavailable

                    raise ASRDeviceUnavailable(
                        f"profile {self.profile.name!r} requires CUDA but it is unavailable ({why}); "
                        "use p0 / p1_disfluent_prompt on CPU instead"
                    )

        if device == "auto":
            try:
                import ctranslate2  # type: ignore

                device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
            except Exception:
                device = "cpu"
        compute_type = compute_type or ("float16" if device == "cuda" else "int8")
        try:
            self.model = WhisperModel(
                model_size, device=device, compute_type=compute_type
            )
            if (
                device == "cuda"
            ):  # validar librerías CUDA (cuBLAS/cuDNN) con una inferencia mínima
                list(
                    self.model.transcribe(
                        np.zeros(16000, np.float32), language=language
                    )[0]
                )
        except Exception as exc:
            if strict_device == "cuda":
                from mrcd.asr_profiles import ASRDeviceUnavailable

                raise ASRDeviceUnavailable(
                    f"profile {self.profile.name!r} requires CUDA but loading/probing failed "
                    f"({type(exc).__name__}: {str(exc)[:160]}); no CPU fallback for GPU profiles"
                ) from exc
            if device != "cuda":
                raise
            print(f"[asr] GPU no utilizable ({str(exc)[:120]}); usando CPU int8")
            device, compute_type = "cpu", "int8"
            self.model = WhisperModel(
                model_size, device=device, compute_type=compute_type
            )
        self.device, self.compute_type, self.model_size = (
            device,
            compute_type,
            model_size,
        )
        self.language = language
        self._model_bin_sha256: str | None = None
        self._model_dir: str | None = None
        self._model_info_done = False

    def model_info(self) -> dict:
        """Model folder and sha256 of ``model.bin`` (hashed once, lazily). None when it cannot be resolved."""
        if not self._model_info_done:
            from pathlib import Path

            from mrcd.reproducibility import file_sha256

            try:
                d = Path(self._model_dir) if getattr(getattr(self, "profile", None), "backend", "") == "crisperwhisper_ct2" else Path(self.model_size)
                if not d.is_dir():
                    from faster_whisper.utils import download_model  # type: ignore

                    d = Path(download_model(self.model_size, local_files_only=True))
                mb = d / "model.bin"
                self._model_dir = str(d)
                self._model_bin_sha256 = file_sha256(mb) if mb.exists() else None
            except Exception as exc:  # REVISION7 F-8: never swallow silently; log and expose it
                import logging

                self._model_info_error = f"{type(exc).__name__}: {exc}"
                logging.getLogger(__name__).warning(
                    "model_info: could not resolve/hash model.bin for %r (%s)", self.model_size, self._model_info_error
                )
            self._model_info_done = True
        err = getattr(self, "_model_info_error", None)
        return {
            **({"model_info_error": err} if err else {}),
            "model_size": self.model_size,
            "model_dir": self._model_dir,
            "model_bin_sha256": self._model_bin_sha256,
            "compute_type": self.compute_type,
            "device": self.device,
        }

    def detect_language(
        self, audio: np.ndarray, seconds: float = 30.0
    ) -> tuple[str, float]:
        """Idioma dominante en los primeros `seconds` s -> (código, probabilidad)."""
        seg = audio[: int(seconds * 16000)].astype(np.float32)
        _, info = self.model.transcribe(
            seg, language=None, beam_size=1, without_timestamps=True
        )
        return info.language, float(info.language_probability)

    def transcribe(self, audio: np.ndarray) -> list[Word]:
        # objects built without __init__ (test doubles) fall back to the default profile
        profile = get_profile(getattr(self, "profile", None))
        if profile.backend == "crisperwhisper":
            return self._transcribe_crisper(audio)
        if profile.backend == "crisperwhisper_ct2":
            return self._transcribe_crisper_ct2(audio)
        kwargs = profile.decode_kwargs(self.language)
        words = _decode_words(self.model, audio, kwargs)
        if profile.loop_redecode:
            words = VerbatimASR._redecode_if_looped(self, audio, kwargs, words)
        return words

    def _transcribe_crisper(self, audio: np.ndarray) -> list[Word]:
        """Opt-in ``cw2``: verbatim decode; filled-pause tags -> lexicon words (``mrcd.verbatim.FP_TAG_NORM``), other tags dropped.

        CrisperWhisper gives no per-word probability: ``prob`` is 1.0 (neutral, NOT a confidence). Long audio is handled by the
        package's own long-form strategy, not by MRCD's 8 s chunking.
        """
        from mrcd import verbatim

        r = self.model.transcribe(
            audio.astype(np.float32), language=self.language, mode="verbatim", word_timestamps=True, sr=SAMPLE_RATE
        )
        words: list[Word] = []
        for w in r.words or []:
            tok, dropped = verbatim.map_token(w.word)
            if dropped is not None:
                self.dropped_tags.append(dropped)
            if tok is None:
                continue
            words.append(Word(tok.strip(), normalize(tok), float(w.start), float(w.end), 1.0, _punct(tok)))
        return [w for w in words if w.norm]

    def _transcribe_crisper_ct2(self, audio: np.ndarray) -> list[Word]:
        """Opt-in ``cw2_ct2``: same tag mapping as ``cw2`` (``mrcd.verbatim.map_token``) on the CTranslate2 decode of
        ``mrcd.verbatim_ct2``; 30 s windows with 4 s overlap for long audio (its own rule, not MRCD's 8 s chunking)."""
        from mrcd import verbatim

        words: list[Word] = []
        for text, start, end, prob in self.model.transcribe(audio.astype(np.float32)):
            tok, dropped = verbatim.map_token(text)
            if dropped is not None:
                self.dropped_tags.append(dropped)
            if tok is None:
                continue
            p = prob if verbatim.TAG_RE.fullmatch(text.strip()) is None else 1.0
            words.append(Word(tok.strip(), normalize(tok), float(start), float(end), float(p), _punct(tok)))
        return [w for w in words if w.norm]

    def _redecode_if_looped(self, audio: np.ndarray, kwargs: dict, words: list[Word]) -> list[Word]:
        """Opt-in (profiles ``*_noloop``): re-decode a chunk whose text looks like a repetition loop.

        First clean attempt of ``LOOP_REDECODE_LADDER`` replaces the chunk; if none is clean the ORIGINAL words are kept.
        Every replacement or failure is appended to ``self.redecode_events`` (reset by ``transcribe_chunked``/the engine).
        """
        from mrcd.asr_profiles import LOOP_REDECODE_LADDER, LOOP_REDECODE_SEED, chunk_looped

        if not chunk_looped(" ".join(w.text for w in words)):
            return words
        if not hasattr(self, "redecode_events"):
            self.redecode_events = []
        import ctranslate2  # type: ignore

        for j, override in enumerate(LOOP_REDECODE_LADDER, start=1):
            ctranslate2.set_random_seed(LOOP_REDECODE_SEED)
            alt = _decode_words(self.model, audio, {**kwargs, **override})
            if not chunk_looped(" ".join(w.text for w in alt)):
                self.redecode_events.append(
                    {
                        "attempt": j,
                        "audio_s": round(len(audio) / SAMPLE_RATE, 3),
                        "chunk_offset_s": round(float(getattr(self, "_chunk_offset_s", 0.0)), 3),
                        "words_before": len(words),
                        "words_after": len(alt),
                        # a rung that samples (T > 0) is NOT bit-reproducible even with the fixed seed
                        "reproducible": float(override.get("temperature", 0.0)) == 0.0,
                    }
                )
                return alt
        self.redecode_events.append(
            {
                "attempt": None,
                "kept_original": True,
                "audio_s": round(len(audio) / SAMPLE_RATE, 3),
                "chunk_offset_s": round(float(getattr(self, "_chunk_offset_s", 0.0)), 3),
                "words_before": len(words),
                "reproducible": True,  # the original (deterministic) text is kept
            }
        )
        return words

    def transcribe_chunked(
        self, audio: np.ndarray, chunk_s: float = 8.0, overlap_s: float = 2.0
    ) -> list[Word]:
        """Transcribe en trozos solapados y conserva solo la region nueva de cada uno.

        Por que no transcribir el audio entero: Whisper se entrena para producir
        texto legible, y cuanto mas contexto ve de una vez, mas «corrige» hacia
        prosa fluida. Medido sobre una sesion de 90 s (results/asr_chunk_study.md):
        con trozos de 8-13 s sobreviven las 2 repeticiones que el hablante dijo
        realmente; con 30 s o con el archivo entero sobreviven 0. Las repeticiones
        no las pierde el clasificador, las borra el ASR antes.

        Importa para el entrenamiento: `detect_repetitions` opera sobre estas
        palabras, asi que una repeticion que el ASR no transcribe no puede
        etiquetarse, y la clase queda sub-representada en la supervision debil.

        ◆ Deduplicacion por REGION, no por texto: una repeticion legitima («la la»)
        es indistinguible de un duplicado de solape si se compara texto, y
        deduplicar asi borraria justo el fenomeno a detectar. El solape solo da
        contexto al ASR; esas palabras ya las aporto el trozo anterior.
        """
        self.redecode_events = []
        self._chunk_offset_s = 0.0
        if getattr(getattr(self, "profile", None), "backend", "faster_whisper") in ("crisperwhisper", "crisperwhisper_ct2"):
            return self.transcribe(audio)  # cw2: the package's long-form chunking; cw2_ct2: its own 30 s windows
        if chunk_s <= 0 or len(audio) <= chunk_s * SAMPLE_RATE:
            return self.transcribe(audio)

        step = int((chunk_s - overlap_s) * SAMPLE_RATE)
        size = int(chunk_s * SAMPLE_RATE)
        out: list[Word] = []
        for k, start in enumerate(range(0, len(audio), step)):
            seg = audio[start : start + size]
            if len(seg) < SAMPLE_RATE:  # < 1 s no aporta
                break
            offset = start / SAMPLE_RATE
            self._chunk_offset_s = offset  # lets the opt-in re-decode record where the chunk sits
            new_from = 0.0 if k == 0 else overlap_s
            for w in self.transcribe(seg):
                if (w.start + w.end) / 2 < new_from:
                    continue
                out.append(
                    Word(
                        w.text,
                        w.norm,
                        w.start + offset,
                        w.end + offset,
                        w.prob,
                        w.punct_after,
                    )
                )
        self._chunk_offset_s = 0.0
        out.sort(key=lambda w: w.start)
        return out


# ------------------------------------------------------ rasgos por palabra
WORD_FEAT_NAMES = (
    "is_filler",
    "is_function_word",
    "is_edit_term",
    "repeats_prev",
    "is_fragment",
    "clause_end",
    "log_duration",
    "asr_prob",
    "gap_before",
    "gap_after",
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
            float(
                w.text.endswith("-")
                or (
                    prev is not None
                    and prev.norm
                    and w.norm.startswith(prev.norm)
                    and prev.norm != w.norm
                )
            ),
            float(w.punct_after in {".", "?", "!", "…", "...", ";", ":"}),
            np.log(max(w.end - w.start, 1e-2)),
            w.prob,
            (w.start - prev.end) if prev else 0.0,
            (nxt.start - w.end) if nxt else 0.0,
        ]
    return feats


class SyntacticEncoder:
    """Embeddings contextuales RoBERTa-BNE por palabra (media de subtokens)."""

    def __init__(
        self, model_name: str = "PlanTL-GOB-ES/roberta-base-bne", device: str = "cpu"
    ):
        from transformers import AutoModel, AutoTokenizer  # type: ignore

        self.tok = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True)
        self.model = AutoModel.from_pretrained(model_name).to(device).eval()
        self.device = device

    def encode(self, words: list[Word], max_words: int = 400) -> np.ndarray:
        import torch

        out = []
        for s in range(0, len(words), max_words):
            chunk = [w.text for w in words[s : s + max_words]]
            enc = self.tok(
                chunk, is_split_into_words=True, return_tensors="pt", truncation=True
            )
            with torch.no_grad():
                h = self.model(
                    **{k: v.to(self.device) for k, v in enc.items()}
                ).last_hidden_state[0]
            ids = enc.word_ids()
            for wi in range(len(chunk)):
                idx = [j for j, x in enumerate(ids) if x == wi]
                out.append(
                    h[idx].mean(0).cpu().numpy() if idx else np.zeros(h.shape[-1])
                )
        return np.asarray(out, dtype=np.float32)
