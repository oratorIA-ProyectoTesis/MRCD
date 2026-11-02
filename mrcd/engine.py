"""API pública del motor MRCD: superficie mínima para integrarlo en otro proyecto.

Diseño
------
Un proyecto que integra esto no quiere conocer `build_window`, `predict`,
`speaker_f0_center` ni el orden del vector 12-D. Quiere dos cosas: cargar el
motor una vez y pasarle audio. Todo lo demás es detalle interno y puede cambiar
sin romper a nadie.

    from mrcd import DisfluencyEngine

    engine = DisfluencyEngine.load("models/ckpt_audio_text.pt")   # una sola vez
    result = engine.analyze(audio, sample_rate=16000)             # por peticion
    for ev in result.events:
        print(ev.label, ev.start, ev.end, ev.confidence)

Coste de carga
--------------
`load()` trae a memoria el checkpoint y el modelo de Whisper. Es caro (segundos)
y NO debe repetirse por peticion: crea una instancia al arrancar tu servicio y
reutilizala. La instancia es reentrante para lectura, pero `analyze()` no es
thread-safe si compartes el mismo objeto entre hilos sin serializar; para varios
trabajadores, una instancia por proceso.

Honestidad sobre los resultados
-------------------------------
El checkpoint distribuido se entreno con SUPERVISION DEBIL (etiquetas de una
cascada heuristica, no anotacion humana). `AnalysisResult.warning` lo declara y
NO es decorativo: un integrador que muestre estas salidas como «deteccion de
disfluencias» sin ese aviso esta afirmando una exactitud que no se ha medido.
El acuerdo del juez automatico con anotacion humana sigue sin establecerse.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Literal

import numpy as np

from mrcd.constants import (
    BACKGROUND,
    ACOUSTIC_HOP_S,
    CONTEXT_S,
    SAMPLE_RATE,
    STRIDE_S,
    WINDOW_S,
)

Mode = Literal["audio_only", "audio_text", "trimodal", "kinesic_only"]


@dataclass(frozen=True)
class Event:
    """Un fenomeno detectado, fundido a partir de ventanas contiguas.

    Se reportan EVENTOS y no ventanas a proposito: con ventana de 3 s y paso de
    0.5 s cada instante cae en unas seis ventanas, asi que contar ventanas
    multiplicaria por seis cualquier tasa por minuto.
    """

    label: str
    start: float  # s desde el inicio del audio
    end: float
    confidence: float  # probabilidad maxima de la clase en el tramo
    n_windows: int
    text: str = ""  # transcripcion del tramo, si la hay
    modalities: tuple[str, ...] = ()  # ramas que aportaron evidencia
    #: only set when an experimental head is active: what is and is not validated for this label
    validation: str = ""
    #: only set (non-empty) when an EXPERIMENTAL ASR profile produced the transcript; empty for p0 so the
    #: default event schema and values are unchanged
    asr_profile: str = ""

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass(frozen=True)
class Window:
    """Una ventana evaluada. Util para depurar o pintar una linea de tiempo."""

    start: float
    end: float
    label: str
    confidence: float
    probabilities: dict[str, float]
    text: str = ""
    #: True for the extra window that ends exactly at the end of the span (see ``terminal_window``)
    terminal: bool = False
    #: experimental head scores (not calibrated), only when the engine runs with ``head=``
    head_scores: dict | None = None  # {"experimental": True, "calibrated": False, "head": name, "scores": {class: score}}


@dataclass
class AnalysisResult:
    events: list[Event]
    windows: list[Window] = field(default_factory=list)
    transcript: str = ""
    duration: float = 0.0
    mode: str = ""
    warning: str | None = None
    #: how this result was produced: window plan, effective environment (OMP, threads, versions)
    metadata: dict = field(default_factory=dict)
    #: experimental head decisions (Vc); None when the engine runs the default V0
    head: dict | None = None
    #: ASR words with timestamps (dicts); only filled when ``analyze(return_words=True)``, else None
    words: list[dict] | None = None

    def __iter__(self) -> Iterator[Event]:
        return iter(self.events)

    def __len__(self) -> int:
        return len(self.events)

    def by_label(self, label: str) -> list[Event]:
        return [e for e in self.events if e.label == label]

    def rate_per_minute(self, *, legacy_zero: bool = False) -> float | None:
        """Events per minute, or ``None`` ("not available") if the audio is <= 1 s.

        "Absence is never zero": a rate that cannot be measured is not 0.0 events/min. The historical
        behaviour (0.0) stays reproducible with ``legacy_zero=True``.
        """
        if self.duration > 1.0:
            return len(self.events) / (self.duration / 60)
        return 0.0 if legacy_zero else None

    def to_dict(self) -> dict:
        out = {
            "mode": self.mode,
            "duration": self.duration,
            "warning": self.warning,
            "transcript": self.transcript,
            "metadata": self.metadata,
            "events": [
                {
                    "label": e.label,
                    "start": e.start,
                    "end": e.end,
                    "confidence": e.confidence,
                    "n_windows": e.n_windows,
                    "text": e.text,
                    "modalities": list(e.modalities),
                    **({"validation": e.validation} if e.validation else {}),
                    **(
                        {"asr_profile": e.asr_profile, "asr_experimental": True}
                        if e.asr_profile
                        else {}
                    ),
                }
                for e in self.events
            ],
        }
        if self.head is not None:
            out["head"] = self.head
        return out


class DisfluencyEngine:
    """Motor de deteccion de disfluencias. Cargalo una vez, reutilizalo."""

    #: Audio mas corto que esto no se puede analizar: el ramal linguistico mira
    #: una ventana de contexto de CONTEXT_S y sin ella no hay nada que decidir.
    MIN_AUDIO_S = CONTEXT_S + WINDOW_S

    def __init__(
        self,
        net,
        norm,
        labels: list[str],
        asr,
        provenance: dict | None = None,
        head=None,
    ):
        self._head = head  # mrcd.adaptation.LoadedHead or None (V0)
        self._net = net
        self._norm = norm
        self._labels = labels
        self._asr = asr
        self._provenance = provenance or {}
        self._ling_dim = net.ling.proj[0].in_features if net.ling is not None else 0
        self._include_pros = net.pros is not None
        self._context = self._ling_dim > 10

    # ------------------------------------------------------------- carga
    @classmethod
    def load(
        cls,
        checkpoint: str | Path,
        *,
        whisper_size: str = "small",
        device: str | None = None,
        head: str | None = None,
        head_checkpoint: str | Path | None = None,
        head_sha256: str | None = None,
        asr_profile: str = "p0",
    ) -> DisfluencyEngine:
        """Carga el motor. Operacion CARA: hazla una vez al arrancar el servicio.

        `device=None` elige CUDA si esta disponible. Si las librerias de CUDA
        fallan, el ASR cae a CPU por su cuenta y lo avisa por stdout.

        `head=None` (predeterminado) es V0: la salida no cambia. `head="vc"` carga, de forma
        EXPERIMENTAL y opt-in, la cabeza supervisada Vc (DIHANA folds 1-3, turnos, clases
        filler/prolongation/repetition). Su sha256 esta fijado en `mrcd.adaptation.VC_SHA256` y se
        verifica ANTES de deserializar; un archivo modificado se rechaza. `head_checkpoint` solo
        reubica el archivo de Vc; para cualquier otro checkpoint de cabeza hay que dar `head_sha256`.

        `asr_profile="p0"` (predeterminado) es el ASR historico, sin cambios. `"p1_disfluent_prompt"`
        es EXPERIMENTAL y opt-in: solo cambia el `initial_prompt` de Whisper; no esta validado fuera de
        DIHANA (ver `mrcd/asr_profiles.py`). Un perfil desconocido se rechaza con `ValueError`. El perfil,
        el hash del prompt, el modelo y los parametros quedan en `result.metadata["asr_profile"]`.

        Si `OMP_NUM_THREADS` no estaba en el entorno antes de importar numpy, avisa
        (`RuntimeWarning`): el texto del ASR depende de ese valor. No se modifica el entorno.
        """
        import warnings

        import torch

        from mrcd.extractors.linguistic import VerbatimASR
        from mrcd.inference import load_ckpt
        from mrcd.reproducibility import file_sha256, omp_status, omp_warning

        from mrcd.asr_profiles import get_profile

        profile = get_profile(asr_profile)  # reject unknown profiles before any expensive load
        message = omp_warning(omp_status())
        if message:
            warnings.warn(message, RuntimeWarning, stacklevel=2)
        loaded_head = None
        if head is not None:
            from mrcd import adaptation

            base_sha = file_sha256(checkpoint)
            if head == "vc":
                loaded_head = adaptation.load_vc(base_sha, head_checkpoint)
            elif head_checkpoint is not None and head_sha256:
                loaded_head = adaptation.load_head(
                    head_checkpoint, expected_sha256=head_sha256, base_sha256=base_sha, name=str(head)
                )
            else:
                raise ValueError(
                    "head must be None or 'vc'; a custom head needs head_checkpoint and head_sha256"
                )

        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        net, norm, labels, prov = load_ckpt(Path(checkpoint), dev)
        asr = VerbatimASR(
            whisper_size, device="auto" if device is None else device, profile=profile
        )
        engine = cls(net, norm, list(labels), asr, prov, head=loaded_head)
        engine._device = dev
        return engine

    @property
    def mode(self) -> str:
        return self._net.mode

    @property
    def labels(self) -> list[str]:
        return list(self._labels)

    @property
    def warning(self) -> str | None:
        """Aviso de procedencia del checkpoint. Propagalo a tus usuarios."""
        return self._provenance.get("warning")

    # ---------------------------------------------------------- analisis
    def analyze(
        self,
        audio: np.ndarray,
        *,
        sample_rate: int = SAMPLE_RATE,
        asr_chunk_s: float = 8.0,
        keep_windows: bool = False,
        terminal_window: bool = False,
        span: tuple[float, float] | None = None,
        return_words: bool = False,
    ) -> AnalysisResult:
        """Analiza audio mono float32 en [-1, 1] y devuelve los eventos.

        `asr_chunk_s` controla en trozos de cuantos segundos se transcribe.
        No es cosmetico: Whisper borra repeticiones y muletillas cuando ve mucho
        contexto de una vez. Medido en `results/asr_chunk_study.md`. El valor 0
        transcribe el archivo entero y pierde esa evidencia.

        ``terminal_window`` (predeterminado False = V0 exacto): ademas de la rejilla de 0.5 s
        (`inicio = a, a+0.5, ...` mientras `inicio + 3 <= b`) evalua una ventana final que TERMINA
        en `b` (`inicio = b - 3`) cuando la rejilla no cae ahi. La receta del piloto la incluye;
        sin ella el ultimo tramo de hasta 0.5 s de cada segmento nunca queda cubierto por una
        ventana completa. `span=(a, b)` acota la rejilla a un segmento del audio: los rasgos se
        extraen igualmente sobre el audio COMPLETO (como en el piloto) y solo se evaluan las
        ventanas de [a, b]. Sin `span`, [a, b] = [0, duracion].

        Fusion con la rejilla y los eventos: la ventana terminal cuenta como una ventana mas
        (`n_windows`, `confidence`). Cada ventana inaugura la franja `[inicio, inicio + 0.5)`;
        si la ventana terminal sigue a la ultima de la rejilla, la franja de esa ultima se
        recorta a `inicio_terminal` para que los eventos no se solapen.

        ◆ Advertencia sobre `asr_chunk_s`: trozos cortos tambien generan
        repeticiones FALSAS en las costuras (auditado en
        `results/repetition_audit.md`: ~47 % de las repeticiones nuevas a 8 s
        cruzan un limite de trozo, frente a un 6.9 % de base). No lo bajes sin
        medir el efecto en tu dominio.
        """
        if self.mode not in ("audio_only", "audio_text"):
            raise ValueError(
                f"El modo {self.mode} requiere evidencia visual; analyze solo recibe audio."
            )
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim != 1:
            raise ValueError("Se espera audio mono en un arreglo de una dimension.")
        if not np.isfinite(audio).all():
            raise ValueError("El audio debe contener solo muestras finitas.")
        if sample_rate != SAMPLE_RATE:
            raise ValueError(
                f"El motor espera {SAMPLE_RATE} Hz y recibio {sample_rate}. "
                "Remuestrea antes de llamar (p. ej. librosa.resample o ffmpeg): "
                "el remuestreo silencioso aqui ocultaria un desajuste de tu pipeline."
            )
        duration = len(audio) / SAMPLE_RATE
        if duration < self.MIN_AUDIO_S:
            raise ValueError(
                f"Audio de {duration:.1f} s; se necesitan al menos {self.MIN_AUDIO_S:.0f} s. "
                "El ramal linguistico mira una ventana de contexto mas ancha que la de "
                "analisis y sin ella no hay decision que tomar."
            )

        if hasattr(self._asr, "redecode_events"):  # opt-in loop re-decode log is per analysis
            self._asr.redecode_events = []
        words = (
            self._asr.transcribe_chunked(audio, chunk_s=asr_chunk_s)
            if asr_chunk_s
            else self._asr.transcribe(audio)
        )
        windows, head_report = self._evaluate(
            audio, words, keep_windows, terminal_window=terminal_window, span=span
        )
        events = self._merge(windows)
        if head_report is not None:
            events = [self._mark(e, head_report) for e in events]
        transcript = " ".join(w.text for w in words)
        meta = self._metadata(terminal_window, span, asr_chunk_s)
        warning = self.warning
        profile_meta = self._asr_profile_meta(audio, transcript, asr_chunk_s)
        meta["asr_profile"] = profile_meta
        if profile_meta["experimental"]:
            from dataclasses import replace as _replace

            events = [_replace(e, asr_profile=profile_meta["name"]) for e in events]
            # Only an opt-in experimental profile touches `warning`; p0 keeps the historical value.
            from mrcd.asr_profiles import EXPERIMENTAL_NOTE, anomaly_warning

            from mrcd.asr_profiles import loop_redecode_warning

            extra = [
                EXPERIMENTAL_NOTE,
                anomaly_warning(profile_meta["anomalies"]),
                loop_redecode_warning(profile_meta.get("loop_redecode", {}).get("events", [])),
            ]
            warning = " ".join(x for x in [warning, *extra] if x)
        return AnalysisResult(
            events=events,
            windows=windows if keep_windows else [],
            transcript=transcript,
            duration=duration,
            mode=self._net.mode,
            warning=warning,
            metadata=meta,
            head=head_report,
            words=[w.to_dict() for w in words] if return_words else None,
        )

    def _mark(self, e: Event, head_report: dict) -> Event:
        from dataclasses import replace

        if e.label in head_report["unvalidated_classes"]:
            note = "not validated (no compatible human reference); V0 window argmax, not covered by the Vc head"
        elif e.label in head_report["classes"]:
            note = (
                "V0 window argmax; the experimental Vc decision is turn-level (result.head), "
                "uncalibrated, trained on DIHANA"
            )
        else:
            return e
        return replace(e, validation=note)

    def _metadata(self, terminal_window: bool, span, asr_chunk_s: float) -> dict:
        from mrcd.reproducibility import engine_environment, omp_status, omp_warning

        meta = {
            "window_plan": {
                "window_s": WINDOW_S,
                "stride_s": STRIDE_S,
                "terminal_window": bool(terminal_window),
                "span": list(span) if span is not None else None,
                "asr_chunk_s": asr_chunk_s,
            },
            "head": self._head.name if self._head is not None else None,
            "environment": engine_environment(self._asr),
        }
        message = omp_warning(omp_status())
        if message:
            meta["environment"]["warning"] = message
        return meta

    @staticmethod
    def _asr_model_info(asr) -> dict:
        if hasattr(asr, "model_info"):
            return asr.model_info()
        return {
            "model_size": getattr(asr, "model_size", None),
            "model_dir": None,
            "model_bin_sha256": None,
            "compute_type": getattr(asr, "compute_type", None),
            "device": getattr(asr, "device", None),
        }

    def _asr_profile_meta(self, audio: np.ndarray, transcript: str, asr_chunk_s: float) -> dict:
        """Profile, prompt hash, model, parameters, cache key and output anomalies for this run.

        Anomalies (prompt echo, hallucinated loops) are reported here; the transcript is never altered.
        The engine has no ASR cache: `asr_fingerprint.profile_fingerprint` is the key an integrator's cache
        must use so that profiles never mix (`asr_cache.key` is a deprecated alias of it).
        """
        from mrcd.asr_profiles import (
            asr_cache_key,
            audio_sha256,
            detect_asr_anomalies,
            get_profile,
        )

        asr = self._asr
        raw_profile = getattr(asr, "profile", None)
        if raw_profile is None:
            # REVISION7 F-7: an ASR object that exposes no profile is NOT reported as the verified p0.
            return {
                "name": "unknown",
                "experimental": None,
                "validation": "unknown: the ASR object exposes no .profile; decoding parameters are not verified",
                "model": self._asr_model_info(asr),
                "language": getattr(asr, "language", "es"),
                "chunk_s_requested": asr_chunk_s,
                "asr_cache": None,
                "asr_fingerprint": None,
                "anomalies": [],
            }
        profile = get_profile(raw_profile)
        info = self._asr_model_info(asr)
        from mrcd.reproducibility import engine_environment

        env = engine_environment(asr)
        omp = env.get("omp_num_threads")
        fingerprint = asr_cache_key(
            profile,
            audio_sha256=audio_sha256(audio),
            model_bin_sha256=info["model_bin_sha256"],
            compute_type=info["compute_type"],
            device=info["device"],
            omp_num_threads=omp,
            language=getattr(asr, "language", "es"),
            chunk_s=asr_chunk_s,
        )
        out = profile.describe()
        if profile.loop_redecode:  # every chunk whose text was replaced (or could not be) is listed here
            events = list(getattr(asr, "redecode_events", []) or [])
            out["loop_redecode"]["events"] = events
            out["loop_redecode"]["n_replaced"] = sum(1 for e in events if e.get("attempt") is not None)
            out["loop_redecode"]["n_unresolved"] = sum(1 for e in events if e.get("attempt") is None)
            n_nr = sum(1 for e in events if e.get("reproducible") is False)
            out["loop_redecode"]["n_non_reproducible"] = n_nr
            out["loop_redecode"]["reproducible"] = n_nr == 0  # False once any chunk text came from the T=0.4 rung
        out.update(
            {
                "model": info,
                "language": getattr(asr, "language", "es"),
                "chunk_s_requested": asr_chunk_s,
                "omp_num_threads_effective": omp,
                "omp_effective_verified": env.get("effective"),
                "asr_fingerprint": {
                    "profile_fingerprint": fingerprint,
                    "namespace": profile.cache_namespace(),
                    "note": "fingerprint of profile + decode parameters + model + environment + audio; "
                    "the engine has NO ASR cache. Use it as the key of a cache you add.",
                },
                # Deprecated alias kept for consumers of the first version (2026-10-31): `key` ==
                # asr_fingerprint.profile_fingerprint. It does NOT mean a cache exists (engine_cache is None).
                "asr_cache": {
                    "engine_cache": None,
                    "namespace": profile.cache_namespace(),
                    "key": fingerprint,
                    "deprecated": "alias of asr_fingerprint.profile_fingerprint; the engine has no cache",
                },
                "anomalies": detect_asr_anomalies(transcript, profile.initial_prompt),
            }
        )
        if profile.device is None and info.get("device") == "cuda":
            # device="auto" picked the GPU for a profile that is documented as CPU int8: the text can differ
            # (S1: 4/13 units identical to CPU). Only added in that case; the CPU output is unchanged.
            out["device_warning"] = (
                f"Profile {profile.name!r} is documented as CPU int8 but ran on CUDA ({info.get('compute_type')}) because "
                "device='auto' found a usable GPU; the text can differ from the CPU int8 text the models were trained on. "
                "Pass device='cpu' to pin it."
            )
        return out

    def analyze_file(self, path: str | Path, **kw) -> AnalysisResult:
        """Analiza un WAV mono a 16 kHz. Para otros formatos, decodifica tu."""
        import wave

        try:
            wav = wave.open(str(path), "rb")
        except wave.Error as exc:
            raise ValueError(f"{Path(path).name}: se espera WAV PCM de 16 bits") from exc
        with wav as w:
            if w.getnchannels() != 1:
                raise ValueError(f"{Path(path).name}: se espera audio mono")
            if w.getsampwidth() != 2 or w.getcomptype() != "NONE":
                raise ValueError(f"{Path(path).name}: se espera WAV PCM de 16 bits")
            sr = w.getframerate()
            pcm = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
        return self.analyze(pcm.astype(np.float32) / 32768.0, sample_rate=sr, **kw)

    def unit_features(self, clip: np.ndarray, words: list[dict]) -> dict:
        """Unit-level features of one short clip (a "turn"), with the recipe the optimised pipelines were trained on.

        Additive and opt-in: ``analyze`` never calls it. ``clip`` is mono float32 at 16 kHz; ``words`` are ASR word
        dicts with times RELATIVE to the clip. A clip shorter than one window is zero-padded at the end to one
        window (``windowing.plan_unit``, pad mode ``zero``), features are extracted over the padded clip, and the
        window probabilities and pooled head inputs are those of the base checkpoint. Returns ``probs`` (W, C),
        ``pooled`` (W, H), ``pause`` (internal-pause stats over the UNPADDED frames) and ``n_samples`` (unpadded).
        """
        from mrcd.adaptation import predict_pooled
        from mrcd.dataset import acoustic_windows, center_f0, linguistic_windows
        from mrcd.extractors.acoustic import extract_acoustic
        from mrcd.extractors.linguistic import Word, word_features
        from mrcd.extractors.prosody import prosody_windows, speaker_f0_center
        from mrcd.windowing import internal_pause_stats, pad_audio, plan_unit

        clip = np.asarray(clip, dtype=np.float32)
        if clip.ndim != 1 or not np.isfinite(clip).all():
            raise ValueError("clip must be finite mono audio")
        n = len(clip)
        starts, _, n_extra, _ = plan_unit(n, 0.0, n / SAMPLE_RATE)
        padded = pad_audio(clip, n_extra, "zero")
        track = extract_acoustic(padded, use_silero=True)
        frames = track.frame_matrix()
        center = speaker_f0_center(frames)
        ws = [Word(**w) for w in words]
        ling, lpos, lmask = linguistic_windows(
            [w.to_dict() for w in ws], word_features(ws), starts, context=self._context
        )
        tensors = {
            "ac": acoustic_windows(center_f0(frames, center), starts),
            "ling": ling[:, :, : self._ling_dim],
            "lpos": lpos,
            "lmask": lmask,
        }
        if self._include_pros:
            tensors["pros"] = prosody_windows(frames, starts, center)
        probs, pooled = predict_pooled(
            self._net, self._norm, tensors, self._device, bs=1
        )
        return {
            "probs": probs.astype(np.float32),
            "pooled": pooled.astype(np.float32),
            "pause": internal_pause_stats(track.vad, n_valid=int(np.ceil(n / 160))),
            "n_samples": n,
        }

    # ------------------------------------------------------------ interno
    def _evaluate(
        self,
        audio: np.ndarray,
        words,
        keep: bool,
        *,
        terminal_window: bool = False,
        span: tuple[float, float] | None = None,
    ) -> tuple[list[Window], dict | None]:
        from mrcd.inference import predict
        from mrcd.dataset import acoustic_windows, center_f0, linguistic_windows
        from mrcd.extractors.acoustic import extract_acoustic
        from mrcd.extractors.linguistic import word_features
        from mrcd.extractors.prosody import prosody_windows, speaker_f0_center
        from mrcd.live.window_builder import window_text

        duration = len(audio) / SAMPLE_RATE
        if span is None:
            starts = np.arange(0.0, max(duration - WINDOW_S, 0.0) + 1e-9, STRIDE_S)
            end_s = duration
        else:
            a, end_s = float(span[0]), float(span[1])
            if not (0.0 <= a and end_s <= duration + 1e-9 and end_s - a >= WINDOW_S - 1e-9):
                raise ValueError(
                    f"span {span} must satisfy 0 <= a, b <= duration ({duration:.2f} s) and b - a >= {WINDOW_S} s"
                )
            starts = np.arange(a, end_s - WINDOW_S + 1e-9, STRIDE_S)
        n_grid = len(starts)
        if terminal_window and (end_s - WINDOW_S) - starts[-1] > ACOUSTIC_HOP_S:
            starts = np.append(starts, end_s - WINDOW_S)
        # Extraer una sola vez sobre el mismo audio completo que usaba cada ventana.
        frames = extract_acoustic(audio, use_silero=True).frame_matrix()
        center = speaker_f0_center(frames)
        ling, lpos, lmask = linguistic_windows(
            [word.to_dict() for word in words], word_features(words), starts,
            context=self._context,
        )
        if self._ling_dim > ling.shape[-1]:
            raise ValueError(
                f"El checkpoint espera ling={self._ling_dim}, pero el extractor produce {ling.shape[-1]}"
            )
        tensors = {
            "ac": acoustic_windows(center_f0(frames, center), starts),
            "ling": ling[:, :, :self._ling_dim], "lpos": lpos, "lmask": lmask,
        }
        if self._include_pros:
            tensors["pros"] = prosody_windows(frames, starts, center)
        # Mantener el lote de inferencia original evita cambios numericos por lote.
        probabilities = predict(self._net, self._norm, tensors, self._device, bs=1)
        head_scores, head_report = None, None
        if self._head is not None:
            head_scores, head_report = self._run_head(tensors, probabilities, span, end_s, len(starts))
        out: list[Window] = []
        for k, (t0, p) in enumerate(zip(starts, probabilities)):
            i = int(p.argmax())
            out.append(
                Window(
                    start=float(t0),
                    end=float(t0) + WINDOW_S,
                    label=self._labels[i],
                    confidence=float(p[i]),
                    probabilities={l: float(v) for l, v in zip(self._labels, p)},
                    text=window_text(words, float(t0)),
                    terminal=k >= n_grid,
                    head_scores=(
                        {
                            "experimental": True,
                            "calibrated": False,
                            "head": self._head.name,
                            "scores": {c: float(v) for c, v in zip(self._head.classes, head_scores[k])},
                        }
                        if head_scores is not None and keep
                        else None
                    ),
                )
            )
        return out, head_report

    def _run_head(self, tensors, probabilities, span, end_s, n_windows):
        """Experimental Vc: pooled vector of the base net -> head scores -> turn-level decision.

        V0 probabilities, windows and events are NOT touched. Classes without a compatible human
        reference stay with V0 and are listed as unvalidated.
        """
        from mrcd.adaptation import VC_UNCOVERED, predict_pooled

        probs2, pooled = predict_pooled(self._net, self._norm, tensors, self._device, bs=1)
        if not np.allclose(probs2, probabilities, atol=1e-6):
            raise RuntimeError("pooled-feature path diverges from predict(); refusing to score the head")
        scores = self._head.window_scores(pooled)
        a = float(span[0]) if span is not None else 0.0
        report = {
            "experimental": True,
            "provenance": self._head.provenance(),
            "unit": {
                "start": a,
                "end": float(end_s),
                "n_windows": int(n_windows),
                "note": (
                    "The decision is 'present anywhere in this unit'. Thresholds were frozen on DIHANA "
                    "turns (mostly < 10 s); on a long unit presence saturates. Use span= to score a turn-sized segment."
                ),
            },
            "scores_calibrated": False,
            "decision_rule": "max over windows of the head sigmoid score >= frozen per-class threshold",
            "classes": self._head.unit_decision(scores),
            "unvalidated_classes": {
                c: {"source": "V0", "validated": False, "reason": why} for c, why in VC_UNCOVERED.items()
            },
        }
        return scores, report

    def _merge(self, windows: list[Window]) -> list[Event]:
        """Funde ventanas contiguas de la misma clase en un evento SIN SOLAPE.

        ◆ Por que el intervalo se calcula con el paso y no con el ancho de ventana.
        Cada ventana cubre WINDOW_S segundos pero avanza STRIDE_S, asi que seis
        ventanas cubren cada instante. Si el evento se extendiera hasta
        `ultima_ventana.start + WINDOW_S`, dos eventos consecutivos de clases
        distintas se solaparian hasta WINDOW_S - STRIDE_S = 2.5 s, y una linea de
        tiempo construida con eso mostraria dos clases activas a la vez en el mismo
        instante. Se usa `+ STRIDE_S`: cada ventana se atribuye a la franja que
        inaugura, y los eventos embaldosan sin pisarse.

        Contrapartida declarada: el modelo vio WINDOW_S segundos de evidencia para
        decidir, no STRIDE_S. El intervalo reportado es la RESOLUCION de la
        deteccion, no la extension de la evidencia que la sustenta. Para auditar el
        contexto completo, usa `keep_windows=True` y mira las ventanas.
        """
        mods = {
            "audio_only": ("acoustic",),
            "audio_text": ("acoustic", "linguistic"),
            "trimodal": ("acoustic", "linguistic", "kinesic"),
            "kinesic_only": ("kinesic",),
        }.get(self._net.mode, ())
        events, i, n = [], 0, len(windows)
        while i < n:
            if windows[i].label == BACKGROUND:
                i += 1
                continue
            j = i
            while (
                j + 1 < n
                and windows[j + 1].label == windows[i].label
                and windows[j + 1].start - windows[j].start <= STRIDE_S * 1.5
            ):
                j += 1
            run = windows[i : j + 1]
            # Each window owns [start, start + STRIDE_S). The terminal window starts inside the last
            # grid stride, so the grid window before it is cut at the terminal start (no overlap).
            if j + 1 < n and windows[j + 1].terminal:
                end = windows[j + 1].start
            else:
                end = run[-1].start + STRIDE_S
            events.append(
                Event(
                    label=windows[i].label,
                    start=run[0].start,
                    end=end,
                    confidence=max(w.confidence for w in run),
                    n_windows=len(run),
                    text=next((w.text for w in run if w.text), ""),
                    modalities=mods,
                )
            )
            i = j + 1
        return events

    _device: str = "cpu"
