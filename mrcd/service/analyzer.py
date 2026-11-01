"""File-in / JSON-out analysis through the normal engine path (``DisfluencyEngine.analyze``).

Defaults keep the engine untouched: ASR profile ``p0``, head V0 (no head), ``asr_chunk_s=8.0``,
``terminal_window=False``. The experimental ASR profile P1 and the Vc head are opt-in and every response
that used them says so (``experimental``, per-event ``experimental`` and a warning).

``MrcdAnalyzer.analyze_*`` never raises for expected failures: it returns a contract-valid document with
``status: "failed"`` and a structured ``error``.
"""

from __future__ import annotations

import json
import threading
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

from mrcd.constants import SAMPLE_RATE
from mrcd.service.audio_io import (
    DecodedAudio,
    ServiceError,
    audio_sha256,
    check_array,
    decode_wav,
    inspect_header,
)
from mrcd.service.contract import (
    ANALYSIS_SCHEMA_VERSION,
    ERROR_CODES,
    INTERVAL_SEMANTICS,
    NOT_CALIBRATED_LABEL,
    CONTRACT_REVISION,
    SCORE_KIND,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT = ROOT / "models" / "ckpt_audio_text.pt"
DEFAULT_MAX_DURATION_S = 600.0
ALLOWED_HEADS = (None, "vc")

_MODALITIES_BY_MODE = {
    "audio_only": ("acoustic",),
    "audio_text": ("acoustic", "linguistic"),
    "trimodal": ("acoustic", "linguistic", "kinesic"),
    "kinesic_only": ("kinesic",),
}
_VIDEO_REASON = "service v1 receives audio only; no video was provided or processed"


@dataclass(frozen=True)
class AnalysisOptions:
    asr_profile: str = "p0"
    head: str | None = None
    include_text: bool = False


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _jsonable(value: Any) -> Any:
    """Round-trip through JSON so numpy scalars / tuples become plain JSON values."""

    def default(o: Any):
        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (set, tuple)):
            return list(o)
        return str(o)

    return json.loads(json.dumps(value, default=default, allow_nan=False))


def measure(value: float | int | None, reason: str | None = None) -> dict:
    """``{available, value, unavailable_reason}``: absence is never encoded as ``0``."""
    if value is None:
        return {
            "available": False,
            "value": None,
            "unavailable_reason": reason or "not measured",
        }
    return {"available": True, "value": value, "unavailable_reason": None}


def _modality(available: bool, reason: str | None = None) -> dict:
    return (
        {"available": True, "reason": None}
        if available
        else {"available": False, "reason": reason}
    )


def error_document(
    code: str,
    message: str,
    *,
    details: dict | None = None,
    options: AnalysisOptions | None = None,
    header: dict | None = None,
    digest: str | None = None,
    provenance: dict | None = None,
) -> dict:
    """A contract-valid ``status: "failed"`` document."""
    if code not in ERROR_CODES:
        raise ValueError(f"unknown error code {code!r}")
    options = options or AnalysisOptions()
    header = header or {}
    reason = "audio could not be read" if not header else "not measured"
    return {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "status": "failed",
        "generated_at": _now(),
        "error": {
            "code": code,
            "message": message,
            "retryable": ERROR_CODES[code][1],
            "details": _jsonable(details or {}),
        },
        "request": _request_block(
            options,
            digest,
            header.get("duration_s"),
            header.get("sample_rate_hz"),
            header.get("channels"),
            reason,
        ),
        "events": [],
        "summary": None,
        "modalities": {
            "acoustic": _modality(False, "analysis did not run"),
            "linguistic": _modality(False, "analysis did not run"),
            "kinesic_video": _modality(False, _VIDEO_REASON),
        },
        "provenance": provenance or _provenance(None, None, None, None, None, None),
        "experimental": _experimental(options, None),
        "head_decision": None,
        "transcript": None,
        "warnings": [],
    }


def _request_block(options, digest, duration, rate, channels, reason) -> dict:
    return {
        "audio_sha256": digest,
        "audio_duration_s": measure(
            None if duration is None else round(float(duration), 3), reason
        ),
        "sample_rate_hz": measure(None if rate is None else int(rate), reason),
        "channels": measure(None if channels is None else int(channels), reason),
        "asr_profile": options.asr_profile,
        "head": options.head,
        "include_text": bool(options.include_text),
    }


def _service_block() -> dict:
    import mrcd

    return {
        "name": "mrcd.service",
        "contract": ANALYSIS_SCHEMA_VERSION,
        "contract_revision": CONTRACT_REVISION,
        "mrcd_version": mrcd.__version__,
    }


def _provenance(checkpoint, asr, head, config, environment, _unused) -> dict:
    return {
        "service": _service_block(),
        "checkpoint": checkpoint,
        "asr": asr,
        "head": head,
        "config": config,
        "environment": environment,
    }


def _experimental(options: AnalysisOptions, asr_meta: dict | None) -> dict:
    components = []
    asr_experimental = (
        asr_meta.get("experimental") if asr_meta else options.asr_profile != "p0"
    )
    if asr_experimental:
        components.append(
            f"asr_profile:{(asr_meta or {}).get('name') or options.asr_profile}"
        )
    if options.head is not None:
        components.append(f"head:{options.head}")
    return {"active": bool(components), "components": components}


def _asr_block(meta: dict | None) -> dict | None:
    if not meta:
        return None
    model = dict(meta.get("model") or {})
    model.pop("model_dir", None)  # absolute local path: personal data, not needed
    return _jsonable(
        {
            "profile": meta.get("name"),
            "experimental": meta.get("experimental"),
            "validation": meta.get("validation"),
            "initial_prompt_sha256": meta.get("initial_prompt_sha256"),
            "model": model,
            "language": meta.get("language"),
            "decode": {
                k: meta.get(k)
                for k in (
                    "temperature",
                    "beam_size",
                    "condition_on_previous_text",
                    "vad_filter",
                    "word_timestamps",
                    "chunk_s_requested",
                )
            },
            "profile_fingerprint": (
                (meta.get("asr_fingerprint") or {}).get("profile_fingerprint")
            ),
            "anomalies": meta.get("anomalies") or [],
        }
    )


def build_analysis(
    result: Any,
    *,
    options: AnalysisOptions,
    audio: DecodedAudio | None,
    engine_info: dict,
) -> dict:
    """Map an ``AnalysisResult`` to the v1 document. Pure function: no engine, no I/O."""
    meta = result.metadata or {}
    asr_meta = meta.get("asr_profile") or None
    experimental = _experimental(options, asr_meta)
    mode = result.mode
    mods = _MODALITIES_BY_MODE.get(mode, ())
    has_words = bool((result.transcript or "").strip())

    events = []
    for i, e in enumerate(
        sorted(result.events, key=lambda ev: (ev.start, ev.end)), start=1
    ):
        e_exp = bool(getattr(e, "asr_profile", "")) or options.head is not None
        events.append(
            {
                "id": f"ev-{i:04d}",
                "class": e.label,
                "start_s": round(float(e.start), 3),
                "end_s": round(float(e.end), 3),
                "interval_semantics": INTERVAL_SEMANTICS,
                "score": {
                    "value": round(float(e.confidence), 6),
                    "kind": SCORE_KIND,
                    "calibrated": False,
                    "label": NOT_CALIBRATED_LABEL,
                },
                "n_windows": int(e.n_windows),
                "modalities_used": list(e.modalities),
                "text": (e.text or None) if options.include_text and e.text else None,
                "validation": e.validation or None,
                "experimental": e_exp,
            }
        )

    counts: dict[str, int] = {}
    for ev in events:
        counts[ev["class"]] = counts.get(ev["class"], 0) + 1
    duration = float(result.duration)
    per_minute = (
        measure(round(len(events) / (duration / 60), 4))
        if duration > 1.0
        else measure(
            None, "audio shorter than 1 s: a per-minute rate is not meaningful"
        )
    )

    warnings: list[dict] = [
        {
            "code": "scores_uncalibrated",
            "message": "Scores are the maximum window class probability of an uncalibrated model; "
            "they are not probabilities of correctness.",
        },
        {
            "code": "no_human_validation",
            "message": "Classes are window-level decisions of a model trained with weak (heuristic) "
            "supervision; agreement with human annotation has not been established.",
        },
    ]
    if result.warning:
        warnings.append({"code": "engine_provenance", "message": str(result.warning)})
    if any(ev["class"] == "block" for ev in events):
        warnings.append(
            {
                "code": "class_block_not_measurable",
                "message": "The 'block' class is not measurable with the development corpus (n=2); "
                "there is no evidence that a 'block' event is correct.",
            }
        )
    if not events:
        warnings.append(
            {
                "code": "no_events",
                "message": "No event was emitted; this does not certify fluent speech.",
            }
        )
    if experimental["active"]:
        warnings.append(
            {
                "code": "experimental_components",
                "message": "Experimental, opt-in components were used: "
                + ", ".join(experimental["components"])
                + ". They are not validated outside their development corpus.",
            }
        )
    anomalies = (asr_meta or {}).get("anomalies") or []
    if anomalies:
        warnings.append(
            {
                "code": "asr_anomaly",
                "message": "The ASR output shows anomalies (prompt echo or loops); the transcript was "
                "not altered. Detail: "
                + json.dumps(_jsonable(anomalies), ensure_ascii=False)[:500],
            }
        )
    env = _jsonable(meta.get("environment")) if meta.get("environment") else None
    if env and env.get("warning"):
        warnings.append(
            {"code": "environment_not_pinned", "message": str(env["warning"])}
        )

    head_decision = (
        _jsonable(result.head) if getattr(result, "head", None) is not None else None
    )
    request = _request_block(
        options,
        audio.sha256 if audio else None,
        audio.duration_s if audio else duration,
        audio.sample_rate if audio else SAMPLE_RATE,
        audio.channels if audio else 1,
        "not measured",
    )
    return {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "status": "completed",
        "generated_at": _now(),
        "error": None,
        "request": request,
        "events": events,
        "summary": {
            "n_events": len(events),
            "events_per_minute": per_minute,
            "counts_by_class": counts,
        },
        "modalities": {
            "acoustic": _modality(
                "acoustic" in mods, f"engine mode {mode!r} has no acoustic branch"
            ),
            "linguistic": _modality(
                "linguistic" in mods and has_words,
                f"engine mode {mode!r} has no linguistic branch"
                if "linguistic" not in mods
                else "the ASR returned no words for this audio",
            ),
            "kinesic_video": _modality(False, _VIDEO_REASON),
        },
        "provenance": _provenance(
            engine_info.get("checkpoint"),
            _asr_block(asr_meta),
            _jsonable(head_decision.get("provenance"))
            if head_decision and head_decision.get("provenance")
            else None,
            _jsonable(meta.get("window_plan")),
            env,
            None,
        ),
        "experimental": experimental,
        "head_decision": head_decision,
        "transcript": (result.transcript if options.include_text else None),
        "warnings": warnings,
    }


# ----------------------------------------------------------------------------- engine plumbing
def _default_factory(
    checkpoint: Path, whisper_size: str, device: str | None
) -> Callable[..., Any]:
    def factory(options: AnalysisOptions):
        from mrcd import DisfluencyEngine

        return DisfluencyEngine.load(
            checkpoint,
            whisper_size=whisper_size,
            device=device,
            head=options.head,
            asr_profile=options.asr_profile,
        )

    return factory


def _classify_load_failure(
    exc: BaseException, options: AnalysisOptions
) -> ServiceError:
    frames = " ".join(f.filename for f in traceback.extract_tb(exc.__traceback__))
    kind = type(exc).__name__
    if options.head is not None and "adaptation" in frames:
        return ServiceError(
            "head_unavailable",
            f"The experimental head {options.head!r} could not be loaded.",
            {"exception_type": kind},
        )
    if any(
        k in frames
        for k in ("faster_whisper", "ctranslate2", "huggingface_hub", "linguistic.py")
    ):
        return ServiceError(
            "asr_unavailable",
            "The ASR (Whisper) model could not be loaded. The model must already be in the local cache; "
            "the service never downloads models.",
            {"exception_type": kind},
        )
    return ServiceError(
        "engine_load_failed",
        "The engine could not be loaded.",
        {"exception_type": kind},
    )


class MrcdAnalyzer:
    """Holds one loaded engine per ``(asr_profile, head)`` and serializes calls (the engine is not thread-safe)."""

    def __init__(
        self,
        checkpoint: str | Path = DEFAULT_CHECKPOINT,
        *,
        whisper_size: str = "small",
        device: str | None = None,
        engine_factory: Callable[[AnalysisOptions], Any] | None = None,
        max_duration_s: float = DEFAULT_MAX_DURATION_S,
    ):
        self.checkpoint = Path(checkpoint)
        self._factory = engine_factory or _default_factory(
            self.checkpoint, whisper_size, device
        )
        self._custom_factory = engine_factory is not None
        self.max_duration_s = max_duration_s
        self._engines: dict[tuple, tuple[Any, dict]] = {}
        self._lock = threading.Lock()

    @property
    def loaded(self) -> bool:
        return bool(self._engines)

    # ------------------------------------------------------------------ options
    @staticmethod
    def validate_options(options: AnalysisOptions) -> None:
        from mrcd.asr_profiles import PROFILES

        if options.asr_profile not in PROFILES:
            raise ServiceError(
                "invalid_request",
                f"Unknown asr_profile {options.asr_profile!r}.",
                {"allowed": sorted(PROFILES)},
            )
        if options.head not in ALLOWED_HEADS:
            raise ServiceError(
                "invalid_request",
                f"Unknown head {options.head!r}.",
                {"allowed": ["vc"]},
            )

    def _engine(self, options: AnalysisOptions) -> tuple[Any, dict]:
        key = (options.asr_profile, options.head)
        if key in self._engines:
            return self._engines[key]
        info: dict = {}
        if not self._custom_factory:
            if not self.checkpoint.is_file():
                raise ServiceError(
                    "checkpoint_unavailable",
                    "The engine checkpoint file was not found.",
                    {"checkpoint_file": self.checkpoint.name},
                )
            from mrcd.reproducibility import file_sha256

            info = {
                "sha256": file_sha256(self.checkpoint),
                "file": self.checkpoint.name,
            }
        try:
            engine = self._factory(options)
        except ServiceError:
            raise
        except Exception as exc:  # noqa: BLE001 - classified, never leaked verbatim
            raise _classify_load_failure(exc, options) from exc
        info = {
            "file": info.get("file")
            or getattr(engine, "checkpoint_name", "<injected>"),
            "sha256": info.get("sha256") or getattr(engine, "checkpoint_sha256", None),
            "mode": getattr(engine, "mode", None),
            "labels": list(getattr(engine, "labels", [])),
            "checkpoint_notice": getattr(engine, "warning", None),
        }
        self._engines[key] = (engine, {"checkpoint": _jsonable(info)})
        return self._engines[key]

    # ------------------------------------------------------------------ entry points
    def analyze_bytes(
        self, data: bytes, options: AnalysisOptions | None = None
    ) -> dict:
        options = options or AnalysisOptions()
        header = inspect_header(data)
        digest = audio_sha256(data) if data else None
        try:
            self.validate_options(options)
            decoded = decode_wav(data)
            return self._run(decoded, options)
        except ServiceError as err:
            return error_document(
                err.code,
                err.message,
                details=err.details,
                options=options,
                header=header,
                digest=digest,
            )

    def analyze_file(
        self, path: str | Path, options: AnalysisOptions | None = None
    ) -> dict:
        options = options or AnalysisOptions()
        try:
            data = Path(path).read_bytes()
        except OSError as exc:
            return error_document(
                "invalid_audio_file",
                "The audio file could not be read.",
                details={"exception_type": type(exc).__name__},
                options=options,
            )
        return self.analyze_bytes(data, options)

    def analyze_array(
        self,
        samples: np.ndarray,
        sample_rate: int = SAMPLE_RATE,
        options: AnalysisOptions | None = None,
    ) -> dict:
        options = options or AnalysisOptions()
        try:
            self.validate_options(options)
            arr = check_array(samples, sample_rate)
            raw = np.ascontiguousarray(arr).tobytes()
            decoded = DecodedAudio(
                arr, sample_rate, 1, len(arr) / sample_rate, audio_sha256(raw)
            )
            return self._run(decoded, options)
        except ServiceError as err:
            return error_document(
                err.code, err.message, details=err.details, options=options
            )

    # ------------------------------------------------------------------ core
    def _run(self, audio: DecodedAudio, options: AnalysisOptions) -> dict:
        from mrcd.engine import DisfluencyEngine

        header = {
            "duration_s": audio.duration_s,
            "sample_rate_hz": audio.sample_rate,
            "channels": audio.channels,
        }
        min_s = float(DisfluencyEngine.MIN_AUDIO_S)
        if audio.duration_s < min_s:
            raise ServiceError(
                "audio_too_short",
                f"Audio of {audio.duration_s:.1f} s; the engine needs at least {min_s:.0f} s.",
                {"min_duration_s": min_s, "duration_s": round(audio.duration_s, 3)},
            )
        if audio.duration_s > self.max_duration_s:
            raise ServiceError(
                "audio_too_long",
                f"Audio of {audio.duration_s:.0f} s exceeds the local limit of {self.max_duration_s:.0f} s.",
                {
                    "max_duration_s": self.max_duration_s,
                    "duration_s": round(audio.duration_s, 3),
                },
            )
        with self._lock:
            engine, info = self._engine(options)
            try:
                result = engine.analyze(audio.samples, sample_rate=audio.sample_rate)
            except Exception as exc:  # noqa: BLE001
                err = ServiceError(
                    "analysis_failed",
                    "The engine failed while analysing the audio.",
                    {"exception_type": type(exc).__name__},
                )
                return error_document(
                    err.code,
                    err.message,
                    details=err.details,
                    options=options,
                    header=header,
                    digest=audio.sha256,
                    provenance=_provenance(
                        info.get("checkpoint"), None, None, None, None, None
                    ),
                )
        return build_analysis(result, options=options, audio=audio, engine_info=info)
