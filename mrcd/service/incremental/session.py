"""Incremental session store (B incremental): idempotent batches assembled by sample offset, early chunk decoding,
progress events, expiry and guaranteed discard.

Privacy: the PCM lives in a ``bytearray`` and the decoded words in a ``ChunkCache``, both in memory only and both
discarded (zeroed / cleared) when the session ends in ANY way. Nothing here touches the filesystem. Progress events
carry counts only, never text, words or the result document (the terminal event only says ``completed``/``failed``;
the document is read through ``GET /v1/sessions/{id}``). Every session endpoint requires the per-session
subscription token issued when the session is opened.

Zeroing guarantee (exact): the controlled buffers are overwritten with zeros when they are no longer needed (the
session PCM, the canonical WAV built at close, the request body accumulator and the per-chunk raw/float32 copies). NOT
controlled, therefore only released: immutable ``bytes`` objects produced by the HTTP framework while reading a body,
the ``BytesIO``/float32 arrays created inside the (unchanged) offline analysis, the engine's own internal copies and
whatever the Python allocator keeps after freeing. See ``PRIVACY_DECISION.md`` section 8.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import secrets
import struct
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Callable

import numpy as np

from mrcd.service.analyzer import AnalysisOptions, MrcdAnalyzer, error_document
from mrcd.service.audio_io import ServiceError
from mrcd.service.contract import SESSION_SCHEMA_VERSION
from mrcd.service.incremental import plan as chunk_plan
from mrcd.service.incremental.cache import ChunkCache, segment_key
from mrcd.service.incremental.cached_asr import CachedASR, asr_namespace, decode_with

log = logging.getLogger(__name__)

SAMPLE_RATE = 16_000
BYTES_PER_SAMPLE = 2
PRIO_FINALIZE, PRIO_CLOSING_CHUNK, PRIO_CHUNK = 0, 1, 2
WAV_HEADER_BYTES = 44


class SessionAbandoned(Exception):
    """Raised at a safe point (before a real chunk decode) when the session was cancelled while closing."""


@dataclass(frozen=True)
class IncrementalConfig:
    max_sessions: int = 4
    idle_timeout_s: float = 120.0
    max_session_seconds: float = 600.0
    max_batch_bytes: int = 1 << 20
    reorder_window_s: float = 64.0
    result_retention_s: float = 300.0
    # Global cap on the audio bytes held by ALL live sessions (PCM buffers + canonical WAVs at close).
    # 128 MiB: four full 10-minute sessions (~154 MB with both copies) cannot all close; the app falls back to offline.
    max_total_audio_bytes: int = 128 << 20


def canonical_wav(pcm: bytes | bytearray | memoryview) -> bytes:
    """The same 44-byte-header WAV the app builds from its buffer (``SessionAudioBuffer.to_wav_bytes``)."""
    return bytes(_canonical_wav(pcm))


def _canonical_wav(pcm: bytes | bytearray | memoryview) -> bytearray:
    n = len(pcm)
    header = (
        b"RIFF"
        + struct.pack("<I", 36 + n)
        + b"WAVEfmt "
        + struct.pack(
            "<IHHIIHH",
            16,
            1,
            1,
            SAMPLE_RATE,
            SAMPLE_RATE * BYTES_PER_SAMPLE,
            BYTES_PER_SAMPLE,
            16,
        )
        + b"data"
        + struct.pack("<I", n)
    )
    out = bytearray(WAV_HEADER_BYTES + n)
    out[:WAV_HEADER_BYTES] = header
    out[WAV_HEADER_BYTES:] = pcm
    return out


def session_namespace(options: AnalysisOptions, real) -> str:
    """Cache namespace: the requested ASR profile name PLUS everything that can change a chunk text. The profile is
    fixed per session today; keeping it in the key is defence in depth against a future per-request profile."""
    return f"{options.asr_profile}|{asr_namespace(real)}"


def _event(session: "IncrementalSession", kind: str, data: dict) -> None:
    session.events.append({"id": len(session.events) + 1, "type": kind, "data": data})


class IncrementalSession:
    def __init__(
        self,
        session_id: str,
        options: AnalysisOptions,
        cfg: IncrementalConfig,
        owner: "SessionRegistry",
    ):
        self.id = session_id
        self.options = options
        self.cfg = cfg
        self._owner = owner
        self.lock = threading.RLock()
        self.status = "open"
        self.buf = bytearray()
        self.batches: dict[
            int, tuple[int, int, str]
        ] = {}  # seq -> (offset_samples, n_samples, sha256)
        self._intervals: list[tuple[int, int]] = []
        self.contiguous = 0
        self.cache = ChunkCache()
        self.events: list[dict] = []
        self.result: dict | None = None
        self.next_enqueue = 0
        self.chunks_decoded = 0
        self.chunks_failed = 0
        self.decode_calls = 0
        self.cache_stats: dict[str, int] | None = None
        self.close_params: tuple[int, str, int] | None = None
        self._wav: bytearray | None = None
        self.token = secrets.token_urlsafe(32)
        self._reserved = 0
        self.created = self.last_activity = time.monotonic()
        self.ended_at: float | None = None
        _event(self, "state", self._counts())

    def token_ok(self, presented: str | None) -> bool:
        return bool(presented) and hmac.compare_digest(
            presented.encode("utf-8", "replace"), self.token.encode()
        )

    def _reserve(self, n_bytes: int) -> None:
        if n_bytes > 0:
            self._owner.reserve_bytes(n_bytes)
            self._reserved += n_bytes

    # ---------------------------------------------------------------- views
    def _counts(self) -> dict:
        return {
            "status": self.status,
            "received_s": round(self._received_samples() / SAMPLE_RATE, 3),
            "chunks_decoded": self.chunks_decoded,
            "chunks_complete": len(chunk_plan.complete_chunks(self.contiguous)),
        }

    def _received_samples(self) -> int:
        return sum(n for _o, n, _s in self.batches.values())

    def missing_seqs(self, upto: int | None = None) -> list[int]:
        top = (max(self.batches) if self.batches else -1) if upto is None else upto
        return [s for s in range(top + 1) if s not in self.batches][:50]

    def state(self) -> dict:
        with self.lock:
            doc = {
                "schema_version": SESSION_SCHEMA_VERSION,
                "session_id": self.id,
                "status": self.status,
                "received_samples": self._received_samples(),
                "contiguous_samples": self.contiguous,
                "batches_received": len(self.batches),
                "missing_seqs": self.missing_seqs(),
                "chunks_decoded": self.chunks_decoded,
                "chunks_complete": len(chunk_plan.complete_chunks(self.contiguous)),
            }
            if self.status in ("completed", "failed"):
                doc["result"] = self.result
            return doc

    @property
    def alive(self) -> bool:
        return self.status in ("open", "closing")

    # ---------------------------------------------------------------- audio in
    def put(
        self, seq: int, offset_samples: int, data: bytes, sha256: str | None
    ) -> dict:
        with self.lock:
            self._require_open()
            n_bytes = len(data)
            if n_bytes == 0 or n_bytes % BYTES_PER_SAMPLE:
                raise ServiceError(
                    "invalid_request",
                    "A batch must be a non-empty whole number of PCM16 samples.",
                    {},
                )
            if n_bytes > self.cfg.max_batch_bytes:
                raise ServiceError(
                    "batch_too_large",
                    "The batch is too large.",
                    {"max_bytes": self.cfg.max_batch_bytes},
                )
            digest = hashlib.sha256(data).hexdigest()
            if sha256 is not None and sha256.lower() != digest:
                raise ServiceError(
                    "invalid_request",
                    "The batch does not match its declared sha256.",
                    {},
                )
            n = n_bytes // BYTES_PER_SAMPLE
            if seq < 0 or offset_samples < 0:
                raise ServiceError(
                    "invalid_request", "seq and offset must be non-negative.", {}
                )
            self.last_activity = time.monotonic()
            known = self.batches.get(seq)
            if known is not None:
                if known == (offset_samples, n, digest):
                    return self._put_reply(duplicate=True)
                raise ServiceError(
                    "sequence_conflict",
                    "This seq was already received with different content.",
                    {"seq": seq},
                )
            end = offset_samples + n
            if end > int(self.cfg.max_session_seconds * SAMPLE_RATE):
                raise ServiceError(
                    "audio_too_long",
                    "The session exceeds the maximum duration.",
                    {"max_duration_s": self.cfg.max_session_seconds},
                )
            if offset_samples > self.contiguous + int(
                self.cfg.reorder_window_s * SAMPLE_RATE
            ):
                raise ServiceError(
                    "offset_mismatch",
                    "The batch starts too far beyond the contiguous audio.",
                    {"seq": seq},
                )
            for a, b in self._intervals:
                if offset_samples < b and end > a:
                    raise ServiceError(
                        "offset_mismatch",
                        "The batch overlaps audio already received.",
                        {"seq": seq},
                    )
            growth = end * BYTES_PER_SAMPLE - len(self.buf)
            if growth > 0:
                self._reserve(growth)
                self.buf.extend(bytes(growth))
            self.buf[offset_samples * BYTES_PER_SAMPLE : end * BYTES_PER_SAMPLE] = data
            self.batches[seq] = (offset_samples, n, digest)
            self._intervals.append((offset_samples, end))
            self._intervals.sort()
            self._advance_contiguous()
            self._enqueue_ready()
            return self._put_reply(duplicate=False)

    def _put_reply(self, *, duplicate: bool) -> dict:
        ready = len(chunk_plan.complete_chunks(self.contiguous))
        return {
            "received_samples": self._received_samples(),
            "contiguous_samples": self.contiguous,
            "chunks_decoded": self.chunks_decoded,
            "chunks_queued": max(
                0, self.next_enqueue - self.chunks_decoded - self.chunks_failed
            ),
            "chunks_complete": ready,
            "duplicate": duplicate,
        }

    def _advance_contiguous(self) -> None:
        pos = self.contiguous
        for a, b in self._intervals:
            if a <= pos < b:
                pos = b
        self.contiguous = pos

    def _require_open(self) -> None:
        if self.status == "open":
            return
        if self.status == "expired":
            raise ServiceError("session_expired", "The session expired.", {})
        raise ServiceError(
            "session_closed",
            "The session no longer accepts audio.",
            {"status": self.status},
        )

    # ---------------------------------------------------------------- early decoding
    def _enqueue_ready(self) -> None:
        ready = chunk_plan.complete_chunks(self.contiguous)
        while self.next_enqueue < len(ready):
            k, start = ready[self.next_enqueue]
            self.next_enqueue += 1
            prio = PRIO_CLOSING_CHUNK if self.status == "closing" else PRIO_CHUNK
            self._owner.submit_task(
                prio, lambda k=k, start=start: self._decode_chunk(k, start)
            )

    def _decode_chunk(self, k: int, start: int) -> None:
        """Runs on the single engine worker."""
        with self.lock:
            if not self.alive:
                return
            lo = start * BYTES_PER_SAMPLE
            raw = bytearray(
                memoryview(self.buf)[
                    lo : lo + chunk_plan.CHUNK_SAMPLES * BYTES_PER_SAMPLE
                ]
            )
        seg = (
            np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
        )  # same conversion as decode_wav
        raw[:] = bytes(len(raw))  # transient copy: overwrite before releasing
        del raw
        try:
            real = self._owner.analyzer.asr_for(self.options)
            ns = session_namespace(self.options, real)
            key = segment_key(ns, start, seg)
            if key not in self.cache:
                words, events = decode_with(real, seg, start)
                self.decode_calls += 1
                with self.lock:
                    if not self.alive:
                        return
                    self.cache.put(key, words, events)
            with self.lock:
                if self.alive:
                    self.chunks_decoded += 1
                    _event(self, "asr_progress", self._counts())
        except Exception as exc:  # noqa: BLE001 - a failed chunk is simply decoded at close
            with self.lock:
                self.chunks_failed += 1
            log.warning("incremental chunk decode failed: %s", type(exc).__name__)
        finally:
            seg.fill(0.0)  # transient float32 copy

    # ---------------------------------------------------------------- close
    def close(
        self, total_samples: int, wav_sha256: str, last_seq: int
    ) -> tuple[int, dict]:
        with self.lock:
            params = (total_samples, wav_sha256.lower(), last_seq)
            if self.status in ("closing", "completed", "failed"):
                if params != self.close_params:
                    raise ServiceError(
                        "close_mismatch",
                        "The session was already closed with other parameters.",
                        {},
                    )
                return (
                    (200, self.state())
                    if self.status != "closing"
                    else (202, {"status": "closing"})
                )
            self._require_open()
            self.last_activity = time.monotonic()
            missing = self.missing_seqs(last_seq)
            if missing or self.contiguous < max(
                (b for _a, b in self._intervals), default=0
            ):
                raise ServiceError(
                    "audio_gap",
                    "The received audio is not contiguous.",
                    {"missing_seqs": missing, "first_missing_offset": self.contiguous},
                )
            if (
                self._received_samples() != total_samples
                or self.contiguous != total_samples
            ):
                raise ServiceError(
                    "close_mismatch",
                    "total_samples does not match the audio received.",
                    {
                        "received_samples": self._received_samples(),
                        "contiguous_samples": self.contiguous,
                    },
                )
            wav = _canonical_wav(
                memoryview(self.buf)[: total_samples * BYTES_PER_SAMPLE]
            )
            if hashlib.sha256(wav).hexdigest() != params[1]:
                wav[:] = bytes(len(wav))
                raise ServiceError(
                    "close_mismatch",
                    "wav_sha256 does not match the audio received.",
                    {},
                )
            try:
                # Reserved only after the sha is verified, so a refused close leaves no reservation behind.
                self._reserve(len(wav))
            except ServiceError:
                wav[:] = bytes(len(wav))
                raise
            self._wav = wav
            self.close_params = params
            self.status = "closing"
            _event(self, "finalizing", {"stage": "full_audio_analysis"})
        self._owner.submit_task(PRIO_FINALIZE, self._finalize)
        return 202, {"status": "closing"}

    def _finalize(self) -> None:
        """Runs on the single engine worker: the OFFLINE path over the canonical WAV, with the chunk cache."""
        with self.lock:
            wav = self._wav
            if self.status != "closing" or wav is None:
                return
        try:
            real = self._owner.analyzer.asr_for(self.options)
            ns = session_namespace(self.options, real)
            result = self._owner.analyzer.analyze_bytes(
                wav,
                self.options,
                asr_wrapper=lambda asr: CachedASR(
                    asr, self.cache, ns, should_stop=self._abandoned
                ),
            )
        except Exception as exc:  # noqa: BLE001
            result = error_document(
                "analysis_failed",
                "Unexpected service failure.",
                details={"exception_type": type(exc).__name__},
                options=self.options,
            )
        del wav
        with self.lock:
            if self.status != "closing":
                return  # cancelled/expired while closing: nothing to publish (the audio is already discarded)
            self.result = result
            self.status = (
                "completed" if result.get("status") == "completed" else "failed"
            )
            self.cache_stats = {"hits": self.cache.hits, "misses": self.cache.misses}
            # Counts and state only: neither the document nor any text travels on the event stream.
            _event(
                self,
                self.status,
                {**self._counts(), "session_id": self.id, "result_available": True},
            )
            self._discard()
            self.ended_at = time.monotonic()

    # ---------------------------------------------------------------- end of life
    def _abandoned(self) -> bool:
        """Safe-point probe for the analysis thread (read without the lock: a plain attribute read)."""
        return self.status != "closing"

    def _discard(self) -> None:
        view = memoryview(self.buf)
        view[:] = bytes(len(view))
        view.release()
        self.buf.clear()
        self.cache.clear()
        wav, self._wav = self._wav, None
        if wav is not None:
            try:
                wav[:] = bytes(len(wav))
            except BufferError:  # still exported to the running analysis; it is released with it
                pass
        if self._reserved:
            self._owner.release_bytes(self._reserved)
            self._reserved = 0

    def end(self, status: str) -> None:
        """cancelled / expired: discard everything and publish the terminal event."""
        with self.lock:
            if not self.alive:
                return
            self.status = status
            _event(self, status, {"status": status})
            self._discard()
            self.ended_at = time.monotonic()


class SessionRegistry:
    def __init__(
        self,
        analyzer: MrcdAnalyzer,
        submit_task: Callable[[int, Callable[[], None]], None],
        cfg: IncrementalConfig | None = None,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.analyzer = analyzer
        self.submit_task = submit_task
        self.cfg = cfg or IncrementalConfig()
        self._clock = clock
        self._sessions: dict[str, IncrementalSession] = {}
        self._lock = threading.Lock()
        self._bytes_lock = threading.Lock()
        self._bytes_in_use = 0
        self._stop = threading.Event()
        self._sweeper = threading.Thread(
            target=self._sweep_loop, name="mrcd-session-sweeper", daemon=True
        )

    def start(self) -> None:
        if not self._sweeper.is_alive():
            self._sweeper.start()

    def reserve_bytes(self, n: int) -> None:
        with self._bytes_lock:
            if self._bytes_in_use + n > self.cfg.max_total_audio_bytes:
                raise ServiceError(
                    "session_limit",
                    "The service-wide audio memory budget is exhausted.",
                    {"max_total_audio_bytes": self.cfg.max_total_audio_bytes},
                )
            self._bytes_in_use += n

    def release_bytes(self, n: int) -> None:
        with self._bytes_lock:
            self._bytes_in_use = max(0, self._bytes_in_use - n)

    @property
    def bytes_in_use(self) -> int:
        with self._bytes_lock:
            return self._bytes_in_use

    def open(self, options: AnalysisOptions) -> IncrementalSession:
        if not self.analyzer.is_loaded(options) or "cuda" in options.asr_profile:
            raise ServiceError(
                "profile_not_supported_incremental",
                "Incremental sessions only accept CPU profiles that were preloaded at start.",
                {"asr_profile": options.asr_profile},
            )
        from mrcd.asr_profiles import get_profile

        if get_profile(options.asr_profile).loop_redecode:
            # Exactness guard (S8 review H8): the sampled rung (T=0.4) of the loop re-decode is not bit-reproducible, and
            # it WAS observed to differ between the early decode and the offline one on real Whisper (p1_noloop).
            raise ServiceError(
                "profile_not_supported_incremental",
                "Profiles with the loop re-decode are not bit-reproducible, so they cannot be decoded early; "
                "use the offline job endpoint.",
                {"asr_profile": options.asr_profile},
            )
        with self._lock:
            active = sum(1 for s in self._sessions.values() if s.alive)
            if active >= self.cfg.max_sessions:
                raise ServiceError(
                    "session_limit",
                    "Too many simultaneous sessions.",
                    {"max_sessions": self.cfg.max_sessions},
                )
            sid = uuid.uuid4().hex
            session = IncrementalSession(sid, options, self.cfg, self)
            self._sessions[sid] = session
        return session

    def get(self, sid: str) -> IncrementalSession:
        with self._lock:
            session = self._sessions.get(sid)
        if session is None:
            raise ServiceError(
                "session_not_found",
                "No such session (sessions are kept in memory only).",
                {},
            )
        return session

    def cancel(self, sid: str) -> None:
        """Idempotent: discards audio and cache, then forgets the session."""
        with self._lock:
            session = self._sessions.pop(sid, None)
        if session is not None:
            session.end("cancelled")

    def sweep(self) -> None:
        now = self._clock()
        with self._lock:
            sessions = list(self._sessions.items())
        for sid, s in sessions:
            if s.status == "open" and now - s.last_activity > self.cfg.idle_timeout_s:
                s.end("expired")
            elif (
                s.ended_at is not None
                and now - s.ended_at > self.cfg.result_retention_s
            ):
                with s.lock:
                    s.result = None
                with self._lock:
                    self._sessions.pop(sid, None)

    def _sweep_loop(self) -> None:
        interval = max(0.05, min(1.0, self.cfg.idle_timeout_s / 4))
        while not self._stop.wait(interval):
            try:
                self.sweep()
            except Exception:  # noqa: BLE001
                log.exception("session sweep failed")

    def shutdown(self) -> None:
        self._stop.set()
        with self._lock:
            sessions = list(self._sessions.values())
            self._sessions.clear()
        for s in sessions:
            s.end("cancelled")

    @property
    def active(self) -> int:
        with self._lock:
            return sum(1 for s in self._sessions.values() if s.alive)
