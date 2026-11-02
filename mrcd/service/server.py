"""Minimal local development HTTP server for the MRCD service (FastAPI, already installed in the venv).

Scope: ``localhost`` development only. No authentication, no TLS, no API keys. The server refuses to bind a
non-loopback address unless explicitly told to. Raw audio is decoded in memory and dropped as soon as the job
finishes; nothing is written to disk. Job results (JSON only, no audio, transcript only if the request asked
for it) are kept in memory, bounded, and vanish when the process stops.

    POST /v1/jobs?asr_profile=p0&head=&include_text=false&include_out_of_scope=false&decision=false&text_route=&decision_profile=o2026_11   body = raw WAV bytes  -> 202 job status
    GET  /v1/jobs/{job_id}                                                         -> job status (+ result)
    GET  /v1/health, GET /v1/schema
"""

from __future__ import annotations

import asyncio
import itertools
import json
import logging
import queue
import threading
import uuid
from collections import OrderedDict
from datetime import datetime, timezone
from typing import Any, Callable

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from mrcd.service.analyzer import AnalysisOptions, MrcdAnalyzer, error_document
from mrcd.service.decision_profile import DEFAULT_DECISION_PROFILE
from mrcd.service.audio_io import ServiceError
from mrcd.service.contract import ERROR_CODES, JOB_SCHEMA_VERSION, SCHEMA
from mrcd.service.incremental.routes import register as register_session_routes
from mrcd.service.incremental.session import IncrementalConfig, SessionRegistry
from mrcd.service.incremental.warmup import warm_up

LOOPBACK = ("127.0.0.1", "localhost", "::1")
DEFAULT_MAX_BODY_BYTES = 64 * 1024 * 1024
MAX_RETAINED_JOBS = 32
MAX_QUEUED_JOBS = 8
# A queued POST /v1/jobs (the app's offline fallback) is promoted to the front class after this long, so it
# cannot starve behind chunk decodes of other sessions. Documented bound for a job:
# JOB_PROMOTE_AFTER_S + the one task running when it is promoted + earlier closes/promoted jobs.
JOB_PROMOTE_AFTER_S = 20.0


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class JobRunner:
    """One worker thread (the engine is not thread-safe) fed by a priority queue.

    Priorities: 0 session close (and promoted jobs), 1 chunks of a closing session, 2 chunks of an open session,
    3 ``POST /v1/jobs``. FIFO inside each priority. A close therefore never waits behind queued chunks of any session:
    its wait is bounded by the ONE task running on the worker when it is enqueued (a chunk decode, or an analysis). Every engine call (chunk decode or full analysis) goes through this one thread.
    """

    def __init__(self, analyzer: MrcdAnalyzer, promote_after_s: float = JOB_PROMOTE_AFTER_S):
        self.analyzer = analyzer
        self.promote_after_s = promote_after_s
        self._jobs: OrderedDict[str, dict] = OrderedDict()
        self._queue: queue.PriorityQueue = queue.PriorityQueue()
        self._seq = itertools.count()
        self._queued_jobs = 0
        self._lock = threading.Lock()
        self._thread = threading.Thread(
            target=self._loop, name="mrcd-service-worker", daemon=True
        )
        self._thread.start()

    def submit_task(self, priority: int, fn: Callable[[], None]) -> None:
        self._queue.put((priority, next(self._seq), fn))

    def call(self, fn: Callable[[], Any], timeout: float | None = None) -> Any:
        """Runs ``fn`` on the worker thread (priority 0) and returns its result (used for the warm-up)."""
        done, box = threading.Event(), {}

        def task() -> None:
            try:
                box["value"] = fn()
            except BaseException as exc:  # noqa: BLE001
                box["error"] = exc
            finally:
                done.set()

        self.submit_task(0, task)
        if not done.wait(timeout):
            raise TimeoutError("engine worker did not finish in time")
        if "error" in box:
            raise box["error"]
        return box.get("value")

    def submit(self, data: bytes, options: AnalysisOptions) -> dict:
        job_id = uuid.uuid4().hex
        job = {
            "schema_version": JOB_SCHEMA_VERSION,
            "job_id": job_id,
            "status": "queued",
            "created_at": _now(),
            "result": None,
            "_data": data,
            "_options": options,
        }
        with self._lock:
            if self._queued_jobs >= MAX_QUEUED_JOBS:
                raise ServiceError(
                    "queue_full",
                    "Too many pending jobs; retry later.",
                    {"max_queued": MAX_QUEUED_JOBS},
                )
            self._queued_jobs += 1
            self._jobs[job_id] = job
            self._evict()
        self.submit_task(3, lambda: self._run_job(job_id))
        promote = threading.Timer(
            self.promote_after_s,
            lambda: self.submit_task(0, lambda: self._run_job(job_id)),
        )
        promote.daemon = True
        promote.start()
        job["_promote"] = promote
        return self.public(job_id)

    def public(self, job_id: str) -> dict | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            return {k: v for k, v in job.items() if not k.startswith("_")}

    def _evict(self) -> None:
        while len(self._jobs) > MAX_RETAINED_JOBS:
            for jid, job in self._jobs.items():
                if job["status"] in ("completed", "failed"):
                    del self._jobs[jid]
                    break
            else:
                return

    def _loop(self) -> None:
        while True:
            _prio, _n, fn = self._queue.get()
            try:
                fn()
            except Exception:  # noqa: BLE001 - the worker must never die
                logging.getLogger(__name__).exception("engine worker task failed")

    def _run_job(self, job_id: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job["status"] != "queued":
                return  # already run (the promoted copy and the original are both queued)
            self._queued_jobs = max(0, self._queued_jobs - 1)
            job["status"] = "running"
            promote = job.pop("_promote", None)
            data, options = job["_data"], job["_options"]
        if promote is not None:
            promote.cancel()
        try:
            result = self.analyzer.analyze_bytes(data, options)
        except Exception as exc:  # noqa: BLE001 - the worker must never die
            result = error_document(
                "analysis_failed",
                "Unexpected service failure.",
                details={"exception_type": type(exc).__name__},
                options=options,
            )
        with self._lock:
            job["result"] = result
            job["status"] = result["status"]
            job["_data"] = None  # raw audio is not retained after the job ends


EngineWorker = JobRunner


def _query_options(params: Any) -> AnalysisOptions:
    raw_head = (params.get("head") or "").strip().lower()
    head = None if raw_head in ("", "none", "v0") else raw_head
    text = (params.get("include_text") or "false").strip().lower()
    if text not in ("true", "false"):
        raise ServiceError(
            "invalid_request",
            "include_text must be 'true' or 'false'.",
            {"param": "include_text"},
        )

    def _flag(name: str) -> bool:
        v = (params.get(name) or "false").strip().lower()
        if v not in ("true", "false"):
            raise ServiceError(
                "invalid_request", f"{name} must be 'true' or 'false'.", {"param": name}
            )
        return v == "true"

    route = (params.get("text_route") or "").strip().lower()
    options = AnalysisOptions(
        asr_profile=(params.get("asr_profile") or "p0").strip(),
        head=head,
        include_text=(text == "true"),
        output_scope="all" if _flag("include_out_of_scope") else "evaluated",
        decision=_flag("decision"),
        text_route=None if route in ("", "none") else route,
        decision_profile=(params.get("decision_profile") or DEFAULT_DECISION_PROFILE).strip().lower(),
    )
    MrcdAnalyzer.validate_options(options)
    return options


def create_app(
    analyzer: MrcdAnalyzer,
    *,
    max_body_bytes: int = DEFAULT_MAX_BODY_BYTES,
    cors_origins: tuple[str, ...] = (),
    enable_docs: bool = False,
    incremental: IncrementalConfig | None = None,
):

    app = FastAPI(
        title="MRCD service (local development)",
        description=(
            "Local development surface of the MRCD engine (contract mrcd.analysis/v1). "
            "No authentication. Scores are uncalibrated softmax scores, not probabilities."
        ),
        docs_url="/docs" if enable_docs else None,
        redoc_url=None,
        openapi_url="/openapi.json" if enable_docs else None,
    )
    if cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(cors_origins),
            allow_methods=["GET", "POST", "PUT", "DELETE"],
            allow_headers=["Content-Type", "X-Offset-Samples", "X-Content-SHA256", "Last-Event-ID"],
            allow_credentials=False,
        )
    runner = JobRunner(analyzer)
    app.state.runner = runner
    cfg = incremental or IncrementalConfig()
    registry = SessionRegistry(analyzer, runner.submit_task, cfg) if incremental else None
    if registry is not None:
        registry.start()
    app.state.registry = registry
    app.state.warmed_up = False

    def fail(err: ServiceError, options: AnalysisOptions | None = None) -> JSONResponse:
        doc = error_document(
            err.code, err.message, details=err.details, options=options
        )
        return JSONResponse(
            doc,
            status_code=ERROR_CODES[err.code][0],
            headers={"Cache-Control": "no-store"},
        )

    @app.get("/v1/health")
    async def health():
        return {
            "status": "ok",
            "contract": SCHEMA["$id"],
            "engine_loaded": analyzer.loaded,
            "engines_loaded": analyzer.loaded_keys(),
            "warmed_up": app.state.warmed_up,
            "incremental": registry is not None,
            "torch_threads": _torch_threads(),
            "active_sessions": registry.active if registry else 0,
        }

    @app.get("/v1/schema")
    async def schema():
        return SCHEMA

    @app.post(
        "/v1/jobs",
        status_code=202,
        summary="Submit a mono 16 kHz PCM16 WAV for analysis",
        openapi_extra={
            "requestBody": {
                "required": True,
                "content": {
                    "audio/wav": {"schema": {"type": "string", "format": "binary"}}
                },
            },
            "parameters": [
                {
                    "name": name,
                    "in": "query",
                    "required": False,
                    "schema": schema,
                    "description": desc,
                }
                for name, schema, desc in (
                    (
                        "asr_profile",
                        {"type": "string", "default": "p0"},
                        "ASR profile; non-default profiles are experimental",
                    ),
                    (
                        "head",
                        {"type": "string", "default": "v0"},
                        "classification head; 'vc' is experimental",
                    ),
                    (
                        "include_text",
                        {
                            "type": "string",
                            "enum": ["false", "true"],
                            "default": "false",
                        },
                        "include ASR text in the result",
                    ),
                    (
                        "include_out_of_scope",
                        {
                            "type": "string",
                            "enum": ["false", "true"],
                            "default": "false",
                        },
                        "also return unvalidated out-of-scope classes",
                    ),
                    (
                        "decision",
                        {
                            "type": "string",
                            "enum": ["false", "true"],
                            "default": "false",
                        },
                        "add the optional three-state decision",
                    ),
                    (
                        "text_route",
                        {"type": "string"},
                        "experimental text route, e.g. rep_ctx_p1",
                    ),
                    (
                        "decision_profile",
                        {"type": "string", "enum": ["o2026_11", "v0"], "default": "o2026_11"},
                        "decision profile: o2026_11 (service default) or v0 (historical engine decision)",
                    ),
                )
            ],
        },
    )
    async def create_job(request: Request):
        try:
            options = _query_options(request.query_params)
        except ServiceError as err:
            return fail(err)
        declared = request.headers.get("content-length")
        if declared and declared.isdigit() and int(declared) > max_body_bytes:
            return fail(
                ServiceError(
                    "payload_too_large",
                    "The request body is too large.",
                    {"max_bytes": max_body_bytes},
                ),
                options,
            )
        chunks, total = [], 0
        async for chunk in request.stream():
            total += len(chunk)
            if total > max_body_bytes:
                return fail(
                    ServiceError(
                        "payload_too_large",
                        "The request body is too large.",
                        {"max_bytes": max_body_bytes},
                    ),
                    options,
                )
            chunks.append(chunk)
        try:
            job = runner.submit(b"".join(chunks), options)
        except ServiceError as err:
            return fail(err, options)
        return JSONResponse(job, status_code=202, headers={"Cache-Control": "no-store"})

    @app.get("/v1/jobs/{job_id}")
    async def get_job(job_id: str):
        job = runner.public(job_id)
        if job is None:
            return fail(
                ServiceError(
                    "job_not_found", "No such job (jobs are kept in memory only).", {}
                )
            )
        return JSONResponse(job, headers={"Cache-Control": "no-store"})

    register_session_routes(app, analyzer, registry, cfg, _query_options, fail)
    return app


def _torch_threads() -> int | None:
    """Effective torch thread count, reported only if torch is already imported (never triggers the import)."""
    import sys

    torch = sys.modules.get("torch")
    return int(torch.get_num_threads()) if torch is not None else None


def prepare(app, analyzer: MrcdAnalyzer, preload: tuple[AnalysisOptions, ...], warmup: bool) -> None:
    """Fixed start-up order (the process state decides the torch thread count, DESIGN_B_INCREMENTAL 1.A.1):
    1. preload every enabled profile, 2. warm up on the engine worker (imports Silero). These are the calls the first
    request would make; no model parameter changes."""
    for options in preload:
        analyzer.preload(options)
    if warmup:
        app.state.runner.call(lambda: warm_up(analyzer, preload))
        app.state.warmed_up = True


def serve(
    analyzer: MrcdAnalyzer,
    host: str,
    port: int,
    cors_origins: tuple[str, ...] = (),
    allow_non_loopback: bool = False,
    enable_docs: bool = False,
    incremental: IncrementalConfig | None = None,
    preload: tuple[AnalysisOptions, ...] = (),
    warmup: bool = False,
) -> None:
    if host not in LOOPBACK and not allow_non_loopback:
        raise SystemExit(
            f"refusing to bind {host!r}: this development server has no authentication. "
            "Use 127.0.0.1, or pass --allow-non-loopback if you accept that."
        )
    import uvicorn

    app = create_app(
        analyzer, cors_origins=cors_origins, enable_docs=enable_docs, incremental=incremental
    )
    prepare(app, analyzer, preload, warmup)
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="warning",
    )
