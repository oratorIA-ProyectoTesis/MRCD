"""Minimal local development HTTP server for the MRCD service (FastAPI, already installed in the venv).

Scope: ``localhost`` development only. No authentication, no TLS, no API keys. The server refuses to bind a
non-loopback address unless explicitly told to. Raw audio is decoded in memory and dropped as soon as the job
finishes; nothing is written to disk. Job results (JSON only, no audio, transcript only if the request asked
for it) are kept in memory, bounded, and vanish when the process stops.

    POST /v1/jobs?asr_profile=p0&head=&include_text=false&include_out_of_scope=false&decision=false&text_route=   body = raw WAV bytes  -> 202 job status
    GET  /v1/jobs/{job_id}                                                         -> job status (+ result)
    GET  /v1/health, GET /v1/schema
"""

from __future__ import annotations

import queue
import threading
import uuid
from collections import OrderedDict
from datetime import datetime, timezone
from typing import Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from mrcd.service.analyzer import AnalysisOptions, MrcdAnalyzer, error_document
from mrcd.service.audio_io import ServiceError
from mrcd.service.contract import ERROR_CODES, JOB_SCHEMA_VERSION, SCHEMA

LOOPBACK = ("127.0.0.1", "localhost", "::1")
DEFAULT_MAX_BODY_BYTES = 64 * 1024 * 1024
MAX_RETAINED_JOBS = 32
MAX_QUEUED_JOBS = 8


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class JobRunner:
    """One worker thread: the engine is not thread-safe, so analyses run one at a time."""

    def __init__(self, analyzer: MrcdAnalyzer):
        self.analyzer = analyzer
        self._jobs: OrderedDict[str, dict] = OrderedDict()
        self._queue: queue.Queue[str] = queue.Queue(maxsize=MAX_QUEUED_JOBS)
        self._lock = threading.Lock()
        self._thread = threading.Thread(
            target=self._loop, name="mrcd-service-worker", daemon=True
        )
        self._thread.start()

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
            self._jobs[job_id] = job
            self._evict()
        try:
            self._queue.put_nowait(job_id)
        except queue.Full:
            with self._lock:
                self._jobs.pop(job_id, None)
            raise ServiceError(
                "queue_full",
                "Too many pending jobs; retry later.",
                {"max_queued": MAX_QUEUED_JOBS},
            )
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
            job_id = self._queue.get()
            with self._lock:
                job = self._jobs.get(job_id)
                if job is None:
                    continue
                job["status"] = "running"
                data, options = job["_data"], job["_options"]
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
            raise ServiceError("invalid_request", f"{name} must be 'true' or 'false'.", {"param": name})
        return v == "true"

    route = (params.get("text_route") or "").strip().lower()
    options = AnalysisOptions(
        asr_profile=(params.get("asr_profile") or "p0").strip(),
        head=head,
        include_text=(text == "true"),
        output_scope="all" if _flag("include_out_of_scope") else "evaluated",
        decision=_flag("decision"),
        text_route=None if route in ("", "none") else route,
    )
    MrcdAnalyzer.validate_options(options)
    return options


def create_app(
    analyzer: MrcdAnalyzer,
    *,
    max_body_bytes: int = DEFAULT_MAX_BODY_BYTES,
    cors_origins: tuple[str, ...] = (),
):

    app = FastAPI(
        title="MRCD service (local development)",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    if cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(cors_origins),
            allow_methods=["GET", "POST"],
            allow_headers=["Content-Type"],
            allow_credentials=False,
        )
    runner = JobRunner(analyzer)
    app.state.runner = runner

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
        }

    @app.get("/v1/schema")
    async def schema():
        return SCHEMA

    @app.post("/v1/jobs")
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

    return app


def serve(
    analyzer: MrcdAnalyzer,
    host: str,
    port: int,
    cors_origins: tuple[str, ...] = (),
    allow_non_loopback: bool = False,
) -> None:
    if host not in LOOPBACK and not allow_non_loopback:
        raise SystemExit(
            f"refusing to bind {host!r}: this development server has no authentication. "
            "Use 127.0.0.1, or pass --allow-non-loopback if you accept that."
        )
    import uvicorn

    uvicorn.run(
        create_app(analyzer, cors_origins=cors_origins),
        host=host,
        port=port,
        log_level="warning",
    )
