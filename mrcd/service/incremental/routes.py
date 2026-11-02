"""HTTP surface of the incremental sessions (contract rev 1.3): /v1/sessions.

Access control: ``POST /v1/sessions`` returns a ``subscription_token`` (shown once). Every other session endpoint
(audio, close, state, events, delete) requires it in the ``X-Session-Token`` header (or ``Authorization: Bearer``).
A missing or wrong token answers 404 ``session_not_found`` (the same answer as an unknown id, so it is no oracle for
session existence). The event stream never carries text or the result document; the document is read with
``GET /v1/sessions/{id}`` once the terminal event says ``completed`` or ``failed``."""

from __future__ import annotations

import asyncio
import json
from typing import Any, Callable

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

from mrcd.service.analyzer import MrcdAnalyzer
from mrcd.service.audio_io import ServiceError
from mrcd.service.contract import SESSION_SCHEMA_VERSION
from mrcd.service.incremental.session import (
    SAMPLE_RATE,
    IncrementalConfig,
    SessionRegistry,
)

NO_STORE = {"Cache-Control": "no-store"}
HEARTBEAT_S = 15.0
TERMINAL_EVENTS = ("completed", "failed", "expired", "cancelled")


def _presented_token(request: Request) -> str | None:
    token = request.headers.get("x-session-token")
    if token:
        return token.strip()
    auth = request.headers.get("authorization", "")
    return auth[7:].strip() if auth.lower().startswith("bearer ") else None


def _sse(event: dict) -> str:
    return f"id: {event['id']}\nevent: {event['type']}\ndata: {json.dumps(event['data'], ensure_ascii=False)}\n\n"


async def _stream(session, last_id: int | None, poll_s: float = 0.05):
    """Server-side wait loop (the browser never polls). A fresh subscriber first gets the CURRENT state, so there is
    no race between finishing and subscribing; ``Last-Event-ID`` resumes without losing or repeating events."""
    snapshot = None
    with session.lock:
        if last_id is None:
            terminal = [e for e in session.events if e["type"] in TERMINAL_EVENTS]
            if terminal and not session.alive:
                yield _sse(terminal[-1])
                return
            snapshot = {
                "id": len(session.events),
                "type": "state",
                "data": session._counts(),
            }
            sent = len(session.events)
        else:
            sent = max(0, last_id)
    if snapshot:
        yield _sse(snapshot)
    idle = 0.0
    while True:
        with session.lock:
            new = session.events[sent:]
            sent += len(new)
            done = not session.alive
        for e in new:
            idle = 0.0
            yield _sse(e)
        if done and not new:
            return
        await asyncio.sleep(poll_s)
        idle += poll_s
        if idle >= HEARTBEAT_S:
            idle = 0.0
            yield ": keep-alive\n\n"


def register(
    app: FastAPI,
    analyzer: MrcdAnalyzer,
    registry: SessionRegistry | None,
    cfg: IncrementalConfig,
    query_options: Callable[[Any], Any],
    fail: Callable[..., Any],
) -> None:
    def guard() -> SessionRegistry:
        if registry is None:
            raise ServiceError(
                "incremental_disabled",
                "Incremental sessions are disabled on this service.",
                {},
            )
        return registry

    def authorized(session_id: str, request: Request):
        session = guard().get(session_id)
        if not session.token_ok(_presented_token(request)):
            raise ServiceError(
                "session_not_found",
                "No such session (sessions are kept in memory only).",
                {},
            )
        return session

    @app.post(
        "/v1/sessions", status_code=201, summary="Open an incremental analysis session"
    )
    async def open_session(request: Request):
        try:
            reg = guard()
            options = query_options(request.query_params)
            session = reg.open(options)
        except ServiceError as err:
            return fail(err)
        body = {
            "schema_version": SESSION_SCHEMA_VERSION,
            "session_id": session.id,
            "subscription_token": session.token,
            "status": "open",
            "plan": {
                "chunk_s": 8,
                "overlap_s": 2,
                "sample_rate_hz": SAMPLE_RATE,
                "sample_format": "pcm_s16le",
                "channels": 1,
            },
            "limits": {
                "max_duration_s": cfg.max_session_seconds,
                "idle_timeout_s": cfg.idle_timeout_s,
                "max_batch_bytes": cfg.max_batch_bytes,
            },
        }
        return JSONResponse(body, status_code=201, headers=NO_STORE)

    @app.put(
        "/v1/sessions/{session_id}/audio/{seq}",
        summary="Send one batch of raw PCM16 LE audio (idempotent by seq)",
        openapi_extra={
            "requestBody": {
                "required": True,
                "content": {
                    "application/octet-stream": {
                        "schema": {"type": "string", "format": "binary"}
                    }
                },
            },
            "parameters": [
                {
                    "name": "X-Offset-Samples",
                    "in": "header",
                    "required": True,
                    "schema": {"type": "integer"},
                },
                {
                    "name": "X-Content-SHA256",
                    "in": "header",
                    "required": False,
                    "schema": {"type": "string"},
                },
            ],
        },
    )
    async def put_audio(session_id: str, seq: int, request: Request):
        try:
            session = authorized(session_id, request)
            raw_offset = request.headers.get("x-offset-samples", "")
            if not raw_offset.isdigit():
                raise ServiceError(
                    "invalid_request",
                    "X-Offset-Samples is required.",
                    {"header": "X-Offset-Samples"},
                )
            declared = request.headers.get("content-length")
            if declared and declared.isdigit() and int(declared) > cfg.max_batch_bytes:
                raise ServiceError(
                    "batch_too_large",
                    "The batch is too large.",
                    {"max_bytes": cfg.max_batch_bytes},
                )
            body = bytearray()  # zeroed below: the only body copy this code controls
            try:
                async for chunk in request.stream():
                    if len(body) + len(chunk) > cfg.max_batch_bytes:
                        raise ServiceError(
                            "batch_too_large",
                            "The batch is too large.",
                            {"max_bytes": cfg.max_batch_bytes},
                        )
                    body += chunk
                reply = session.put(
                    seq,
                    int(raw_offset),
                    body,
                    request.headers.get("x-content-sha256"),
                )
            finally:
                body[:] = bytes(len(body))
        except ServiceError as err:
            return fail(err)
        return JSONResponse(reply, headers=NO_STORE)

    @app.post(
        "/v1/sessions/{session_id}/close",
        summary="Close the session and start the full-audio analysis",
    )
    async def close_session(session_id: str, request: Request):
        try:
            session = authorized(session_id, request)
            try:
                payload = await request.json()
                total, sha, last_seq = (
                    int(payload["total_samples"]),
                    str(payload["wav_sha256"]),
                    int(payload["last_seq"]),
                )
            except (ValueError, KeyError, TypeError):
                raise ServiceError(
                    "invalid_request",
                    "Body must be JSON with total_samples, wav_sha256 and last_seq.",
                    {},
                ) from None
            status, body = session.close(total, sha, last_seq)
        except ServiceError as err:
            return fail(err)
        return JSONResponse(body, status_code=status, headers=NO_STORE)

    @app.get(
        "/v1/sessions/{session_id}",
        summary="Session state (and the result once finished)",
    )
    async def get_session(session_id: str, request: Request):
        try:
            return JSONResponse(
                authorized(session_id, request).state(), headers=NO_STORE
            )
        except ServiceError as err:
            return fail(err)

    @app.get(
        "/v1/sessions/{session_id}/events",
        summary="Server-sent progress events (counts only: no text, no result document)",
    )
    async def session_events(session_id: str, request: Request):
        try:
            session = authorized(session_id, request)
        except ServiceError as err:
            return fail(err)
        raw = request.headers.get("last-event-id", "")
        last_id = int(raw) if raw.isdigit() else None
        return StreamingResponse(
            _stream(session, last_id),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
        )

    @app.delete(
        "/v1/sessions/{session_id}",
        status_code=204,
        summary="Discard the session audio and cache (idempotent; acts only with the session token)",
    )
    async def delete_session(session_id: str, request: Request):
        try:
            # Idempotent and with no existence oracle: without a valid token (unknown id, wrong or missing token) the
            # answer is the same 204 and NOTHING is done; only the token holder actually cancels.
            reg = guard()
            try:
                authorized(session_id, request)
            except ServiceError:
                return Response(status_code=204, headers=NO_STORE)
            reg.cancel(session_id)
        except ServiceError as err:
            return fail(err)
        return Response(status_code=204, headers=NO_STORE)

