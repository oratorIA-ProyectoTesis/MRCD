"""Service-to-service guard for the MRCD HTTP service (TODO_PLAN action 7.2): bearer token + per-client rate limit.

What this is: an optional shared-secret bearer token (``MRCD_SERVICE_TOKEN``), compared in constant time, that is
REQUIRED automatically when the server binds a non-loopback address, and a per-client token bucket that answers
``429`` with ``Retry-After``. What this is NOT: TLS (terminate it at a reverse proxy, see ``deploy/caddy``), user
authentication, or protection against a hostile network between the proxy and this process.

Never logs or echoes a token, audio or text. Errors raised here precede the ``mrcd.analysis/v1`` envelope, so they
use a small ``{"error": {"code", "message"}}`` body instead of the contract error document.
"""

from __future__ import annotations

import hashlib
import hmac
import ipaddress
import json
import math
import os
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Callable, Mapping

TOKEN_ENV = "MRCD_SERVICE_TOKEN"
MIN_TOKEN_CHARS = 32
MAX_TRACKED_CLIENTS = 4096


class TokenError(ValueError):
    """The configured service token is unusable (the message never contains the token)."""


def load_token(environ: Mapping[str, str] | None = None) -> str | None:
    env = os.environ if environ is None else environ
    raw = env.get(TOKEN_ENV)
    if raw is None or raw == "":
        return None
    if len(raw) < MIN_TOKEN_CHARS or raw != raw.strip():
        raise TokenError(
            f"{TOKEN_ENV} must be at least {MIN_TOKEN_CHARS} characters with no surrounding whitespace "
            '(generate one with: python -c "import secrets; print(secrets.token_urlsafe(32))")'
        )
    return raw


def tokens_match(presented: str, expected_digest: bytes) -> bool:
    """Constant-time comparison of SHA-256 digests (equal length, so neither content nor length leaks)."""
    return hmac.compare_digest(
        hashlib.sha256(presented.encode("utf-8")).digest(), expected_digest
    )


@dataclass(frozen=True)
class RateLimitConfig:
    rate_per_s: float = 10.0  # sustained requests per second per client
    burst: int = 30  # bucket capacity
    failed_auth_burst: int = (
        10  # wrong-token attempts allowed before 429 (refills at 1 per 6 s)
    )

    def __post_init__(self):
        if self.rate_per_s <= 0 or self.burst < 1 or self.failed_auth_burst < 1:
            raise ValueError("rate_per_s must be > 0 and burst sizes >= 1")


class TokenBucket:
    __slots__ = ("capacity", "rate", "tokens", "stamp")

    def __init__(self, capacity: float, rate: float, now: float):
        self.capacity, self.rate, self.tokens, self.stamp = (
            capacity,
            rate,
            capacity,
            now,
        )

    def take(self, now: float) -> float:
        """0.0 if a token was taken, else the seconds to wait until one is available."""
        self.tokens = min(self.capacity, self.tokens + (now - self.stamp) * self.rate)
        self.stamp = now
        if self.tokens >= 1.0:
            self.tokens -= 1.0
            return 0.0
        return (1.0 - self.tokens) / self.rate


@dataclass
class RateLimiter:
    config: RateLimitConfig
    clock: Callable[[], float] = time.monotonic
    _buckets: "OrderedDict[tuple, TokenBucket]" = field(default_factory=OrderedDict)

    def check(self, kind: str, client: str) -> float:
        """Seconds to wait (0.0 = allowed). Memory is bounded: least recently used clients are forgotten."""
        key = (kind, client)
        now = self.clock()
        b = self._buckets.pop(key, None)
        if b is None:
            cfg = self.config
            b = (
                TokenBucket(cfg.failed_auth_burst, 1 / 6.0, now)
                if kind == "auth"
                else TokenBucket(cfg.burst, cfg.rate_per_s, now)
            )
        self._buckets[key] = b
        while len(self._buckets) > MAX_TRACKED_CLIENTS:
            self._buckets.popitem(last=False)
        return b.take(now)


def _json(
    status: int, code: str, message: str, headers: list[tuple[bytes, bytes]]
) -> tuple[dict, bytes]:
    body = json.dumps({"error": {"code": code, "message": message}}).encode()
    start = {
        "type": "http.response.start",
        "status": status,
        "headers": [
            (b"content-type", b"application/json"),
            (b"content-length", str(len(body)).encode()),
            (b"cache-control", b"no-store"),
            *headers,
        ],
    }
    return start, body


def _rate_key(addr: str) -> str:
    """IPv6 -> its /64 network (IPv4-mapped IPv6 -> the embedded IPv4); anything else is returned unchanged."""
    try:
        ip = ipaddress.ip_address(addr.split("%", 1)[0])  # drop a zone id such as fe80::1%eth0
    except ValueError:
        return addr
    if isinstance(ip, ipaddress.IPv6Address):
        if ip.ipv4_mapped is not None:
            return str(ip.ipv4_mapped)
        return str(ipaddress.ip_network((ip, 64), strict=False))
    return str(ip)


class ServiceGuard:
    """Pure ASGI middleware (streaming responses such as SSE are untouched). CORS preflights pass through."""

    def __init__(
        self,
        app,
        *,
        token: str | None,
        limiter: RateLimiter | None,
        trusted_proxies: frozenset[str] = frozenset(),
    ):
        self.app = app
        self.trusted_proxies = trusted_proxies
        self._digest = hashlib.sha256(token.encode("utf-8")).digest() if token else None
        self.limiter = limiter

    @staticmethod
    def _bearer(scope) -> str | None:
        for k, v in scope["headers"]:
            if k == b"authorization":
                text = v.decode("latin-1")
                if text[:7].lower() == "bearer ":
                    return text[7:].strip()
                return None
        return None

    def _client(self, scope) -> str:
        """Rate-limit key. Peer address; behind a TRUSTED proxy only, the rightmost X-Forwarded-For entry (the one
        the proxy itself appended, so a client-supplied value cannot be chosen). Without a trusted proxy the header is
        ignored. IPv6 addresses are aggregated to their /64 (a single host or subscriber routinely controls the whole
        prefix, so per-address keys would let it dodge the limit); IPv4 is unchanged."""
        peer = (scope.get("client") or ("unknown", 0))[0]
        if peer in self.trusted_proxies:
            for k, v in scope["headers"]:
                if k == b"x-forwarded-for":
                    last = v.decode("latin-1").split(",")[-1].strip()
                    if last:
                        return _rate_key(last[:64])
        return _rate_key(peer)

    async def _reply(self, send, status, code, message, extra=()):
        start, body = _json(status, code, message, list(extra))
        await send(start)
        await send({"type": "http.response.body", "body": body})

    async def __call__(self, scope, receive, send):
        if scope["type"] == "lifespan":
            return await self.app(scope, receive, send)
        if scope["type"] != "http":
            if self._digest is None:
                return await self.app(scope, receive, send)
            # Auth is required and this guard only authenticates HTTP: refuse every other protocol (websocket, ...)
            # instead of letting it bypass the token.
            if scope["type"] == "websocket":
                await send({"type": "websocket.close", "code": 1008})
            return None
        if scope["method"] == "OPTIONS":
            return await self.app(scope, receive, send)
        client = self._client(scope)
        if self._digest is not None:
            presented = self._bearer(scope)
            if presented is None or not tokens_match(presented, self._digest):
                if (
                    scope["path"] == "/v1/health"
                ):  # liveness stays open but minimal (no environment details)
                    wait = self.limiter.check("req", client) if self.limiter else 0.0
                    if wait > 0:
                        return await self._reply(
                            send,
                            429,
                            "rate_limited",
                            "Rate limit exceeded; retry later.",
                            [(b"retry-after", str(max(1, math.ceil(wait))).encode())],
                        )
                    return await self._minimal_health(send)
                wait = self.limiter.check("auth", client) if self.limiter else 0.0
                if wait > 0:
                    return await self._reply(
                        send,
                        429,
                        "rate_limited",
                        "Too many failed attempts.",
                        [(b"retry-after", str(math.ceil(wait)).encode())],
                    )
                return await self._reply(
                    send,
                    401,
                    "unauthorized",
                    "A valid bearer token is required.",
                    [(b"www-authenticate", b"Bearer")],
                )
        if self.limiter is not None:
            wait = self.limiter.check("req", client)
            if wait > 0:
                return await self._reply(
                    send,
                    429,
                    "rate_limited",
                    "Rate limit exceeded; retry later.",
                    [(b"retry-after", str(max(1, math.ceil(wait))).encode())],
                )
        return await self.app(scope, receive, send)

    async def _minimal_health(self, send):
        body = b'{"status":"ok"}'
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode()),
                    (b"cache-control", b"no-store"),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})
