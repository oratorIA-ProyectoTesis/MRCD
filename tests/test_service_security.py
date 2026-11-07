"""Service-to-service guard (action 7.2): bearer token, constant-time compare, rate limit, no secrets in logs."""

from __future__ import annotations

import hmac
import logging
import secrets

import pytest
from starlette.testclient import TestClient

from mrcd.service import MrcdAnalyzer
from mrcd.service import security as SEC
from mrcd.service.server import create_app, serve
from tests.test_service_analyzer import FakeEngine, wav


def _app(**kw):
    analyzer = MrcdAnalyzer("u.pt", engine_factory=lambda o: FakeEngine())
    return create_app(analyzer, **kw)


def _client(app, ip="10.0.0.1"):
    return TestClient(app, client=(ip, 50000))


def _tok():
    return secrets.token_urlsafe(32)


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


# ------------------------------------------------------------------------------------------------ token loading
def test_no_token_configured_means_none():
    assert SEC.load_token({}) is None and SEC.load_token({SEC.TOKEN_ENV: ""}) is None


def test_short_or_padded_token_is_refused_without_echoing_it():
    for bad in ("short", _tok() + " ", " " + _tok()):
        with pytest.raises(SEC.TokenError) as e:
            SEC.load_token({SEC.TOKEN_ENV: bad})
        assert bad.strip() not in str(e.value) or bad.strip() == ""
    good = _tok()
    assert SEC.load_token({SEC.TOKEN_ENV: good}) == good


# ------------------------------------------------------------------------------------------------ loopback dev mode
def test_loopback_dev_mode_works_without_a_token_and_without_rate_limit():
    client = _client(_app())
    for _ in range(200):
        assert client.get("/v1/schema").status_code == 200
    r = client.post(
        "/v1/jobs?decision_profile=v0",
        content=wav(),
        headers={"Content-Type": "audio/wav"},
    )
    assert r.status_code == 202


# ------------------------------------------------------------------------------------------------ bearer token
def test_token_required_when_configured():
    tok = _tok()
    client = _client(_app(auth_token=tok))
    assert client.get("/v1/schema").status_code == 401
    r = client.get("/v1/schema", headers={"Authorization": "Bearer " + _tok()})
    assert r.status_code == 401 and r.headers["www-authenticate"] == "Bearer"
    assert r.json()["error"]["code"] == "unauthorized"
    assert (
        client.get("/v1/schema", headers={"Authorization": "Basic " + tok}).status_code
        == 401
    )
    assert client.get("/v1/schema", headers={"Authorization": tok}).status_code == 401
    assert (
        client.get("/v1/schema", headers={"Authorization": "Bearer " + tok}).status_code
        == 200
    )
    assert (
        client.get("/v1/schema", headers={"Authorization": "bearer " + tok}).status_code
        == 200
    )


def test_post_without_token_is_rejected_before_the_job_is_queued():
    tok = _tok()
    app = _app(auth_token=tok)
    client = _client(app)
    assert client.post("/v1/jobs", content=wav()).status_code == 401
    assert app.state.runner._queued_jobs == 0
    ok = client.post(
        "/v1/jobs?decision_profile=v0",
        content=wav(),
        headers={"Authorization": f"Bearer {tok}"},
    )
    assert ok.status_code == 202


def test_comparison_is_constant_time(monkeypatch):
    calls = []
    real = hmac.compare_digest
    monkeypatch.setattr(
        SEC.hmac,
        "compare_digest",
        lambda a, b: calls.append((len(a), len(b))) or real(a, b),
    )
    tok = _tok()
    client = _client(_app(auth_token=tok))
    client.get(
        "/v1/schema", headers={"Authorization": "Bearer x"}
    )  # very different length
    client.get("/v1/schema", headers={"Authorization": "Bearer " + tok})
    assert calls == [(32, 32), (32, 32)]  # digests: same length whatever the input


def test_non_ascii_token_header_is_a_401_not_a_crash():
    client = _client(_app(auth_token=_tok()))
    r = client.get(
        "/v1/schema", headers={"Authorization": "Bearer ñandú".encode("latin-1")}
    )
    assert r.status_code == 401


def test_health_without_token_is_minimal_and_with_token_is_full():
    tok = _tok()
    client = _client(_app(auth_token=tok))
    assert client.get("/v1/health").json() == {"status": "ok"}
    full = client.get("/v1/health", headers={"Authorization": f"Bearer {tok}"}).json()
    assert "environment" in full and "engines_loaded" in full


def test_cors_preflight_is_not_blocked_by_the_token():
    app = _app(auth_token=_tok(), cors_origins=("http://localhost:5173",))
    r = _client(app).options(
        "/v1/jobs",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert r.status_code == 200
    denied = _client(app).get("/v1/schema", headers={"Origin": "http://localhost:5173"})
    assert (
        denied.status_code == 401
        and denied.headers.get("access-control-allow-origin") == "http://localhost:5173"
    )


# ------------------------------------------------------------------------------------------------ rate limit
def _limited(clock, **cfg):
    config = SEC.RateLimitConfig(**cfg)
    return _app(rate_limit=SEC.RateLimiter(config, clock=clock))


def test_burst_then_429_with_retry_after_then_refill():
    clock = Clock()
    client = _client(_limited(clock, rate_per_s=2.0, burst=3))
    assert [client.get("/v1/schema").status_code for _ in range(3)] == [200, 200, 200]
    r = client.get("/v1/schema")
    assert r.status_code == 429 and int(r.headers["retry-after"]) >= 1
    assert r.json()["error"]["code"] == "rate_limited"
    clock.t += 0.6  # 1.2 tokens refilled
    assert client.get("/v1/schema").status_code == 200
    assert client.get("/v1/schema").status_code == 429


def test_each_client_has_its_own_bucket():
    clock = Clock()
    app = _limited(clock, rate_per_s=1.0, burst=1)
    a, b = _client(app, "10.0.0.1"), _client(app, "10.0.0.2")
    assert a.get("/v1/schema").status_code == 200
    assert a.get("/v1/schema").status_code == 429
    assert b.get("/v1/schema").status_code == 200


def test_wrong_tokens_are_throttled_per_client():
    clock = Clock()
    tok = _tok()
    app = _app(
        auth_token=tok,
        rate_limit=SEC.RateLimiter(
            SEC.RateLimitConfig(failed_auth_burst=3), clock=clock
        ),
    )
    attacker = _client(app, "10.9.9.9")
    codes = [
        attacker.get(
            "/v1/schema", headers={"Authorization": "Bearer " + _tok()}
        ).status_code
        for _ in range(5)
    ]
    assert codes == [401, 401, 401, 429, 429]
    assert (
        _client(app, "10.0.0.1")
        .get("/v1/schema", headers={"Authorization": f"Bearer {tok}"})
        .status_code
        == 200
    )


def test_tracked_clients_are_bounded(monkeypatch):
    monkeypatch.setattr(SEC, "MAX_TRACKED_CLIENTS", 5)
    lim = SEC.RateLimiter(SEC.RateLimitConfig())
    for i in range(50):
        lim.check("req", f"10.0.0.{i}")
    assert len(lim._buckets) == 5


# ------------------------------------------------------------------------------------------------ binding policy
def test_non_loopback_bind_without_token_refuses_to_start(monkeypatch):
    monkeypatch.delenv(SEC.TOKEN_ENV, raising=False)
    analyzer = MrcdAnalyzer("u.pt", engine_factory=lambda o: FakeEngine())
    with pytest.raises(SystemExit) as e:
        serve(analyzer, "0.0.0.0", 8765, allow_non_loopback=True)
    assert SEC.TOKEN_ENV in str(e.value)


def test_non_loopback_bind_with_weak_token_refuses_to_start(monkeypatch):
    monkeypatch.setenv(SEC.TOKEN_ENV, "weak")
    analyzer = MrcdAnalyzer("u.pt", engine_factory=lambda o: FakeEngine())
    with pytest.raises(SystemExit) as e:
        serve(analyzer, "0.0.0.0", 8765, allow_non_loopback=True)
    assert "weak" not in str(e.value)


def test_non_loopback_with_token_starts_with_auth_and_rate_limit_on(monkeypatch):
    tok = _tok()
    monkeypatch.setenv(SEC.TOKEN_ENV, tok)
    captured = {}
    import uvicorn

    monkeypatch.setattr(
        uvicorn, "run", lambda app, **kw: captured.update(app=app, **kw)
    )
    analyzer = MrcdAnalyzer("u.pt", engine_factory=lambda o: FakeEngine())
    serve(analyzer, "0.0.0.0", 8765, allow_non_loopback=True)
    client = _client(captured["app"])
    assert client.get("/v1/schema").status_code == 401
    assert (
        client.get("/v1/schema", headers={"Authorization": f"Bearer {tok}"}).status_code
        == 200
    )
    assert captured["app"].state.rate_limiter is not None


def test_loopback_with_token_env_enforces_it_too(monkeypatch):
    tok = _tok()
    monkeypatch.setenv(SEC.TOKEN_ENV, tok)
    captured = {}
    import uvicorn

    monkeypatch.setattr(
        uvicorn, "run", lambda app, **kw: captured.update(app=app, **kw)
    )
    serve(
        MrcdAnalyzer("u.pt", engine_factory=lambda o: FakeEngine()), "127.0.0.1", 8765
    )
    assert _client(captured["app"]).get("/v1/schema").status_code == 401


# ------------------------------------------------------------------------------------------------ secrets
def test_tokens_never_reach_logs_or_responses(caplog):
    tok, wrong = _tok(), _tok()
    app = _app(auth_token=tok)
    client = _client(app)
    with caplog.at_level(logging.DEBUG):
        bodies = [
            client.get("/v1/schema", headers={"Authorization": "Bearer " + wrong}),
            client.get("/v1/schema", headers={"Authorization": "Bearer " + tok}),
            client.get("/v1/health", headers={"Authorization": "Bearer " + wrong}),
            client.get("/v1/health", headers={"Authorization": "Bearer " + tok}),
        ]
    text = "\n".join(r.getMessage() for r in caplog.records)
    assert tok not in text and wrong not in text
    for r in bodies[:1] + bodies[2:3]:
        assert tok not in r.text and wrong not in r.text
    assert tok not in repr(app.state.__dict__.get("rate_limiter"))


# ------------------------------------------------------------------------------------------------ reverse proxy
def test_forwarded_for_is_ignored_unless_the_peer_is_a_trusted_proxy():
    clock = Clock()
    lim = lambda: SEC.RateLimiter(SEC.RateLimitConfig(rate_per_s=1.0, burst=1), clock=clock)  # noqa: E731
    spoof = {"X-Forwarded-For": "6.6.6.6"}
    plain = _client(_app(rate_limit=lim()), "10.0.0.1")
    assert plain.get("/v1/schema", headers=spoof).status_code == 200
    assert plain.get("/v1/schema", headers={"X-Forwarded-For": "7.7.7.7"}).status_code == 429  # same peer key
    proxied = _client(_app(rate_limit=lim(), trusted_proxies=("127.0.0.1",)), "127.0.0.1")
    assert proxied.get("/v1/schema", headers={"X-Forwarded-For": "1.1.1.1"}).status_code == 200
    assert proxied.get("/v1/schema", headers={"X-Forwarded-For": "2.2.2.2"}).status_code == 200  # other client
    assert proxied.get("/v1/schema", headers={"X-Forwarded-For": "1.1.1.1"}).status_code == 429
    # a client-supplied leading entry cannot pick the key: the proxy-appended (rightmost) one wins
    assert proxied.get("/v1/schema", headers={"X-Forwarded-For": "9.9.9.9, 1.1.1.1"}).status_code == 429


def test_ipv6_clients_share_one_rate_limit_key_per_slash_64():
    assert SEC._rate_key("2001:db8:1:2:aaaa:bbbb:cccc:dddd") == "2001:db8:1:2::/64"
    assert SEC._rate_key("2001:db8:1:2::1") == SEC._rate_key("2001:db8:1:2:ffff:ffff:ffff:ffff")
    assert SEC._rate_key("2001:db8:1:3::1") != SEC._rate_key("2001:db8:1:2::1")  # other /64
    assert SEC._rate_key("::ffff:192.0.2.7") == "192.0.2.7"  # IPv4-mapped keeps the IPv4 key
    assert SEC._rate_key("10.0.0.1") == "10.0.0.1" and SEC._rate_key("unknown") == "unknown"
    assert SEC._rate_key("fe80::1%eth0") == "fe80::/64"


def test_rotating_ipv6_addresses_inside_one_64_hit_the_same_bucket():
    clock = Clock()
    lim = SEC.RateLimiter(SEC.RateLimitConfig(rate_per_s=1.0, burst=1), clock=clock)
    app = _app(rate_limit=lim)
    assert _client(app, "2001:db8:1:2::1").get("/v1/schema").status_code == 200
    assert _client(app, "2001:db8:1:2::2").get("/v1/schema").status_code == 429  # same /64
    assert _client(app, "2001:db8:1:3::1").get("/v1/schema").status_code == 200  # different /64


def test_ipv6_forwarded_for_behind_a_trusted_proxy_is_also_aggregated():
    clock = Clock()
    lim = SEC.RateLimiter(SEC.RateLimitConfig(rate_per_s=1.0, burst=1), clock=clock)
    c = _client(_app(rate_limit=lim, trusted_proxies=("127.0.0.1",)), "127.0.0.1")
    assert c.get("/v1/schema", headers={"X-Forwarded-For": "2001:db8:9::1"}).status_code == 200
    assert c.get("/v1/schema", headers={"X-Forwarded-For": "2001:db8:9::2"}).status_code == 429


def test_serve_disables_uvicorn_proxy_headers_so_the_guard_is_the_only_xff_interpreter(monkeypatch):
    captured = {}
    import uvicorn

    monkeypatch.setattr(uvicorn, "run", lambda app, **kw: captured.update(app=app, **kw))
    serve(MrcdAnalyzer("u.pt", engine_factory=lambda o: FakeEngine()), "127.0.0.1", 8765)
    assert captured["proxy_headers"] is False


def test_real_uvicorn_with_proxy_headers_off_keeps_the_socket_peer_despite_xff():
    """Behavioural check on the real server: the ASGI scope client is the socket peer, not the XFF value."""
    import socket
    import threading
    import time
    import urllib.request

    import uvicorn

    seen = {}

    async def probe(scope, receive, send):
        if scope["type"] == "http":
            seen["client"] = scope["client"][0]
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b"ok"})

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    server = uvicorn.Server(
        uvicorn.Config(probe, host="127.0.0.1", port=port, log_level="error", proxy_headers=False)
    )
    t = threading.Thread(target=server.run, daemon=True)
    t.start()
    try:
        for _ in range(100):
            if server.started:
                break
            time.sleep(0.05)
        assert server.started
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/", headers={"X-Forwarded-For": "6.6.6.6"}
        )
        urllib.request.urlopen(req, timeout=5).read()
    finally:
        server.should_exit = True
        t.join(timeout=10)
    assert seen["client"] == "127.0.0.1"


def test_every_error_code_the_middleware_emits_is_in_the_service_contract():
    import re
    from pathlib import Path

    from mrcd.service.contract import ERROR_CODES, SCHEMA

    src = Path(SEC.__file__).read_text(encoding="utf-8")
    emitted = set(re.findall(r'(?:401|429),\s*"([a-z_]+)"', src))
    assert {"unauthorized", "rate_limited"} <= emitted
    assert emitted <= set(ERROR_CODES) and emitted <= set(SCHEMA["$defs"]["error"]["properties"]["code"]["enum"])
    assert ERROR_CODES["unauthorized"] == (401, False) and ERROR_CODES["rate_limited"] == (429, True)


# ------------------------------------------------------------------------------------------------ non-http scopes
def _drive(make_guard, scope):
    import asyncio

    sent, called = [], []

    async def inner(scope, receive, send):
        called.append(scope["type"])

    guard = make_guard(inner)

    async def receive():
        return {"type": "websocket.connect"}

    async def send(msg):
        sent.append(msg)

    asyncio.run(guard(scope, receive, send))
    return sent, called


def _authed(inner):
    return SEC.ServiceGuard(inner, token=_tok(), limiter=None)


def test_websocket_scope_is_refused_when_auth_is_required():
    scope = {"type": "websocket", "path": "/ws", "headers": [], "client": ("10.0.0.1", 1)}
    sent, called = _drive(_authed, scope)
    assert called == [] and sent == [{"type": "websocket.close", "code": 1008}]


def test_unknown_scope_types_are_dropped_when_auth_is_required_but_lifespan_passes():
    sent, called = _drive(_authed, {"type": "weird", "headers": []})
    assert called == [] and sent == []
    _, called = _drive(_authed, {"type": "lifespan"})
    assert called == ["lifespan"]


def test_non_http_scopes_pass_through_when_no_token_is_configured():
    open_guard = lambda inner: SEC.ServiceGuard(inner, token=None, limiter=None)  # noqa: E731
    _, called = _drive(open_guard, {"type": "websocket", "headers": []})
    assert called == ["websocket"]
