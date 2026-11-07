"""Standard-library client against a real HTTP server (uvicorn in a thread, fake engine)."""

from __future__ import annotations

import socket
import threading
import time

import pytest

from mrcd.client import MrcdClient, MrcdError, pcm16_wav
from mrcd.service import MrcdAnalyzer, validate_document
from mrcd.service.incremental.session import canonical_wav
from mrcd.service.server import create_app
from tests.test_service_analyzer import FakeEngine, wav


@pytest.fixture(scope="module")
def base_url():
    import uvicorn

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    app = create_app(MrcdAnalyzer("u.pt", engine_factory=lambda o: FakeEngine()))
    server = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
    )
    t = threading.Thread(target=server.run, daemon=True)
    t.start()
    end = time.time() + 20
    while not server.started and time.time() < end:
        time.sleep(0.05)
    yield f"http://127.0.0.1:{port}"
    server.should_exit = True
    t.join(timeout=10)


def test_health_schema_and_blocking_analyze(base_url):
    c = MrcdClient(base_url)
    assert c.health()["status"] == "ok"
    assert c.schema()["$id"] == "urn:mrcd:service:analysis:v1"
    doc = c.analyze(wav(), poll_s=0.02, decision_profile="v0", include_text=True)
    assert validate_document(doc) == [] and doc["status"] == "completed"
    assert doc["request"]["include_text"] is True


def test_structured_http_errors(base_url):
    c = MrcdClient(base_url)
    with pytest.raises(MrcdError) as err:
        c.submit(wav(), asr_profile="nope")
    assert err.value.status == 400 and err.value.code == "invalid_request"
    with pytest.raises(MrcdError) as err:
        c.open_session()
    assert err.value.code == "incremental_disabled"
    with pytest.raises(TypeError):
        c.submit(wav(), unknown_option=True)


def test_wav_helper_matches_the_session_canonical_wav():
    pcm = bytes(range(256)) * 10
    assert pcm16_wav(pcm) == canonical_wav(pcm)
