"""Local dev HTTP server: job lifecycle, structured HTTP errors, loopback-only, CORS opt-in, no audio retained."""

from __future__ import annotations

import time

import pytest
from starlette.testclient import TestClient

from mrcd.service import MrcdAnalyzer, validate_document
from mrcd.service.cli import main as cli_main
from mrcd.service.server import create_app, serve
from tests.test_service_analyzer import FakeEngine, wav


def make_client(**kw):
    analyzer = MrcdAnalyzer("u.pt", engine_factory=lambda o: FakeEngine())
    app = create_app(analyzer, **kw)
    return TestClient(app), app


def wait_done(client, job_id, timeout=20):
    end = time.time() + timeout
    while time.time() < end:
        job = client.get(f"/v1/jobs/{job_id}").json()
        assert validate_document(job) == []
        if job["status"] in ("completed", "failed"):
            return job
        time.sleep(0.02)
    raise AssertionError("job did not finish")


def test_post_then_poll_returns_a_valid_completed_job_and_drops_the_audio():
    client, app = make_client()
    r = client.post("/v1/jobs", content=wav(), headers={"Content-Type": "audio/wav"})
    assert r.status_code == 202
    queued = r.json()
    assert validate_document(queued) == [] and queued["status"] in (
        "queued",
        "running",
        "completed",
    )
    assert queued["result"] is None or queued["status"] == "completed"
    job = wait_done(client, queued["job_id"])
    assert (
        job["status"] == "completed"
        and job["result"]["events"][0]["class"] == "filler_word"
    )
    assert (
        job["result"]["request"]["include_text"] is False
        and job["result"]["transcript"] is None
    )
    internal = app.state.runner._jobs[queued["job_id"]]
    assert internal["_data"] is None  # raw audio is not retained once the job ended


def test_a_bad_body_fails_inside_the_job_with_a_structured_error():
    client, _ = make_client()
    queued = client.post("/v1/jobs", content=b"nope").json()
    job = wait_done(client, queued["job_id"])
    assert (
        job["status"] == "failed"
        and job["result"]["error"]["code"] == "invalid_audio_file"
    )


def test_query_parameters_select_experimental_options_and_text():
    client, app = make_client()
    queued = client.post(
        "/v1/jobs?asr_profile=p1_disfluent_prompt&include_text=true", content=wav()
    ).json()
    job = wait_done(client, queued["job_id"])
    assert job["result"]["request"]["asr_profile"] == "p1_disfluent_prompt"
    assert job["result"]["transcript"] == "eh"


@pytest.mark.parametrize("query", ["asr_profile=p9", "head=zz", "include_text=maybe"])
def test_invalid_query_is_a_400_with_the_error_document(query):
    client, _ = make_client()
    r = client.post(f"/v1/jobs?{query}", content=wav())
    assert r.status_code == 400
    body = r.json()
    assert validate_document(body) == [] and body["error"]["code"] == "invalid_request"


def test_unknown_job_is_404():
    client, _ = make_client()
    r = client.get("/v1/jobs/" + "0" * 32)
    assert r.status_code == 404 and r.json()["error"]["code"] == "job_not_found"
    assert validate_document(r.json()) == []


def test_oversized_body_is_413_and_never_reaches_the_engine():
    client, app = make_client(max_body_bytes=1000)
    r = client.post("/v1/jobs", content=wav())
    assert r.status_code == 413 and r.json()["error"]["code"] == "payload_too_large"
    assert app.state.runner._jobs == {}


def test_health_and_schema_endpoints():
    client, _ = make_client()
    assert client.get("/v1/health").json()["status"] == "ok"
    assert client.get("/v1/schema").json()["$id"] == "urn:mrcd:service:analysis:v1"


def test_cors_is_off_by_default_and_opt_in_per_origin():
    client, _ = make_client()
    r = client.get("/v1/health", headers={"Origin": "http://localhost:5173"})
    assert "access-control-allow-origin" not in r.headers
    client, _ = make_client(cors_origins=("http://localhost:5173",))
    r = client.get("/v1/health", headers={"Origin": "http://localhost:5173"})
    assert r.headers["access-control-allow-origin"] == "http://localhost:5173"
    r = client.get("/v1/health", headers={"Origin": "http://evil.example"})
    assert "access-control-allow-origin" not in r.headers


def test_server_refuses_non_loopback_hosts_unless_forced():
    with pytest.raises(SystemExit):
        serve(MrcdAnalyzer("u.pt", engine_factory=lambda o: FakeEngine()), "0.0.0.0", 1)


def test_routes_expose_no_docs_or_openapi_and_no_keys_are_required():
    client, _ = make_client()
    for path in ("/docs", "/redoc", "/openapi.json"):
        assert client.get(path).status_code == 404
    assert client.get("/v1/health").status_code == 200  # no Authorization header needed


def test_cli_schema_and_validate_roundtrip(tmp_path, capsys):
    assert cli_main(["schema"]) == 0
    assert "mrcd.analysis/v1" in capsys.readouterr().out
    good = tmp_path / "ok.json"
    import json

    from mrcd.service import error_document

    good.write_text(json.dumps(error_document("audio_not_mono", "m")), encoding="utf-8")
    assert cli_main(["validate", str(good)]) == 0
    bad = tmp_path / "bad.json"
    bad.write_text("{}", encoding="utf-8")
    assert cli_main(["validate", str(bad)]) == 1


def test_cors_preflight_for_the_browser_client_post_with_audio_content_type():
    client, _ = make_client(cors_origins=("http://localhost:5173",))
    r = client.options(
        "/v1/jobs",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )
    assert r.status_code == 200
    assert r.headers["access-control-allow-origin"] == "http://localhost:5173"
    assert "POST" in r.headers["access-control-allow-methods"]
    assert "access-control-allow-credentials" not in r.headers  # the client sends no credentials
