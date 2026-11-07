"""Minimal smoke test of examples/ and of the TypeScript client against a real HTTP server with a fake engine.

Node-based checks run only when ``node`` is on PATH; the ``tsc`` check runs when ``tsc`` is on PATH or ``MRCD_TSC``
points to it (no network installs from the test suite). Each skip says why.
"""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from mrcd.service import MrcdAnalyzer
from mrcd.service.server import create_app
from tests.test_service_analyzer import FakeEngine, wav

REPO = Path(__file__).resolve().parents[1]
EXAMPLES = REPO / "examples"
TS_CLIENT = REPO / "clients" / "typescript" / "mrcd.ts"


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


@pytest.fixture()
def sample(tmp_path):
    p = tmp_path / "sample.wav"
    p.write_bytes(wav())
    return p


def _run(cmd, cwd, url, timeout=120):
    env = {
        **os.environ,
        "MRCD_URL": url,
        "PYTHONPATH": str(REPO),
        "PYTHONIOENCODING": "utf-8",
    }
    return subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=timeout,
    )


def test_python_example_job_mode(base_url, sample):
    out = _run(
        [sys.executable, str(EXAMPLES / "python_client.py"), str(sample)],
        sample.parent,
        base_url,
    )
    assert out.returncode == 0, out.stdout + out.stderr
    assert "service: ok" in out.stdout and "events per minute" in out.stdout


def _node() -> str:
    node = shutil.which("node")
    if not node:
        pytest.skip("node is not on PATH")
    return node


def test_javascript_example(base_url, sample):
    out = _run(
        [_node(), str(EXAMPLES / "analyze.mjs"), str(sample), base_url],
        sample.parent,
        base_url,
    )
    assert out.returncode == 0, out.stdout + out.stderr  # prints one line per event (possibly none)


def test_typescript_example_runs_with_type_stripping(base_url, sample):
    node = _node()
    major, minor = (
        int(x)
        for x in subprocess.run([node, "--version"], capture_output=True, text=True)
        .stdout.strip()
        .lstrip("v")
        .split(".")[:2]
    )
    if (major, minor) < (22, 6):
        pytest.skip("node >= 22.6 is needed for --experimental-strip-types")
    out = _run(
        [
            node,
            "--experimental-strip-types",
            "--no-warnings",
            str(EXAMPLES / "typescript_client.ts"),
            str(sample),
        ],
        sample.parent,
        base_url,
    )
    assert out.returncode == 0, out.stdout + out.stderr
    assert (
        "incremental sessions: false" in out.stdout and "events per minute" in out.stdout
    )


def test_curl_example(base_url, sample):
    sh, curl = shutil.which("sh"), shutil.which("curl")
    if not (sh and curl):
        pytest.skip("sh and curl are needed")
    out = _run(
        [sh, str(EXAMPLES / "curl.sh"), sample.name, base_url], sample.parent, base_url
    )
    assert out.returncode == 0, out.stdout + out.stderr
    assert '"schema_version":"mrcd.analysis/v1"' in (
        sample.parent / "result.json"
    ).read_text(encoding="utf-8")


def test_typescript_client_typechecks(tmp_path):
    tsc = os.environ.get("MRCD_TSC") or shutil.which("tsc")
    if not tsc:
        pytest.skip("tsc not found (put it on PATH or set MRCD_TSC)")
    cmd = [
        tsc,
        "--noEmit",
        "--strict",
        "--target",
        "es2022",
        "--module",
        "esnext",
        "--moduleResolution",
        "bundler",
        "--lib",
        "es2022,dom",
        str(TS_CLIENT),
    ]
    out = subprocess.run(
        cmd, capture_output=True, text=True, timeout=180, shell=sys.platform == "win32"
    )
    assert out.returncode == 0, out.stdout + out.stderr
