"""Installation surface of 0.4.0: packaged checkpoint (sha256), console entry point, first-run Whisper download,
product page routes, sample WAV and the GitHub Pages copy."""

from __future__ import annotations

import io
import json
import subprocess
import sys
import tomllib
import wave
from pathlib import Path

import pytest
from starlette.testclient import TestClient

import mrcd
import mrcd_launcher
from mrcd import weights
from mrcd.service import MrcdAnalyzer
from mrcd.service import web
from mrcd.service.analyzer import DEFAULT_CHECKPOINT
from mrcd.service.server import create_app
from mrcd.service.whisper_cache import WhisperModelMissing, ensure_whisper_model
from tests.test_service_analyzer import FakeEngine

REPO = Path(__file__).resolve().parents[1]
PYPROJECT = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))


# ----------------------------------------------------------------------------- checkpoint
# The trained checkpoint is NOT distributed in the public repository (models/README.md): the default location is the
# packaged copy when a private build ships one, else models/ckpt_audio_text.pt in a clone; absent here by design.
def test_default_checkpoint_matches_its_pinned_sha256_when_present():
    if not weights.DEFAULT_CHECKPOINT.is_file():
        pytest.skip("checkpoint not distributed (available on request, see models/README.md)")
    assert weights.sha256_of(weights.DEFAULT_CHECKPOINT) == weights.CHECKPOINT_SHA256
    assert weights.verify_packaged_checkpoint() == weights.DEFAULT_CHECKPOINT


def test_default_falls_back_to_the_clone_location_when_nothing_is_packaged():
    assert weights.DEFAULT_CHECKPOINT in (weights.PACKAGED_CHECKPOINT, weights.REPO_CHECKPOINT)
    if not weights.PACKAGED_CHECKPOINT.is_file():
        assert weights.DEFAULT_CHECKPOINT == weights.REPO_CHECKPOINT


def test_repository_copy_is_the_same_file_as_the_packaged_one():
    if not weights.REPO_CHECKPOINT.exists():
        pytest.skip("installed package without a clone")
    assert weights.sha256_of(weights.REPO_CHECKPOINT) == weights.CHECKPOINT_SHA256


def test_resolution_order_explicit_then_env_then_packaged(tmp_path):
    custom = tmp_path / "mine.pt"
    assert weights.resolve_checkpoint(custom, environ={}) == custom
    assert weights.resolve_checkpoint("models/ckpt_audio_text.pt", environ={}) == Path(
        "models/ckpt_audio_text.pt"
    )
    assert (
        weights.resolve_checkpoint(None, environ={weights.ENV_VAR: str(custom)})
        == custom
    )
    assert DEFAULT_CHECKPOINT == weights.DEFAULT_CHECKPOINT
    if weights.DEFAULT_CHECKPOINT.is_file():
        assert weights.resolve_checkpoint(None, environ={}) == weights.DEFAULT_CHECKPOINT
    else:  # not distributed: refused with a pointer to models/README.md, never a silent fallback
        with pytest.raises(weights.CheckpointError, match="not distributed"):
            weights.resolve_checkpoint(None, environ={})


def test_a_modified_default_checkpoint_is_refused_before_unpickling(tmp_path):
    bad = tmp_path / weights.CHECKPOINT_NAME
    bad.write_bytes(b"not the pinned checkpoint")
    with pytest.raises(weights.CheckpointError, match="sha256"):
        weights.verify_packaged_checkpoint(bad)
    with pytest.raises(weights.CheckpointError, match="missing"):
        weights.verify_packaged_checkpoint(tmp_path / "absent.pt")


def test_pyproject_ships_page_and_sample_as_package_data():
    data = PYPROJECT["tool"]["setuptools"]["package-data"]
    assert {"index.html", "sample_es.wav"} <= set(data["mrcd.service.web"])


# ----------------------------------------------------------------------------- version, extras, entry point
def test_version_is_0_4_0_everywhere():
    assert PYPROJECT["project"]["version"] == mrcd.__version__ == "0.4.0"


def test_service_extra_and_console_script():
    extras = PYPROJECT["project"]["optional-dependencies"]
    names = {d.split(">")[0].split("=")[0] for d in extras["service"]}
    assert names == {"fastapi", "uvicorn", "scikit-learn"}
    assert PYPROJECT["project"]["scripts"]["mrcd"] == "mrcd_launcher:main"


def test_launcher_pins_omp_only_when_missing_and_respects_opt_out():
    env: dict = {}
    assert mrcd_launcher.pin_omp(env) == "set" and env["OMP_NUM_THREADS"] == "4"
    env = {"OMP_NUM_THREADS": "8"}
    assert mrcd_launcher.pin_omp(env) == "kept" and env["OMP_NUM_THREADS"] == "8"
    env = {"MRCD_NO_OMP_PIN": "1"}
    assert mrcd_launcher.pin_omp(env) == "opted_out" and "OMP_NUM_THREADS" not in env


def test_launcher_pinned_value_is_the_reproducibility_value():
    from mrcd.reproducibility import DEFAULT_ASR_OMP_THREADS

    assert mrcd_launcher.PINNED_OMP_THREADS == DEFAULT_ASR_OMP_THREADS


def _clean_env() -> dict:
    import os

    env = {
        k: v
        for k, v in os.environ.items()
        if k not in ("OMP_NUM_THREADS", "MRCD_NO_OMP_PIN")
    }
    env["PYTHONPATH"] = str(REPO)
    return env


def test_launcher_sets_omp_before_numpy_so_the_pin_is_effective():
    code = (
        "import sys, json, mrcd_launcher\n"
        "assert 'numpy' not in sys.modules, 'the launcher must not import numpy'\n"
        "mrcd_launcher.pin_omp()\n"
        "import mrcd\n"
        "from mrcd.reproducibility import omp_status\n"
        "print(json.dumps(omp_status()))\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO,
        env=_clean_env(),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert out.returncode == 0, out.stderr
    status = json.loads(out.stdout.strip().splitlines()[-1])
    assert (
        status["omp_num_threads_at_mrcd_import"] == "4" and status["effective"] is True
    )


def test_console_entry_point_runs_the_service_cli(tmp_path):
    code = "import sys, mrcd_launcher; sys.exit(mrcd_launcher.main(sys.argv[1:]))"
    out = subprocess.run(
        [sys.executable, "-c", code, "sample", str(tmp_path / "s.wav")],
        cwd=REPO,
        env=_clean_env(),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert out.returncode == 0, out.stderr
    assert (tmp_path / "s.wav").read_bytes() == web.SAMPLE_WAV.read_bytes()
    out = subprocess.run(
        [sys.executable, "-c", code, "--help"],
        cwd=REPO,
        env=_clean_env(),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert out.returncode == 0 and all(
        c in out.stdout for c in ("serve", "analyze", "download", "sample")
    )


# ----------------------------------------------------------------------------- first-run Whisper download
def test_cached_whisper_is_used_silently():
    calls, err = [], io.StringIO()

    def fake(size, local_only):
        calls.append(local_only)
        return "/cache/small"

    assert ensure_whisper_model(
        "small", stream=err, environ={}, downloader=fake
    ) == Path("/cache/small")
    assert calls == [True] and err.getvalue() == ""


def test_missing_whisper_is_downloaded_once_with_a_size_message():
    calls, err = [], io.StringIO()

    def fake(size, local_only):
        calls.append(local_only)
        if local_only:
            raise FileNotFoundError("not cached")
        return "/cache/small"

    ensure_whisper_model("small", stream=err, environ={}, downloader=fake)
    assert calls == [True, False]
    text = err.getvalue()
    assert (
        "~486 MB" in text and "downloading Whisper 'small'" in text and "ready" in text
    )


def test_offline_and_missing_is_a_clear_error_not_a_download():
    def fake(size, local_only):
        if local_only:
            raise FileNotFoundError("not cached")
        raise AssertionError("must not download when HF_HUB_OFFLINE is set")

    with pytest.raises(WhisperModelMissing, match="mrcd download"):
        ensure_whisper_model(
            "small",
            stream=io.StringIO(),
            environ={"HF_HUB_OFFLINE": "1"},
            downloader=fake,
        )


# ----------------------------------------------------------------------------- page and sample
def _client(**kw):
    return TestClient(
        create_app(MrcdAnalyzer("u.pt", engine_factory=lambda o: FakeEngine()), **kw)
    )


def test_page_and_sample_are_served_only_when_docs_are_enabled():
    off = _client()
    assert (
        off.get("/").status_code == 404 and off.get("/sample_es.wav").status_code == 404
    )
    on = _client(enable_docs=True)
    r = on.get("/")
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/html")
    assert "connect-src 'self'" in r.headers["content-security-policy"]
    assert "frame-ancestors 'none'" in r.headers["content-security-policy"]
    assert r.content == web.INDEX_HTML.read_bytes()
    s = on.get("/sample_es.wav")
    assert s.status_code == 200 and s.headers["content-type"] == "audio/wav"
    assert s.content == web.SAMPLE_WAV.read_bytes()


def test_page_loads_nothing_from_third_parties():
    import re

    html = web.INDEX_HTML.read_text(encoding="utf-8")
    assert not re.search(r"<(script|link|img|iframe)[^>]+(src|href)=\"https?://", html)
    assert "@import" not in html


def test_private_network_preflight_is_answered_for_allowed_origins_only():
    c = _client(cors_origins=("https://pages.example",))
    pre = {
        "Access-Control-Request-Method": "POST",
        "Access-Control-Request-Private-Network": "true",
        "Access-Control-Request-Headers": "content-type,x-session-token",
    }
    ok = c.options("/v1/jobs", headers={"Origin": "https://pages.example", **pre})
    assert ok.status_code == 200
    assert ok.headers.get("access-control-allow-private-network") == "true"
    assert "x-session-token" in ok.headers["access-control-allow-headers"].lower()
    other = c.options("/v1/jobs", headers={"Origin": "https://evil.example", **pre})
    # refused origin: the preflight fails (no allow-origin), whatever else the CORS layer adds
    assert "access-control-allow-origin" not in other.headers
    assert (
        other.status_code == 400
        or "access-control-allow-private-network" not in other.headers
    )


def test_sample_wav_is_distributable_format_and_long_enough():
    with wave.open(str(web.SAMPLE_WAV), "rb") as w:
        assert (
            w.getnchannels(),
            w.getsampwidth(),
            w.getframerate(),
            w.getcomptype(),
        ) == (1, 2, 16000, "NONE")
        assert w.getnframes() / 16000 >= mrcd.DisfluencyEngine.MIN_AUDIO_S
    assert weights.sha256_of(web.SAMPLE_WAV) == web.SAMPLE_WAV_SHA256


def test_precomputed_result_is_a_valid_contract_document_for_the_sample():
    from mrcd.service import validate_document

    doc = json.loads(
        (REPO / "examples" / "sample_result.json").read_text(encoding="utf-8")
    )
    assert validate_document(doc) == []
    assert (
        doc["status"] == "completed"
        and doc["request"]["audio_sha256"] == web.SAMPLE_WAV_SHA256
    )


def test_pages_copy_is_in_sync_with_the_packaged_page():
    out = subprocess.run(
        [sys.executable, "scripts/build_site.py", "--check"],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert out.returncode == 0, out.stdout + out.stderr


# ----------------------------------------------------------------------------- no personal paths in what ships
#: absolute Windows paths into a user profile or the local thesis workspace (the shipped code must use relative
#: paths or environment variables)
_PERSONAL = __import__("re").compile(
    r"[A-Za-z]:[\\/]+(Users|zArnes)[\\/-]", __import__("re").I
)


def test_personal_path_pattern_matches_both_separators():
    assert _PERSONAL.search("x = r'C:" + "\\" + "Users" + "\\" + "someone'")
    assert _PERSONAL.search("cd D:/zArnes-x/repo")
    assert not _PERSONAL.search("<CORPORA_ROOT>/01_DIHANA")


def test_shipped_package_has_no_personal_paths():
    shipped = [
        p
        for d in ("mrcd", "mrcd_launcher")
        for p in (REPO / d).rglob("*")
        if p.is_file()
    ]
    shipped = [
        p
        for p in shipped
        if p.suffix in {".py", ".json", ".html", ".md", ".txt", ".toml"}
        and "__pycache__" not in p.parts
    ]
    hits = [
        f"{p.relative_to(REPO)}: {m.group(0)}"
        for p in shipped
        for m in _PERSONAL.finditer(p.read_text(encoding="utf-8", errors="replace"))
    ]
    assert hits == []


def test_asr_recovery_paths_come_from_the_environment_or_the_checkout():
    code = "import json; from mrcd.asr_recovery import common as c; print(json.dumps([str(c.REPO), str(c.CORPUS)]))"
    env = _clean_env()
    env.pop("MRCD_REPO_ROOT", None)
    env["MRCD_CORPORA_ROOT"] = "/corpora"
    out = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert out.returncode == 0, out.stderr
    repo, corpus = json.loads(out.stdout.strip().splitlines()[-1])
    assert Path(repo) == REPO
    assert Path(corpus) == Path("/corpora") / "01_DIHANA" / "extraido" / "dihana"
