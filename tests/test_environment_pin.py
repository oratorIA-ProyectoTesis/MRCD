"""Environment pinning (action 3.4): the check is truthful and the real process must match the pinned state."""

import logging
import os
import subprocess
import sys
from pathlib import Path

import pytest
from starlette.testclient import TestClient

from mrcd.service import MrcdAnalyzer
from mrcd.service import environment_pin as EP
from mrcd.service.analyzer import AnalysisOptions
from mrcd.service.server import create_app, prepare
from tests.test_service_analyzer import FakeEngine

ROOT = Path(__file__).resolve().parents[1]
GOOD = dict(
    environ={"OMP_NUM_THREADS": "4"},
    omp_status={"set_before_numpy": True, "effective": True},
    device="cpu",
    torch_threads=1,
    load_events=["preload", "warmup"],
    engines_now=["p0"],
    engines_at_warmup=["p0"],
)


def ev(**over):
    return EP.evaluate(**{**GOOD, **over})


def test_all_conditions_met_is_pinned_without_warnings():
    r = ev()
    assert r["environment_is_pinned"] is True and r["warnings"] == []


@pytest.mark.parametrize(
    "over,failed",
    [
        ({"environ": {"OMP_NUM_THREADS": "8"}}, "omp_num_threads"),
        ({"environ": {}}, "omp_num_threads"),
        (
            {"omp_status": {"set_before_numpy": False, "effective": False}},
            "omp_set_before_numpy",
        ),
        (
            {"omp_status": {"set_before_numpy": None, "effective": None}},
            "omp_set_before_numpy",
        ),
        ({"device": None}, "device"),
        ({"device": "cuda"}, "device"),
        ({"torch_threads": 4}, "torch_threads"),
        ({"torch_threads": None}, "torch_threads"),
        ({"load_events": ["preload"]}, "torch_threads"),
        ({"load_events": ["warmup", "preload"]}, "load_order"),
        (
            {"engines_now": ["p0", "p1_disfluent_prompt"]},
            "no_engine_loaded_after_warmup",
        ),
    ],
)
def test_any_difference_is_not_pinned_with_a_named_warning(over, failed):
    r = ev(**over)
    assert r["environment_is_pinned"] is False
    assert r["checks"][failed]["ok"] is False and r["checks"][failed]["reason"]
    assert any(w["code"] == f"environment_not_pinned:{failed}" for w in r["warnings"])


def test_unverified_is_never_reported_as_pinned():
    assert ev(engines_at_warmup=None)["environment_is_pinned"] is False


def _client(**kw):
    analyzer = MrcdAnalyzer("u.pt", engine_factory=lambda o: FakeEngine())
    app = create_app(analyzer, **kw)
    return TestClient(app), app, analyzer


def test_health_reports_unpinned_before_warmup_and_never_claims_true():
    client, _, _ = _client(device="cpu")
    h = client.get("/v1/health").json()
    assert h["environment_is_pinned"] is False
    assert h["environment"]["checks"]["torch_threads"]["ok"] is False


def test_prepare_records_the_load_order_and_health_reflects_the_process(monkeypatch):
    import mrcd.service.server as S

    monkeypatch.setattr(S, "_torch_threads", lambda: 1)
    monkeypatch.setattr(S, "warm_up", lambda a, p: None)
    monkeypatch.setenv("OMP_NUM_THREADS", "4")
    client, app, analyzer = _client(device="cpu")
    prepare(app, analyzer, (AnalysisOptions(),), True)
    assert app.state.load_events == ["preload", "warmup"]
    h = client.get("/v1/health").json()
    checks = h["environment"]["checks"]
    assert checks["load_order"]["ok"] and checks["device"]["ok"]
    # truthful: the rest depends on how THIS test process was started (OMP before numpy)
    assert h["environment_is_pinned"] is checks["omp_set_before_numpy"]["ok"]
    monkeypatch.setenv("OMP_NUM_THREADS", "8")
    assert client.get("/v1/health").json()["environment_is_pinned"] is False


def test_start_up_logs_a_warning_when_not_pinned(caplog):
    _, app, _ = _client(device=None)
    with caplog.at_level(logging.WARNING, logger="mrcd.service.environment"):
        app.state.announce_environment()
    assert any(
        "environment_not_pinned:device" in r.getMessage() for r in caplog.records
    )


# ---- the real process: these FAIL if the environment differs from the documented pinned state
def test_this_process_runs_with_the_pinned_omp_value():
    assert os.environ.get("OMP_NUM_THREADS") == EP.EXPECTED["omp_num_threads"], (
        "run the suite with OMP_NUM_THREADS=4 exported before Python starts"
    )


def test_real_silero_import_leaves_torch_at_the_pinned_thread_count():
    code = "import torch, silero_vad; print(torch.get_num_threads())"
    env = {**os.environ, "OMP_NUM_THREADS": "4", "HF_HUB_OFFLINE": "1"}
    out = subprocess.run(
        [sys.executable, "-I", "-c", code],
        capture_output=True,
        text=True,
        env=env,
        cwd=ROOT,
        timeout=300,
    )
    assert int(out.stdout.split()[-1]) == EP.EXPECTED["torch_threads_after_silero"], (
        out.stderr[-400:]
    )


# ------------------------------------------------------------------ document provenance uses the real check
class _EnvEngine(FakeEngine):
    """FakeEngine whose result carries a (stale, OMP-value-only) environment block, like the real engine."""

    def analyze(self, audio, **kw):
        r = super().analyze(audio, **kw)
        r.metadata["environment"] = {"omp_num_threads": "4", "environment_is_pinned": True}
        return r


def _doc_env(analyzer):
    from tests.test_service_analyzer import wav

    doc = analyzer.analyze_bytes(wav(), AnalysisOptions())
    return doc["provenance"]["environment"]


def test_document_provenance_reports_the_real_pin_verdict_not_the_omp_value(monkeypatch):
    monkeypatch.setenv("OMP_NUM_THREADS", "4")  # the old flag would say True
    analyzer = MrcdAnalyzer("u.pt", engine_factory=lambda o: _EnvEngine())
    app = create_app(analyzer, device="cpu")  # create_app wires the probe; no warm-up has run
    assert app.state.environment()["environment_is_pinned"] is False
    env = _doc_env(analyzer)
    assert env["environment_is_pinned"] is False
    assert "torch_threads" in env["environment_pin_failed_checks"]


def test_document_provenance_flag_follows_the_probe_both_ways():
    analyzer = MrcdAnalyzer("u.pt", engine_factory=lambda o: _EnvEngine())
    analyzer.environment_probe = lambda: EP.evaluate(**GOOD)
    env = _doc_env(analyzer)
    assert env["environment_is_pinned"] is True and env["environment_pin_failed_checks"] == []
    analyzer.environment_probe = lambda: EP.evaluate(**{**GOOD, "device": "cuda"})
    env = _doc_env(analyzer)
    assert env["environment_is_pinned"] is False and env["environment_pin_failed_checks"] == ["device"]


def test_without_a_probe_the_document_is_not_pinned_and_says_why():
    analyzer = MrcdAnalyzer("u.pt", engine_factory=lambda o: _EnvEngine())  # engine reports the legacy True
    env = _doc_env(analyzer)
    assert env["environment_is_pinned"] is False
    assert env["environment_pin_failed_checks"] == ["not_probed"]
