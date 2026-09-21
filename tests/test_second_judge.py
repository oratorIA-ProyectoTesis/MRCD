from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import scripts.llm_judge as J
from scripts.qa_report import cohen_kappa


class _Response:
    def __init__(self, payload):
        self.payload = json.dumps(payload).encode()

    def read(self):
        return self.payload

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def test_gemini_safety_block_is_surfaced(monkeypatch):
    import urllib.request
    monkeypatch.setattr(urllib.request, "urlopen",
                        lambda *args, **kwargs: _Response({"promptFeedback": {"blockReason": "SAFETY"}}))
    with pytest.raises(J.JudgeRefusal, match="SAFETY"):
        J.call_gemini("gemini-2.0-flash", "p", [], "k")


def test_gemini_empty_candidates_are_rejected(monkeypatch):
    import urllib.request
    monkeypatch.setattr(urllib.request, "urlopen", lambda *args, **kwargs: _Response({"candidates": []}))
    with pytest.raises(J.JudgeRefusal, match="sin candidatos"):
        J.call_gemini("gemini-2.0-flash", "p", [], "k")


def test_gemini_normal_response_is_parseable(monkeypatch):
    import urllib.request
    payload = {"candidates": [{"finishReason": "STOP", "content": {
        "parts": [{"text": '{"classification":"fluent","confidence":0.9}'}]}}]}
    monkeypatch.setattr(urllib.request, "urlopen", lambda *args, **kwargs: _Response(payload))
    raw = J.call_gemini("gemini-2.0-flash", "p", [], "k")
    assert J.parse_verdict(raw, J.CLASSES)["classification"] == "fluent"


def test_gemini_http_error_explains_model_failure(monkeypatch):
    import urllib.error
    import urllib.request
    error = urllib.error.HTTPError("https://example.invalid", 404, "Not Found", {},
                                    __import__("io").BytesIO(b'{"error":"model missing"}'))
    monkeypatch.setattr(urllib.request, "urlopen", lambda *args, **kwargs: (_ for _ in ()).throw(error))
    with pytest.raises(RuntimeError, match="Gemini HTTP 404"):
        J.call_gemini("gemini-missing", "p", [], "k")


def test_cohen_kappa_perfect_agreement():
    rows = [{"window_index": 1, "label": "fluent", "n_frames": 3},
            {"window_index": 2, "label": "block", "n_frames": 3}]
    assert cohen_kappa(rows, rows) == pytest.approx(1.0)


def test_cohen_kappa_chance_agreement_is_zero():
    a = [{"window_index": i, "label": label, "n_frames": 3}
         for i, label in enumerate(("fluent", "block", "fluent", "block"))]
    b = [{"window_index": i, "label": label, "n_frames": 3}
         for i, label in enumerate(("fluent", "fluent", "block", "block"))]
    assert cohen_kappa(a, b) == pytest.approx(0.0)


def test_cohen_kappa_imbalanced_classes_does_not_divide_by_zero():
    rows = [{"window_index": i, "label": "fluent", "n_frames": 3} for i in range(4)]
    assert cohen_kappa(rows, rows) == pytest.approx(1.0)


def test_local_judge_rejects_a_model_without_vision(monkeypatch):
    import scripts.second_judge_local as S

    monkeypatch.setattr(S.subprocess, "run", lambda *args, **kwargs: subprocess.CompletedProcess(
        args[0], 0, stdout="Model\n  capabilities    completion tools\n", stderr=""))
    with pytest.raises(RuntimeError, match="no declara la capability 'vision'"):
        S.require_vision_model("llama3.1:8b")


def test_local_judge_accepts_a_model_with_vision(monkeypatch):
    import scripts.second_judge_local as S

    monkeypatch.setattr(S.subprocess, "run", lambda *args, **kwargs: subprocess.CompletedProcess(
        args[0], 0, stdout="Model\n  capabilities    completion vision\n", stderr=""))
    S.require_vision_model("qwen2.5vl:7b")


def test_local_judge_requires_visual_coverage():
    import scripts.second_judge_local as S

    rows = [{"n_frames": 3} for _ in range(54)] + [{"n_frames": 0} for _ in range(6)]
    assert S.visual_coverage(rows) == pytest.approx(0.9)
    with pytest.raises(RuntimeError, match="cobertura visual insuficiente"):
        S.require_visual_coverage(rows[:53], expected=60)


def test_second_judge_skips_existing_window(tmp_path, monkeypatch):
    import scripts.second_judge_gemini as S

    case = tmp_path / "case"
    case.mkdir()
    (case / "X.adjudication_queue.json").write_text(json.dumps([{"window_index": 7, "start": 0, "end": 3}]))
    (case / "X.meta.json").write_text(json.dumps({"words_json": str(tmp_path / "words.json"),
                                                   "features_npz": str(tmp_path / "features.npz"),
                                                   "mp4": str(tmp_path / "x.mp4")}))
    (tmp_path / "words.json").write_text("[]")
    np.savez(tmp_path / "features.npz", placeholder=np.zeros(1))
    (case / "X.adjudication_gemini.jsonl").write_text(
        json.dumps({"window_index": 7, "label": "fluent"}) + "\n")
    monkeypatch.setattr(S, "CASE", case)
    monkeypatch.setenv("GEMINI_API_KEY", "test")
    monkeypatch.setattr(sys, "argv", ["second_judge_gemini.py", "--vid", "X"])
    monkeypatch.setattr(S.J, "load_env_file", lambda: None)
    monkeypatch.setattr(S.J, "ask_judge", lambda *args: (_ for _ in ()).throw(AssertionError("called")))
    S.main()
