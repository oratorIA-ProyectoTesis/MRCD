"""La ablación debe negarse a correr cuando train y test comparten etiquetador.

El fallo que originó este archivo: `run_v2.ps1` llamaba a `run_ablation.py` con
`--test-data data/windows_gold.npz` pero sin `--data`. El valor por defecto de
`--data` es ese mismo archivo, así que el benchmark entrenó y evaluó sobre las
mismas 503 ventanas gold. El resultado subió —macro-F1 trimodal de 0.318 a
0.383— y parecía una mejora de la arquitectura.

No lo era: un test que hereda el sesgo del etiquetador que produjo el
entrenamiento mide ajuste al etiquetador, no acierto. Y el ruido sale barato
justo donde más importa. Por eso la comprobación vive en el código y no en el
script: un guion puede olvidar un argumento, el guardia no.
"""
from __future__ import annotations

import subprocess
import sys
import json
from pathlib import Path

import numpy as np
import pytest

from core.constants import LABELS
from core.contracts import FEATURE_SCHEMA_VERSION

ROOT = Path(__file__).resolve().parents[1]
ABL = ROOT / "scripts/run_ablation.py"


def _npz(path: Path, source: str, n: int = 40, labels=None) -> Path:
    rng = np.random.default_rng(0)
    labels = labels or ["fluent", "filler_word", "block"]
    np.savez_compressed(
        path,
        label_source=np.array(source), labels=np.array(labels),
        ac=rng.normal(size=(n, 300, 5)).astype(np.float32),
        pros=rng.normal(size=(n, 9)).astype(np.float32),
        ling=rng.normal(size=(n, 40, 12)).astype(np.float32),
        lpos=rng.integers(0, 100, (n, 40)), lmask=np.ones((n, 40), bool),
        kin=rng.normal(size=(n, 30, 12)).astype(np.float32),
        kmask=np.ones((n, 30), bool), has_video=np.ones(n, bool),
        y=rng.integers(0, len(labels), n), multi=np.zeros((n, 2), np.int8),
        start_ms=(np.arange(n) * 500).astype(np.int64),
        group=np.array([f"s{i % 6}" for i in range(n)]),
        rec=np.array([f"r{i % 6}" for i in range(n)]))
    return path


def _run(*args) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(ABL), *args, "--folds", "2", "--epochs", "1",
                           "--bootstrap", "5"], capture_output=True, text=True, cwd=ROOT)


def test_refuses_when_data_and_test_data_are_the_same_file(tmp_path):
    g = _npz(tmp_path / "gold.npz", "gold_llm")
    r = _run("--data", str(g), "--test-data", str(g), "--out", str(tmp_path))
    assert r.returncode != 0
    assert "MISMO archivo" in r.stdout + r.stderr


def test_refuses_when_both_sides_share_the_label_source(tmp_path):
    """El caso real: dos archivos distintos, mismo etiquetador."""
    a = _npz(tmp_path / "a.npz", "gold_llm")
    b = _npz(tmp_path / "b.npz", "gold_llm")
    r = _run("--data", str(a), "--test-data", str(b), "--out", str(tmp_path))
    assert r.returncode != 0
    out = r.stdout + r.stderr
    assert "comparten origen de etiquetas" in out
    assert "circular" in out.lower()
    assert "windows_auto.npz" in out, "debe decir cómo corregirlo"


def test_refuses_different_ordered_legacy_taxonomies_before_training(tmp_path):
    a = _npz(tmp_path / "auto.npz", "auto")
    g = _npz(tmp_path / "gold.npz", "gold_llm",
             labels=["fluent", "block", "filler_word"])
    out = tmp_path / "results"
    r = _run("--data", str(a), "--test-data", str(g), "--out", str(out))
    assert r.returncode != 0
    assert "ordered labels differ" in r.stdout + r.stderr
    assert not out.exists()


def test_the_weak_supervision_setup_is_allowed(tmp_path):
    a = _npz(tmp_path / "auto.npz", "auto")
    g = _npz(tmp_path / "gold.npz", "gold_llm")
    r = _run("--data", str(a), "--test-data", str(g), "--out", str(tmp_path))
    assert r.returncode == 0, f"la combinación válida no debe bloquearse:\n{r.stdout}\n{r.stderr}"
    assert "'auto'" in r.stdout and "'gold_llm'" in r.stdout


def test_all_zero_frame_gold_is_reported_as_audio_text_fallback(tmp_path):
    train = _npz(tmp_path / "auto.npz", "auto", labels=list(LABELS))
    gold = _npz(tmp_path / "gold.npz", "gold_llm", labels=list(LABELS))
    with np.load(gold, allow_pickle=False) as source:
        payload = {key: source[key] for key in source.files}
    n = len(payload["y"])
    payload.update(feature_schema=np.array(FEATURE_SCHEMA_VERSION),
                   judge_n_frames=np.zeros(n, np.int16),
                   judge_visual_evidence_used=np.zeros(n, bool),
                   judge_provider=np.array(["openai"] * n),
                   judge_model=np.array(["gpt-4o"] * n),
                   judge_input_fingerprint=np.array(["current"] * n),
                   judge_fallback_note=np.array(["sin_fotogramas"] * n))
    np.savez_compressed(gold, **payload)
    out = tmp_path / "results"
    run = _run("--data", str(train), "--test-data", str(gold), "--out", str(out))
    assert run.returncode == 0, run.stdout + run.stderr
    report = json.loads((out / "ablation_results.json").read_text())
    assert report["judge_visual_evidence"]["status"] == "audio_text_only"
    assert report["judge_visual_evidence"]["fallback_rate"] == 1.0
    assert "NO es una evaluación de juez multimodal" in (out / "ablation_table.md").read_text()


def test_same_source_is_possible_but_must_be_explicit(tmp_path):
    """El techo supervisado es un experimento legítimo; sólo no puede salir por
    descuido."""
    a = _npz(tmp_path / "a.npz", "gold_llm")
    b = _npz(tmp_path / "b.npz", "gold_llm")
    r = _run("--data", str(a), "--test-data", str(b), "--allow-same-label-source",
             "--out", str(tmp_path))
    assert r.returncode == 0, f"{r.stdout}\n{r.stderr}"


def test_guard_runs_before_any_training(tmp_path):
    """Debe fallar en segundos, no tras horas de cómputo."""
    import time
    a = _npz(tmp_path / "a.npz", "gold_llm")
    b = _npz(tmp_path / "b.npz", "gold_llm")
    t0 = time.time()
    r = _run("--data", str(a), "--test-data", str(b), "--epochs", "200", "--out", str(tmp_path))
    assert r.returncode != 0
    assert time.time() - t0 < 30, "el guardia debe actuar antes de entrenar"


# ------------------------------------------------- el script llama bien
def test_run_v2_passes_the_training_set_explicitly():
    """Sin --data, el valor por defecto es windows_gold.npz y el benchmark
    entrena sobre el propio test."""
    t = (ROOT / "run_v2.ps1").read_text(encoding="ascii")
    for line in t.split("\r\n"):
        if "run_ablation.py" in line and "--test-data" in line:
            assert "--data" in line, f"falta --data en: {line.strip()}"
            assert "windows_auto.npz" in line, f"debe entrenar con el conjunto silver: {line.strip()}"


@pytest.mark.parametrize("script", ["run_v2.ps1", "run_agentic.ps1"])
def test_no_script_calls_the_ablation_without_a_training_set(script):
    t = (ROOT / script).read_text(encoding="ascii")
    for line in t.split("\r\n"):
        if "run_ablation.py" not in line or "--test-data" not in line:
            continue
        assert "--data" in line, f"{script}: --test-data sin --data -> circularidad"
