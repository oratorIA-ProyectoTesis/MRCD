"""Guardias de validez del goal: fuga de identidad, soporte mínimo y esquema.

Estas pruebas protegen las propiedades que hacen que el sistema NO pueda emitir
una cifra inflada por construcción. Si alguna cae, el motor puede volver a
reportar un F1 que mide reconocimiento de charla en vez de detección de
disfluencias, y nada en la salida lo delataría.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from mrcd.constants import FEATURE_SCHEMA_VERSION, MIN_SUPPORT_FOR_METRIC
from mrcd.dataset import speaker_kinesic_norm

ROOT = Path(__file__).resolve().parents[1]


# ------------------------------------------- requisito 8: normalización cinésica
def test_speaker_norm_removes_resting_face_offset():
    """Dos hablantes con la misma dinámica pero distinto reposo deben converger.

    Es el caso que motiva el requisito: una persona con la mandíbula más abierta
    en reposo produce valores sistemáticamente mayores. Sin normalizar, el modelo
    aprende esa constante y con folds disjuntos por hablante esa constante es
    ruido en evaluación.
    """
    rng = np.random.default_rng(0)
    dinamica = rng.normal(0, 1, (40, 30, 12)).astype(np.float32)
    mask = np.ones((40, 30), bool)

    a = dinamica + 10.0  # reposo alto
    b = dinamica - 3.0  # reposo bajo

    na, ok_a = speaker_kinesic_norm(a, mask)
    nb, ok_b = speaker_kinesic_norm(b, mask)
    assert ok_a and ok_b

    assert abs(float(na.mean()) - float(nb.mean())) < 0.05, (
        "tras normalizar, el reposo deja de separarlos"
    )
    # y la dinámica sobrevive: la correlación temporal se conserva
    assert np.corrcoef(na.ravel(), nb.ravel())[0, 1] > 0.99


def test_speaker_norm_skips_when_too_few_real_frames():
    """Calibrar con cuatro frames mete más ruido del que quita."""
    kin = np.random.default_rng(1).normal(0, 1, (3, 30, 12)).astype(np.float32)
    mask = np.zeros((3, 30), bool)
    mask[0, :5] = True  # 5 frames reales, muy por debajo del mínimo
    out, applied = speaker_kinesic_norm(kin, mask)
    assert not applied
    assert np.array_equal(out, kin), "sin calibración fiable, se devuelve intacto"


def test_speaker_norm_ignores_interpolated_frames():
    """Los frames sin detección real no deben mover la media del hablante."""
    rng = np.random.default_rng(7)
    kin = np.zeros((10, 30, 12), np.float32)
    kin[:, :15] = rng.normal(1.0, 0.3, (10, 15, 12))  # reales, centrados en 1.0
    kin[:, 15:] = 99.0  # interpolados/basura: deben quedar fuera del cálculo
    mask = np.zeros((10, 30), bool)
    mask[:, :15] = True
    out, applied = speaker_kinesic_norm(kin, mask, min_frames=10)
    assert applied
    # la media salió solo de los reales -> estos quedan centrados en ~0 pese a que
    # los interpolados valen 99 y habrían desplazado la media por completo
    assert abs(float(out[:, :15].mean())) < 0.05
    assert float(out[:, 15:].mean()) > 50, (
        "los interpolados no se centran, solo se escalan"
    )


def test_speaker_norm_refuses_constant_channel():
    """Un canal constante tiene desviación 0: dividir amplificaría ruido numérico."""
    kin = np.ones((20, 30, 12), np.float32)
    mask = np.ones((20, 30), bool)
    _, applied = speaker_kinesic_norm(kin, mask, min_frames=10)
    assert not applied


# --------------------------------------- requisito 10: soporte mínimo por clase
def test_classes_below_min_support_are_excluded_from_macro():
    from scripts.run_ablation import f1s, measurable_classes

    n_cls = 4
    y = np.array([0] * 50 + [1] * 30 + [2] * 30 + [3] * 2)  # clase 3 con n=2
    assert 3 not in measurable_classes(y, n_cls)
    assert 1 in measurable_classes(y, n_cls) and 2 in measurable_classes(y, n_cls)

    # un modelo que acierta todo salvo la clase rara no debe ser penalizado por
    # una clase que no es medible, ni premiado por acertarla de casualidad
    p = y.copy()
    p[y == 3] = 0
    _, macro = f1s(y, p, n_cls)
    assert macro == pytest.approx(1.0), "la clase sin soporte no entra en el macro-F1"


def test_min_support_threshold_is_above_one():
    """Un umbral de 1 haría que n=2 contara como medible y anularía el requisito."""
    assert MIN_SUPPORT_FOR_METRIC >= 5


# ------------------------------------------- requisito 7: guardia de partición
def _run(*extra: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/run_ablation.py"),
            "--data",
            str(ROOT / "data/windows_auto.npz"),
            "--test-data",
            str(ROOT / "data/windows_gold.npz"),
            *extra,
        ],
        capture_output=True,
        text=True,
        timeout=180,
    )


@pytest.mark.skipif(
    not (ROOT / "data/windows_auto.npz").exists(), reason="sin dataset local"
)
@pytest.mark.parametrize("partition", ["recording", "window"])
def test_leaky_partition_aborts_without_explicit_flag(partition):
    r = _run("--partition", partition)
    assert r.returncode != 0, f"--partition {partition} debe abortar sin --leaky-ok"
    assert "INFLADO" in r.stdout + r.stderr


@pytest.mark.skipif(
    not (ROOT / "data/windows_auto.npz").exists(), reason="sin dataset local"
)
def test_speaker_partition_is_the_default():
    """Que la opción segura sea el default no es cosmético: un usuario que no lee
    la ayuda debe obtener la métrica honesta, no la inflada."""
    r = _run("--help")
    assert "--partition" in r.stdout
    assert "default" not in r.stdout.lower() or "speaker" in r.stdout


# -------------------------------------- requisito 11: versión de esquema
def test_schema_version_is_stamped_on_new_datasets(tmp_path):
    """Un npz sin versión no permite saber qué código lo generó."""
    npz = ROOT / "data/v3/windows_auto.npz"
    if not npz.exists():
        pytest.skip("sin dataset v3 local")
    d = np.load(npz, allow_pickle=False)
    assert "schema_version" in d
    assert int(d["schema_version"]) == FEATURE_SCHEMA_VERSION
    assert str(d["kinesic_norm"]) in ("speaker", "raw")
