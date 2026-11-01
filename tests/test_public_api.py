"""Contrato de la API publica del paquete.

Estas pruebas protegen lo que un proyecto ajeno puede asumir. Si alguna cae, la
integracion de alguien se rompe en una version menor, que es justo lo que una API
publica promete que no pasara.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

import mrcd
from mrcd.constants import STRIDE_S, WINDOW_S
from mrcd.engine import AnalysisResult, DisfluencyEngine, Event, Window


# ----------------------------------------------------- superficie publica
def test_public_names_are_importable_from_the_root():
    """Un integrador importa de `mrcd`, no de `mrcd.engine.submodulo`."""
    for name in mrcd.__all__:
        assert hasattr(mrcd, name), f"{name} esta en __all__ pero no existe"


def test_library_does_not_depend_on_scripts():
    """La libreria no puede importar de `scripts/`.

    `scripts/` no se empaqueta (pyproject solo incluye `mrcd*`), asi que un
    import asi funcionaria en el repo y fallaria al instalar el paquete.
    """
    from pathlib import Path

    pkg = Path(mrcd.__file__).parent
    ofensores = [
        p.name
        for p in pkg.rglob("*.py")
        if "from scripts" in p.read_text(encoding="utf-8")
        or "import scripts" in p.read_text(encoding="utf-8")
    ]
    assert not ofensores, f"la libreria importa de scripts/: {ofensores}"


def test_taxonomy_is_stable():
    """Las clases son el contrato con el integrador: no se reordenan ni renombran."""
    assert mrcd.TAXONOMY == (
        "filler_word",
        "prolongation",
        "repetition",
        "block",
        "revision",
        "rhetorical_pause",
        "neutral_pause",
    )
    assert mrcd.BACKGROUND == "fluent"
    assert mrcd.LABELS[0] == mrcd.BACKGROUND


# ------------------------------------------------------ fusion de eventos
def _win(start: float, label: str, conf: float = 0.9) -> Window:
    return Window(start, start + WINDOW_S, label, conf, {label: conf})


def _merge(windows):
    """Invoca la fusion sin construir un motor real (no necesita checkpoint)."""
    eng = DisfluencyEngine.__new__(DisfluencyEngine)
    eng._net = type("N", (), {"mode": "audio_text"})()
    return DisfluencyEngine._merge(eng, windows)


def test_events_do_not_overlap_each_other():
    """El fallo que esto previene.

    Las ventanas se solapan por diseno (3 s de ancho, 0.5 s de paso). Si el evento
    terminara en `ultima_ventana.start + WINDOW_S`, dos eventos consecutivos de
    clases distintas se pisarian hasta 2.5 s, y una linea de tiempo mostraria dos
    clases activas en el mismo instante.
    """
    ws = [
        _win(0.0, "filler_word"),
        _win(0.5, "filler_word"),
        _win(1.0, "repetition"),
        _win(1.5, "repetition"),
    ]
    evs = _merge(ws)
    assert len(evs) == 2
    for a, b in zip(evs, evs[1:]):
        assert a.end <= b.start, (
            f"solape: {a.label} termina en {a.end}, {b.label} empieza en {b.start}"
        )


def test_event_span_uses_stride_not_window_width():
    ws = [_win(0.0, "repetition"), _win(0.5, "repetition"), _win(1.0, "repetition")]
    ev = _merge(ws)[0]
    assert ev.start == pytest.approx(0.0)
    assert ev.end == pytest.approx(1.0 + STRIDE_S)
    assert ev.n_windows == 3


def test_background_windows_produce_no_events():
    """`fluent` es la clase de fondo: no es un evento que reportar."""
    assert _merge([_win(0.0, "fluent"), _win(0.5, "fluent")]) == []


def test_a_gap_splits_one_run_into_two_events():
    ws = [_win(0.0, "filler_word"), _win(0.5, "filler_word"), _win(30.0, "filler_word")]
    evs = _merge(ws)
    assert len(evs) == 2, "un salto temporal no debe fundirse en un solo evento"


def test_confidence_is_the_peak_of_the_run():
    ws = [_win(0.0, "block", 0.40), _win(0.5, "block", 0.95), _win(1.0, "block", 0.60)]
    assert _merge(ws)[0].confidence == pytest.approx(0.95)


# ------------------------------------------------------------- resultado
def test_result_is_iterable_and_serializable():
    r = AnalysisResult(
        events=[Event("filler_word", 1.0, 1.5, 0.9, 1)],
        duration=60.0,
        mode="audio_text",
        warning="aviso",
    )
    assert len(r) == 1
    assert [e.label for e in r] == ["filler_word"]
    assert r.by_label("filler_word") and not r.by_label("block")
    assert r.rate_per_minute() == pytest.approx(1.0)
    d = json.loads(json.dumps(r.to_dict()))  # debe ser JSON puro
    assert d["warning"] == "aviso" and d["events"][0]["label"] == "filler_word"


def test_result_carries_the_provenance_warning():
    """El aviso viaja en el resultado, no solo en la documentacion.

    Un integrador que serialice `to_dict()` y lo mande a su frontend se lleva el
    aviso consigo; no puede perderse por no haber leido el README.
    """
    assert "warning" in AnalysisResult(events=[]).to_dict()


def test_rate_per_minute_is_unavailable_not_zero_for_negligible_audio():
    r = AnalysisResult(events=[], duration=0.0)
    assert r.rate_per_minute() is None  # absence is never zero
    assert AnalysisResult(events=[], duration=1.0).rate_per_minute() is None
    assert r.rate_per_minute(legacy_zero=True) == 0.0  # historical behaviour, opt-in
    assert AnalysisResult(events=[], duration=30.0).rate_per_minute() == 0.0  # measured zero is real


# --------------------------------------------------------- validaciones
def _fake_engine():
    eng = DisfluencyEngine.__new__(DisfluencyEngine)
    eng._net = type("N", (), {"mode": "audio_text"})()
    return eng


def test_wrong_sample_rate_fails_loudly():
    """Remuestrear en silencio ocultaria un desajuste del pipeline del integrador."""
    with pytest.raises(ValueError, match="16000"):
        DisfluencyEngine.analyze(
            _fake_engine(), np.zeros(48000, np.float32), sample_rate=48000
        )


def test_audio_shorter_than_the_context_window_fails_loudly():
    with pytest.raises(ValueError, match="al menos"):
        DisfluencyEngine.analyze(_fake_engine(), np.zeros(16000 * 5, np.float32))


def test_minimum_audio_matches_the_context_the_model_needs():
    from mrcd.constants import CONTEXT_S

    assert DisfluencyEngine.MIN_AUDIO_S == CONTEXT_S + WINDOW_S
