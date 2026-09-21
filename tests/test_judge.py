"""Pruebas del juez multimodal y del test independiente (sin llamadas de red)."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.constants import BACKGROUND, LABEL2ID, TAXONOMY
from core.dataset import gold_windows
import scripts.llm_judge as lj
import scripts.judge_agreement as ja


def test_gold_windows_one_per_judgment():
    starts = np.arange(0, 10, 0.5)
    ev = [dict(category="block", start_ms=2000, end_ms=2900),
          dict(category=BACKGROUND, start_ms=6000, end_ms=9000),
          dict(category="no_existe", start_ms=0, end_ms=100)]
    idx, lab = gold_windows(ev, starts)
    assert len(idx) == 2 and lab.tolist() == [LABEL2ID["block"], LABEL2ID[BACKGROUND]]
    assert abs(starts[idx[0]] + 1.5 - 2.45) < 0.3      # ventana centrada en el evento


def test_sampling_is_stratified_and_includes_negatives(tmp_path):
    recs, man = [], {"recordings": []}
    for s in range(3):
        rid = f"r{s}"
        ev = [dict(category=c, start_ms=1000 * (i + 1), end_ms=1000 * (i + 1) + 600, confidence=0.8, source="auto")
              for i, c in enumerate(["filler_word", "block", "repetition"])]
        ev.append(dict(category="block", start_ms=9000, end_ms=9400, confidence=0.4, source="auto_low_conf"))
        p = tmp_path / f"{rid}.events.json"; p.write_text(json.dumps(ev))
        man["recordings"].append(dict(recording_id=rid, speaker_id=rid, duration_s=60.0,
                                      events_json=str(p.relative_to(tmp_path))))
    lj.ROOT = tmp_path
    items = lj.sample_items(man, per_class=2, low_conf=2, negatives=5, seed=1)
    origins = {i["origin"] for i in items}
    assert origins == {"candidate", "low_conf", "negative"}
    assert sum(i["origin"] == "negative" for i in items) == 5
    assert sum(i["origin"] == "candidate" and i["heur"] == "block" for i in items) == 2
    # los negativos no solapan candidatos
    for it in items:
        if it["origin"] == "negative":
            assert it["end"] - it["start"] == pytest.approx(3.0)


def test_prompt_is_blind():
    """El prompt no debe filtrar la etiqueta ni la evidencia heurística."""
    p = lj.build_prompt("hola que tal", "(sin palabras: silencio)", "entonces",
                        dict(duracion_s=0.9, proporcion_silencio=1.0))
    for leak in ("mouth_press", "heurístic", "candidato", "evidence", "auto_low_conf", "kinesic"):
        assert leak not in p            # el juez no ve la decisión ni la evidencia de las reglas
    assert "fluent" in p and "TAXONOMÍA" in p


def test_kappa_matches_known_value():
    a = ["x"] * 8 + ["y"] * 2
    b = ["x"] * 6 + ["y"] * 4
    assert -1 <= ja.kappa(a, b) <= 1
    assert ja.kappa(a, a) == pytest.approx(1.0)


def test_write_gold_groups_by_recording(tmp_path):
    lj.ROOT = tmp_path
    (tmp_path / "f").mkdir()
    man = {"recordings": [dict(recording_id="r0", events_json="f/r0.events.json"),
                          dict(recording_id="r1", events_json="f/r1.events.json")]}
    done = {("r0", 1.0, 2.0): dict(rid="r0", start=1.0, end=2.0, judge="block", confidence=0.9, model="m",
                                   heuristic="neutral_pause", origin="candidate", rationale="x"),
            ("r1", 3.0, 4.0): dict(rid="r1", start=3.0, end=4.0, judge=BACKGROUND, confidence=0.7, model="m",
                                   heuristic=BACKGROUND, origin="negative", rationale="y")}
    lj.write_gold(man, done)
    g0 = json.loads((tmp_path / "f/r0.gold.json").read_text())
    assert g0[0]["category"] == "block" and g0[0]["source"] == "gold_llm"
    assert json.loads((tmp_path / "f/r1.gold.json").read_text())[0]["category"] == BACKGROUND
