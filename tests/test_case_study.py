"""Pruebas del flujo de estudio de caso: seguimiento facial, checkpoints, eventos, QA."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.constants import BACKGROUND, KIDX, KINESIC_DIM, LABELS, STRIDE_S, WINDOW_S
from core.extractors.face_tracking import (FaceTracker, Track, bbox_area, bbox_iou,
                                           dominance_margin, result_to_faces)


# ------------------------------------------------------------- seguimiento facial
def _box(cx, cy, w, h):
    return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], np.float32)


def test_iou_basics():
    a = _box(0.5, 0.5, 0.2, 0.2)
    assert bbox_iou(a, a) == pytest.approx(1.0)
    assert bbox_iou(a, _box(0.9, 0.9, 0.2, 0.2)) == 0.0
    assert 0.0 < bbox_iou(a, _box(0.55, 0.5, 0.2, 0.2)) < 1.0
    assert bbox_iou(a, None) == 0.0
    assert bbox_iou(a, np.full(4, np.nan, np.float32)) == 0.0


def test_tracker_keeps_identities_when_mediapipe_reorders_faces():
    """El fallo que esto previene.

    MediaPipe no garantiza orden estable de caras. Aquí el orden de la lista se
    invierte en frames alternos, como ocurre en la práctica. Si el extractor
    tomara siempre la cara [0], el vector saltaría entre las dos personas.
    """
    entrevistado = _box(0.35, 0.5, 0.30, 0.30)     # grande, siempre presente
    entrevistador = _box(0.80, 0.5, 0.12, 0.12)    # pequeño
    tr = FaceTracker()
    for fi in range(20):
        ve = np.full(KINESIC_DIM, 1.0, np.float32)     # marca del entrevistado
        vr = np.full(KINESIC_DIM, 9.0, np.float32)     # marca del entrevistador
        faces = [(ve, entrevistado), (vr, entrevistador)]
        if fi % 2:                                      # el orden se invierte
            faces = faces[::-1]
        tr.update(fi, fi * 0.1, faces)

    assert len(tr.tracks) == 2, "debe formar exactamente dos identidades"
    dom = tr.dominant()
    assert dom is not None
    vals = np.array(dom.vecs)
    assert np.allclose(vals, 1.0), "la pista dominante no debe mezclar al otro rostro"
    assert len(dom.frames) == 20


def test_dominant_is_the_persistent_large_face_not_a_brief_closeup():
    tr = FaceTracker()
    big = _box(0.4, 0.5, 0.28, 0.28)
    for fi in range(40):                       # entrevistado, todo el rato
        faces = [(np.full(KINESIC_DIM, 1.0, np.float32), big)]
        if 10 <= fi < 16:                      # plano corto del entrevistador, 6 frames
            faces.append((np.full(KINESIC_DIM, 9.0, np.float32), _box(0.85, 0.5, 0.34, 0.34)))
        tr.update(fi, fi * 0.1, faces)
    dom = tr.dominant()
    assert np.allclose(np.array(dom.vecs), 1.0)
    rows = tr.report()
    assert rows[0]["score"] > rows[1]["score"]
    assert dominance_margin(rows) > 0.25


def test_dominance_margin_flags_a_two_person_split():
    """Cuando el montaje reparte el tiempo, el margen debe avisar."""
    tr = FaceTracker()
    a, b = _box(0.3, 0.5, 0.25, 0.25), _box(0.7, 0.5, 0.25, 0.25)
    for fi in range(40):
        v = np.full(KINESIC_DIM, 1.0, np.float32)
        tr.update(fi, fi * 0.1, [(v, a if fi < 20 else b)])
    assert dominance_margin(tr.report()) < 0.25


def test_track_survives_short_detection_gap():
    tr = FaceTracker(max_gap=5)
    b = _box(0.5, 0.5, 0.2, 0.2)
    for fi in range(10):
        tr.update(fi, fi * 0.1, [] if 4 <= fi <= 6 else [(np.ones(KINESIC_DIM, np.float32), b)])
    assert len(tr.tracks) == 1, "un hueco corto no debe crear una identidad nueva"


def test_result_to_faces_returns_every_face():
    class _P:
        def __init__(s, x, y): s.x, s.y = x, y

    class _R:
        face_landmarks = [[_P(0.1, 0.1), _P(0.3, 0.4)], [_P(0.6, 0.2), _P(0.9, 0.5)]]
        facial_transformation_matrixes = [np.eye(4), np.eye(4)]
        face_blendshapes = [[], []]

    faces = result_to_faces(_R())
    assert len(faces) == 2, "debe devolver las dos caras, no sólo la primera"
    assert bbox_area(faces[0][1]) > 0 and bbox_area(faces[1][1]) > 0


# ----------------------------------------------------------------- eventos
def test_merge_events_collapses_overlapping_windows():
    """Una disfluencia cae en ~6 ventanas; contar ventanas inflaría x6."""
    from scripts.infer_video import merge_events

    labels = list(LABELS)
    fid, bid = labels.index(BACKGROUND), labels.index("filler_word")
    starts = np.arange(10) * STRIDE_S
    cls = np.array([fid, fid, bid, bid, bid, bid, fid, fid, bid, fid])
    probs = np.zeros((10, len(labels)), np.float32)
    probs[np.arange(10), cls] = 0.8
    ev = merge_events(starts, cls, probs, labels)
    assert len(ev) == 2, "cuatro ventanas contiguas son UN evento, no cuatro"
    assert ev[0]["n_windows"] == 4
    # la racha son las ventanas 2..5 -> arranca en starts[2] y termina en starts[5] + ventana
    assert ev[0]["start"] == pytest.approx(1.0)
    assert ev[0]["end"] == pytest.approx(2.5 + WINDOW_S)
    assert all(e["class"] == "filler_word" for e in ev)


def test_merge_events_splits_on_a_time_gap():
    from scripts.infer_video import merge_events

    labels = list(LABELS)
    bid = labels.index("block")
    starts = np.array([0.0, 0.5, 8.0, 8.5])      # salto temporal en medio
    cls = np.array([bid, bid, bid, bid])
    probs = np.zeros((4, len(labels)), np.float32); probs[:, bid] = 0.7
    assert len(merge_events(starts, cls, probs, labels)) == 2


# ------------------------------------------------- rasgos cinésicos por ventana
def test_mouth_press_peak_measures_rise_not_resting_tone():
    from scripts.infer_video import mouth_press_peak

    n = 30
    calm = np.zeros((n, KINESIC_DIM), np.float32)
    calm[:, [KIDX["mouth_press_left"], KIDX["mouth_press_right"]]] = 0.60   # tono alto constante
    tense = np.zeros((n, KINESIC_DIM), np.float32)
    tense[:, [KIDX["mouth_press_left"], KIDX["mouth_press_right"]]] = 0.10
    tense[12:18, [KIDX["mouth_press_left"], KIDX["mouth_press_right"]]] = 0.55  # pico
    m = np.ones(n, bool)
    assert mouth_press_peak(calm, m) < 0.05, "un tono de reposo alto no es un pico"
    assert mouth_press_peak(tense, m) > 0.35, "un ascenso real sí debe detectarse"


def test_kinesic_features_are_zero_without_face():
    from scripts.infer_video import gaze_stability, mouth_press_peak

    k = np.zeros((30, KINESIC_DIM), np.float32)
    assert mouth_press_peak(k, np.zeros(30, bool)) == 0.0
    assert not np.isfinite(gaze_stability(k, np.zeros(30, bool)))


def test_reported_visual_metrics_obey_real_frame_coverage_gate():
    from scripts.infer_video import visual_metrics

    k = np.zeros((30, KINESIC_DIM), np.float32)
    k[15, [KIDX["mouth_press_left"], KIDX["mouth_press_right"]]] = 0.8
    mask = np.ones(30, bool)  # Interpolation can populate the whole grid.
    assert visual_metrics(k, mask, False) == (None, None)
    peak, gaze = visual_metrics(k, mask, True)
    assert peak > 0.7 and gaze == 0.0


# ------------------------------------------- checkpoints: la norma viaja dentro
def test_checkpoint_roundtrip_preserves_predictions(tmp_path):
    """Si la normalización no viaja en el checkpoint, la inferencia da basura."""
    from core.models.fusion import CrossModalFusion
    from scripts.infer_video import load_ckpt, predict
    from scripts.train_checkpoints import apply_stats, stats

    rng = np.random.default_rng(0)
    n = 24
    d = dict(ac=rng.normal(3.0, 7.0, (n, 300, 5)).astype(np.float32),
             pros=rng.normal(-2.0, 4.0, (n, 8)).astype(np.float32),
             ling=rng.normal(0.0, 2.0, (n, 40, 12)).astype(np.float32),
             lpos=rng.integers(0, 100, (n, 40)), lmask=np.ones((n, 40), bool),
             kin=rng.normal(0.0, 1.0, (n, 30, 12)).astype(np.float32),
             kmask=np.ones((n, 30), bool), has_video=np.ones(n, bool))
    st = stats(d)
    net = CrossModalFusion(len(LABELS), ac_dim=5, ling_dim=12, mode="trimodal", pros_dim=8).eval()

    p = tmp_path / "ckpt_trimodal.pt"
    torch.save({"state_dict": net.state_dict(), "mode": "trimodal", "labels": list(LABELS),
                "dims": {"ac": 5, "ling": 12, "pros": 8, "d": 64},
                "norm": {k: (mu.tolist(), sd.tolist()) for k, (mu, sd) in st.items()},
                "provenance": {"label_source": "auto"}}, p)

    net2, norm, labels, prov = load_ckpt(p, "cpu")
    got = predict(net2, norm, d, "cpu")

    ds = apply_stats(d, st)
    with torch.no_grad():
        b = {k: torch.as_tensor(ds[k]) for k in ("ac", "pros", "ling", "lpos", "lmask", "kin", "kmask", "has_video")}
        for k in ("ac", "pros", "ling", "kin"):
            b[k] = b[k].float()
        want = torch.softmax(net(b), -1).numpy()
    assert np.allclose(got, want, atol=1e-5), "la norma guardada debe reproducir el entrenamiento"
    assert labels == list(LABELS) and prov["label_source"] == "auto"


def test_checkpoint_without_norm_changes_predictions():
    """Comprobación de que la normalización IMPORTA (si no, la prueba anterior
    pasaría por casualidad)."""
    from core.models.fusion import CrossModalFusion
    from scripts.train_checkpoints import apply_stats, stats

    rng = np.random.default_rng(1)
    n = 16
    d = dict(ac=rng.normal(5.0, 9.0, (n, 300, 5)).astype(np.float32),
             pros=rng.normal(0.0, 3.0, (n, 8)).astype(np.float32),
             ling=rng.normal(0.0, 2.0, (n, 40, 12)).astype(np.float32),
             lpos=rng.integers(0, 100, (n, 40)), lmask=np.ones((n, 40), bool),
             kin=rng.normal(0.0, 1.0, (n, 30, 12)).astype(np.float32),
             kmask=np.ones((n, 30), bool), has_video=np.ones(n, bool))
    net = CrossModalFusion(len(LABELS), ac_dim=5, ling_dim=12, mode="trimodal", pros_dim=8).eval()

    def fwd(x):
        with torch.no_grad():
            b = {k: torch.as_tensor(x[k]) for k in ("ac", "pros", "ling", "lpos", "lmask", "kin", "kmask", "has_video")}
            for k in ("ac", "pros", "ling", "kin"):
                b[k] = b[k].float()
            return torch.softmax(net(b), -1).numpy()

    assert not np.allclose(fwd(d), fwd(apply_stats(d, stats(d))), atol=1e-3)


# ------------------------------------------------------- blindaje del árbitro
def test_adjudication_queue_hides_predictions_from_the_judge():
    """El juez no puede ver qué predijo cada modo: arbitraría contaminado."""
    from scripts.qa_report import build_queue

    W = [{"timestamp_start": float(i) / 2, "timestamp_end": float(i) / 2 + 3.0,
          "transcription": "eh bueno este",
          "features": {"f0_reset": 3.0, "max_delta_mouth_press": 0.4, "is_dangling": True},
          "predictions": {"audio_only": {"class": "fluent", "prob": .5},
                          "audio_text": {"class": "neutral_pause", "prob": .6},
                          "trimodal": {"class": "block", "prob": .7}}}
         for i in range(5)]
    q = build_queue(W, "https://youtu.be/x", 10)
    assert len(q) == 5

    # el campo oculto existe para el informe, pero se retira antes de construir
    # el prompt; lo que se envía al juez sale de llm_judge.build_prompt
    import scripts.llm_judge as J
    prompt = J.build_prompt("antes", "eh bueno este", "despues",
                            {"duracion_s": 1.2, "proporcion_silencio": 0.5})
    low = prompt.lower()
    for leak in ("trimodal", "audio_text", "audio_only", "prediccion", "modelo dijo"):
        assert leak not in low, f"fuga hacia el juez: {leak}"


def test_queue_is_ordered_by_evidence_strength():
    from scripts.qa_report import build_queue

    def w(i, f0, mp):
        return {"timestamp_start": float(i), "timestamp_end": float(i) + 3.0, "transcription": "x",
                "features": {"f0_reset": f0, "max_delta_mouth_press": mp, "is_dangling": False},
                "predictions": {"audio_only": {"class": "fluent", "prob": .5},
                                "audio_text": {"class": "fluent", "prob": .5},
                                "trimodal": {"class": "block", "prob": .9}}}

    q = build_queue([w(0, 0.1, 0.01), w(1, 6.0, 0.50), w(2, 1.0, 0.05)], None, 10)
    assert q[0]["start"] == 1.0, "la evidencia más fuerte va primero"
    assert q[0]["evidence_strength"] > q[-1]["evidence_strength"]


def test_only_discrepant_windows_enter_the_queue():
    from scripts.qa_report import build_queue

    same = {"timestamp_start": 0.0, "timestamp_end": 3.0, "transcription": "",
            "features": {"f0_reset": 5.0, "max_delta_mouth_press": 0.9, "is_dangling": False},
            "predictions": {m: {"class": "fluent", "prob": .9}
                            for m in ("audio_only", "audio_text", "trimodal")}}
    assert build_queue([same], None, 10) == []


def test_audio_text_agreements_keeps_every_judge_match():
    from scripts.qa_report import audio_text_agreements

    windows = [
        {"predictions": {"audio_text": {"class": "block"}}},
        {"predictions": {"audio_text": {"class": "fluent"}}},
        {"predictions": {"audio_text": {"class": "filler_word"}}},
    ]
    rows = [{"window_index": 2, "label": "filler_word"},
            {"window_index": 1, "label": "block"},
            {"window_index": 0, "label": "block"}]
    assert [x["window_index"] for x in audio_text_agreements(rows, windows)] == [0, 2]


def test_binary_confusion_collapses_fine_taxonomy_without_changing_truth():
    from scripts.analyze_baseline_sensitivity import binary_confusion

    rows = [
        {"judge": "filler_word", "prediction": "revision"},
        {"judge": "repetition", "prediction": "neutral_pause"},
        {"judge": "fluent", "prediction": "prolongation"},
        {"judge": "fluent", "prediction": "fluent"},
    ]
    positive = {"filler_word", "repetition", "revision", "prolongation", "block"}
    assert binary_confusion(rows, positive) == {"tp": 1, "fn": 1, "fp": 1, "tn": 1}


# --------------------------------------------------------------- CLI sensato
@pytest.mark.parametrize("script,args", [
    ("scripts/infer_video.py", ["--vid", "no_existe"]),
    ("scripts/qa_report.py", ["--vid", "no_existe"]),
    ("scripts/adjudicate.py", ["--vid", "no_existe"]),
])
def test_scripts_fail_with_a_message_not_a_traceback(script, args):
    r = subprocess.run([sys.executable, str(ROOT / script), *args],
                       capture_output=True, text=True, cwd=ROOT)
    assert r.returncode != 0
    assert "Traceback" not in r.stderr, f"{script} debe explicar el problema, no reventar"
    assert "Corre antes" in r.stdout or "falta" in r.stdout.lower()
