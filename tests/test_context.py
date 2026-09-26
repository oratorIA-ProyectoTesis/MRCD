"""Pruebas de los rasgos contextuales (v2): frontera prosódica y ventana ampliada.

No basta con comprobar formas de tensores. Cada prueba verifica que la medida
distingue el fenómeno que dice distinguir: si `f0_reset` no separa una frontera
de discurso de una suspensión disfluente, el rasgo no sirve aunque el vector
tenga la longitud correcta.
"""
from __future__ import annotations

import numpy as np
import torch

from core.constants import (ACOUSTIC_HOP_S, CONTEXT_S, MAX_CONTEXT_WORDS, PROSODY_DIM,
                            PROSODY_FIELDS, WINDOW_S)
from core.dataset import center_f0, linguistic_windows
from core.extractors.prosody import (boundary_features, longest_silence, prosody_windows,
                                     speaker_f0_center)
from core.models.fusion import MODES, CrossModalFusion

PIDX = {n: i for i, n in enumerate(PROSODY_FIELDS)}
HOP = ACOUSTIC_HOP_S


def _frames(segments) -> np.ndarray:
    """Construye una matriz (T,5) [log_rms, f0_st, voiced_prob, vad, df0].

    `segments` es una lista de (duracion_s, f0_st_inicio, f0_st_fin, es_voz).
    """
    rows = []
    for dur, f0a, f0b, voiced in segments:
        n = int(round(dur / HOP))
        f0 = np.linspace(f0a, f0b, n) if voiced else np.zeros(n)
        rms = np.full(n, -1.0 if voiced else -5.0)
        rows.append(np.stack([rms, f0, np.full(n, 1.0 if voiced else 0.0),
                              np.full(n, 1.0 if voiced else 0.0), np.zeros(n)], 1))
    m = np.concatenate(rows, 0).astype(np.float32)
    m[:, 4] = np.concatenate([[0.0], np.diff(m[:, 1])])
    return m


# ----------------------------------------------------------------- silencio
def test_longest_silence_finds_the_gap():
    f = _frames([(0.5, 3, 3, True), (0.8, 0, 0, False), (0.4, 3, 3, True),
                 (0.2, 0, 0, False), (1.1, 3, 3, True)])
    a, b = longest_silence(f[:, 3])
    assert abs((b - a) * HOP - 0.8) < 0.02, "debe elegir el silencio de 0.8 s, no el de 0.2 s"
    assert abs(a * HOP - 0.5) < 0.02


def test_no_silence_gives_zero_vector():
    f = _frames([(3.0, 3, 3, True)])
    v = boundary_features(f)
    assert v.shape == (PROSODY_DIM,)
    assert np.allclose(v, 0.0), "sin silencio no hay frontera que medir"


# ------------------------------------------- el rasgo central: pitch reset
def test_pitch_reset_separates_boundary_from_suspension():
    """Frontera de discurso vs. suspensión disfluente, con el mismo silencio.

    Ambas ventanas tienen un silencio de 0.8 s en la misma posición. Lo único
    que cambia es el contorno tonal. Si el rasgo no las separa, el modelo no
    tiene forma de distinguir una pausa retórica de un bloqueo.
    """
    frontera = _frames([(1.2, 6, 0, True), (0.8, 0, 0, False), (1.0, 6, 5, True)])
    suspension = _frames([(1.2, 3, 3, True), (0.8, 0, 0, False), (1.0, 3, 3, True)])

    vf = boundary_features(frontera)
    vs = boundary_features(suspension)

    # misma duración de silencio: la duración por sí sola no discrimina
    assert abs(vf[PIDX["sil_dur"]] - vs[PIDX["sil_dur"]]) < 0.02

    # la frontera cae antes del silencio; la suspensión se mantiene
    assert vf[PIDX["f0_slope_pre"]] < -1.0
    assert abs(vs[PIDX["f0_slope_pre"]]) < 0.5

    # y reinicia hacia arriba al retomar; la suspensión retoma donde quedó
    assert vf[PIDX["f0_reset"]] > 2.0, "una frontera debe mostrar reinicio tonal claro"
    assert abs(vs[PIDX["f0_reset"]]) < 0.5, "una suspensión retoma al mismo nivel"
    assert vf[PIDX["f0_reset"]] - vs[PIDX["f0_reset"]] > 2.0


def test_prosody_windows_shape_and_alignment():
    #      habla 0.0-3.5 | silencio 3.5-4.5 | habla 4.5-6.5
    f = _frames([(3.5, 3, 3, True), (1.0, 0, 0, False), (2.0, 3, 3, True)])
    starts = np.array([0.0, 3.0, 4.0])
    P = prosody_windows(f, starts)
    assert P.shape == (3, PROSODY_DIM)
    assert P[0, PIDX["sil_dur"]] == 0.0, "la ventana [0,3] es toda habla"
    assert P[1, PIDX["sil_dur"]] > 0.9, "la ventana [3,6] contiene el silencio completo"
    # el silencio arranca a 0.5 s de una ventana de 3 s que empieza en 3.0
    assert abs(P[1, PIDX["sil_pos"]] - 0.5 / WINDOW_S) < 0.05


def test_prosody_carries_no_heuristic_decision():
    """Los rasgos son descriptivos: ninguno codifica una etiqueta de la cascada."""
    joined = " ".join(PROSODY_FIELDS).lower()
    for leak in ("filler", "prolong", "block", "repetition", "revision",
                 "rhetorical", "neutral", "conf", "candidate"):
        assert leak not in joined


# ------------------------------------------------ normalización por hablante
def test_speaker_centering_only_touches_voiced_frames():
    f = _frames([(1.0, 12, 12, True), (0.5, 0, 0, False), (1.0, 12, 12, True)])
    c = speaker_f0_center(f)
    assert abs(c - 12.0) < 0.1
    out = center_f0(f, c)
    unvoiced = f[:, 3] < 0.5
    assert np.allclose(out[unvoiced, 1], 0.0), "los frames no sonoros deben seguir en 0"
    assert abs(out[~unvoiced, 1].mean()) < 0.1, "los sonoros quedan centrados en 0"


def test_centering_makes_two_speakers_comparable():
    grave = _frames([(1.0, 0, 3, True), (0.5, 0, 0, False), (1.0, 0, 3, True)])
    agudo = _frames([(1.0, 14, 17, True), (0.5, 0, 0, False), (1.0, 14, 17, True)])
    g = center_f0(grave, speaker_f0_center(grave))
    a = center_f0(agudo, speaker_f0_center(agudo))
    gv, av = g[g[:, 3] > 0.5, 1], a[a[:, 3] > 0.5, 1]
    assert abs(gv.mean() - av.mean()) < 0.2, "tras centrar, el nivel absoluto deja de separarlos"
    assert abs(gv.std() - av.std()) < 0.2, "el movimiento tonal se conserva"


# ---------------------------------------------- ventana lingüística ampliada
def _words(times_texts):
    return [{"start": s, "end": e, "text": t, "norm": t} for s, e, t in times_texts]


def test_context_window_sees_words_before_the_analysis_window():
    """El núcleo de la afirmación 'contextual': mirar más allá de los 3 s."""
    # ventana de análisis [6, 9]; ventana de contexto [1, 11]
    words = _words([(1.5, 1.9, "porque"), (2.0, 2.4, "nunca"), (2.5, 2.9, "funciona"),
                    (6.2, 6.6, "entonces"), (6.8, 7.2, "vamos")])
    wf = np.zeros((len(words), 10), np.float32)
    starts = np.array([6.0])

    _, _, m_ctx = linguistic_windows(words, wf, starts, context=True)
    _, _, m_old = linguistic_windows(words, wf, starts, max_words=16, context=False)

    assert m_old.sum() == 2, "la ventana de 3 s sólo alcanza las dos palabras de dentro"
    assert m_ctx.sum() == 5, "la ventana de contexto alcanza la cláusula previa"


def test_zone_marks_before_inside_after():
    words = _words([(2.5, 2.9, "antes"), (6.2, 6.6, "dentro"), (9.4, 9.8, "despues")])
    wf = np.zeros((len(words), 10), np.float32)
    L, _, mask = linguistic_windows(words, wf, np.array([6.0]), context=True)
    zone = L[0, :int(mask[0].sum()), 10]
    assert list(zone) == [-1.0, 0.0, 1.0]


def test_dangling_function_word_is_flagged():
    """Una pausa tras «de» rompe un constituyente: no es una frontera retórica."""
    words = _words([(1.0, 1.3, "cerca"), (1.35, 1.5, "de"), (2.4, 2.9, "casa")])
    wf = np.zeros((len(words), 10), np.float32)
    wf[1, 1] = 1.0            # «de» es palabra funcional
    wf[1, 9] = 0.90           # seguida de un hueco de 900 ms
    wf[0, 9] = 0.05
    L, _, mask = linguistic_windows(words, wf, np.array([0.5]), context=True)
    dang = L[0, :int(mask[0].sum()), 11]
    assert dang[1] == 1.0, "palabra funcional + hueco = constituyente suspendido"
    assert dang[0] == 0.0 and dang[2] == 0.0


def test_context_keeps_words_nearest_the_centre_when_overflowing():
    words = _words([(i * 0.2, i * 0.2 + 0.15, f"w{i}") for i in range(200)])
    wf = np.zeros((len(words), 10), np.float32)
    L, pos, mask = linguistic_windows(words, wf, np.array([10.0]), context=True)
    assert mask[0].sum() == MAX_CONTEXT_WORDS
    assert pos[0][mask[0]].max() < int(round(CONTEXT_S * 10))


def test_position_index_spans_the_context_not_only_the_window():
    words = _words([(5.2, 5.4, "lejos"), (12.5, 12.8, "tarde")])
    wf = np.zeros((len(words), 10), np.float32)
    _, pos, mask = linguistic_windows(words, wf, np.array([10.0]), context=True)
    p = pos[0][mask[0]]
    assert p.max() > 30, "las posiciones deben cubrir la ventana de contexto, no sólo 30 pasos"


# ----------------------------------------------------------------- el modelo
def _batch(B=4, ling_dim=12):
    return dict(
        ac=torch.randn(B, 300, 5), pros=torch.randn(B, PROSODY_DIM),
        ling=torch.randn(B, MAX_CONTEXT_WORDS, ling_dim),
        lpos=torch.randint(0, int(CONTEXT_S * 10), (B, MAX_CONTEXT_WORDS)),
        lmask=torch.ones(B, MAX_CONTEXT_WORDS, dtype=torch.bool),
        kin=torch.randn(B, 30, 12), kmask=torch.ones(B, 30, dtype=torch.bool),
        has_video=torch.ones(B, dtype=torch.bool),
    )


def test_all_modes_accept_prosody_and_context():
    b = _batch()
    for mode in MODES:
        net = CrossModalFusion(8, mode=mode).eval()
        with torch.no_grad():
            out = net(b)
        assert out.shape == (4, 8)
        assert torch.isfinite(out).all()


def test_prosody_is_present_in_audio_only():
    """La prosodia se deriva del audio: excluirla de audio_only inflaría la
    ventaja aparente de los modos multimodales."""
    net = CrossModalFusion(8, mode="audio_only").eval()
    assert net.pros is not None
    b = _batch()
    with torch.no_grad():
        base = net(b)
        moved = net({**b, "pros": b["pros"] + 5.0})
    assert not torch.allclose(base, moved), "audio_only debe reaccionar a la prosodia"


def test_prosody_changes_predictions_in_every_mode():
    b = _batch()
    for mode in MODES:
        net = CrossModalFusion(8, mode=mode).eval()
        with torch.no_grad():
            a = net(b)
            c = net({**b, "pros": torch.zeros_like(b["pros"])})
        assert not torch.allclose(a, c), f"{mode} ignora la rama prosódica"


def test_modality_weights_report_prosody():
    net = CrossModalFusion(8, mode="trimodal").eval()
    with torch.no_grad():
        _, w = net(_batch(), return_attn=True)
    assert set(w) == {"acoustic", "syntactic", "kinesic", "prosodic"}
    total = sum(v.sum().item() for v in w.values())
    assert abs(total / 4 - 1.0) < 0.05, "la masa de atención por ventana debe sumar ~1"


# --------------- el rasgo no debe morir cuando el habla arranca con sorda
def test_reset_survives_an_unvoiced_onset_after_the_pause():
    """El fallo medido en audio real: tras una pausa el habla suele arrancar con
    consonante sorda (/p/, /t/, /k/, /s/), que no produce F0. Exigir frames
    sonoros dentro de una franja fija de 300 ms dejaba `f0_reset` en cero en la
    mitad de las ventanas con silencio."""
    f = _frames([(1.0, 6, 0, True),      # cae el tono
                 (0.8, 0, 0, False),     # pausa
                 (0.35, 0, 0, False),    # arranque sordo: sin F0
                 (0.85, 6, 6, True)])    # y reinicia arriba
    v = boundary_features(f)
    assert v[PIDX["reset_valid"]] == 1.0, "debe poder medirse pese al arranque sordo"
    assert v[PIDX["f0_reset"]] > 2.0, "el reinicio tonal sigue estando ahí"


def test_reset_valid_separates_no_reset_from_not_measurable():
    """Un cero por «no hubo reinicio» y uno por «no se pudo medir» son lo
    contrario; sin bandera el modelo no puede distinguirlos."""
    medible = _frames([(1.0, 3, 3, True), (0.8, 0, 0, False), (1.0, 3, 3, True)])
    v1 = boundary_features(medible)
    assert v1[PIDX["reset_valid"]] == 1.0
    assert abs(v1[PIDX["f0_reset"]]) < 0.5, "suspensión real: reinicio nulo, pero medido"

    sin_voz = _frames([(0.2, 0, 0, False), (2.0, 0, 0, False), (0.8, 0, 0, False)])
    v2 = boundary_features(sin_voz)
    assert v2[PIDX["reset_valid"]] == 0.0, "sin frames sonoros no hay medida"
    assert v2[PIDX["f0_reset"]] == 0.0


def test_voiced_search_respects_its_distance_limit():
    from core.extractors.prosody import _voiced_near

    f0 = np.zeros(300, np.float32)
    f0[290:295] = 5.0                       # voz a 1.9 s del borde
    assert _voiced_near(f0, 100, back=False, max_s=0.8).size == 0, "demasiado lejos"
    assert _voiced_near(f0, 250, back=False, max_s=0.8).size > 0, "dentro del alcance"
