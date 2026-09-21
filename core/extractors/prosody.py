"""Rasgos de frontera prosódica alrededor del silencio de cada ventana.

Motivación
----------
Distinguir una pausa retórica de una pausa neutra o de un bloqueo no es un
problema de duración del silencio: los tres pueden durar lo mismo. El marcador
que los separa es la forma del contorno tonal en la frontera.

- Si F0 **cae** antes del silencio y **reinicia hacia arriba** al retomar, el
  hablante cerró una unidad de discurso: la pausa es una frontera (retórica o
  neutra, según intención y duración).
- Si F0 queda **suspendido** al mismo nivel y retoma donde quedó, el hablante
  no terminó: la pausa interrumpe un constituyente (bloqueo, duda, revisión).

El `AcousticEncoder` promedia 100 ms por paso (`AdaptiveAvgPool1d(30)`), lo que
borra justamente esa forma. Este módulo la mide de forma explícita y la entrega
como un vector corto y denso, en paralelo a la secuencia acústica.

Todas las medidas son descriptivas (pendientes, rangos, diferencias). No
codifican ninguna decisión de la cascada heurística, así que pueden usarse con
etiquetas de cualquier origen sin inducir circularidad.
"""
from __future__ import annotations

import numpy as np

from core.constants import (ACOUSTIC_HOP_S, PROSODY_DIM, PROSODY_POST_S, PROSODY_PRE_S,
                            WINDOW_S)


def _slope(y: np.ndarray, hop_s: float) -> float:
    """Pendiente por segundo de una serie regular, robusta a series cortas."""
    n = len(y)
    if n < 3:
        return 0.0
    x = np.arange(n, dtype=np.float64) * hop_s
    xm, ym = x.mean(), y.mean()
    den = float(((x - xm) ** 2).sum())
    if den <= 1e-12:
        return 0.0
    return float(((x - xm) * (y - ym)).sum() / den)


def longest_silence(vad: np.ndarray, hop_s: float = ACOUSTIC_HOP_S) -> tuple[int, int]:
    """Índices [a, b) del tramo no-voz más largo. (0, 0) si no hay ninguno."""
    if vad.size == 0:
        return 0, 0
    sil = ~(vad > 0.5)
    best_a = best_b = 0
    a = None
    for i, s in enumerate(sil):
        if s and a is None:
            a = i
        elif not s and a is not None:
            if i - a > best_b - best_a:
                best_a, best_b = a, i
            a = None
    if a is not None and len(sil) - a > best_b - best_a:
        best_a, best_b = a, len(sil)
    return best_a, best_b


def _voiced_near(f0: np.ndarray, edge: int, back: bool, want: int = 3,
                 hop_s: float = ACOUSTIC_HOP_S, max_s: float = 0.8) -> np.ndarray:
    """Frames sonoros más cercanos al borde del silencio, buscando hacia fuera.

    Por qué no basta una franja fija: tras una pausa el habla arranca a menudo
    con consonante sorda (/p/, /t/, /k/, /s/), que no produce F0, y antes de la
    pausa la voz suele apagarse en un sonido sordo. Exigir tres frames sonoros
    dentro de los 300 ms adyacentes dejaba el rasgo sin calcular en la mitad de
    los casos reales —y un cero por falta de datos es indistinguible de un cero
    por ausencia de reinicio tonal, que es justo lo contrario—.

    Se buscan los `want` frames sonoros más próximos al borde, hasta `max_s`
    segundos de distancia. `back=True` mira hacia atrás desde `edge`.
    """
    lim = max(int(round(max_s / hop_s)), want)
    idx = range(edge - 1, max(edge - lim, 0) - 1, -1) if back else \
        range(edge, min(edge + lim, len(f0)))
    got = [f0[i] for i in idx if 0 <= i < len(f0) and f0[i] != 0.0]
    return np.asarray(got[:max(want * 5, 15)], dtype=np.float64)


def boundary_features(frames: np.ndarray, f0_center_st: float = 0.0,
                      hop_s: float = ACOUSTIC_HOP_S, window_s: float = WINDOW_S) -> np.ndarray:
    """Vector de `PROSODY_DIM` rasgos para una ventana acústica.

    `frames` es la matriz (T, 5) de `AcousticTrack.frame_matrix()`:
    [log_rms, f0_st, voiced_prob, vad, df0].

    `f0_center_st` es la mediana de F0 sonoro del hablante en semitonos; se
    resta para que el nivel tonal sea comparable entre hablantes con rangos
    distintos (una voz grave y una aguda no deben verse como clases distintas).
    """
    out = np.zeros(PROSODY_DIM, np.float32)
    if frames.ndim != 2 or frames.shape[0] < 3 or frames.shape[1] < 4:
        return out

    log_rms, f0_st, vad = frames[:, 0], frames[:, 1], frames[:, 3]
    a, b = longest_silence(vad, hop_s)
    sil_dur = (b - a) * hop_s
    if sil_dur <= 0:
        return out                     # ventana sin silencio: todo cero

    out[0] = sil_dur
    out[1] = np.clip((a * hop_s) / max(window_s, 1e-6), 0.0, 1.0)

    npre = max(int(round(PROSODY_PRE_S / hop_s)), 3)
    npost = max(int(round(PROSODY_POST_S / hop_s)), 3)

    # --- tramo previo al silencio: ¿cae el tono o queda suspendido?
    # Los frames sonoros se buscan hacia fuera desde el borde, no en una franja
    # fija: ver `_voiced_near`. `v_pre` queda en orden cronológico para que la
    # pendiente conserve el signo.
    v_pre = _voiced_near(f0_st, a, back=True, hop_s=hop_s)[::-1]
    if v_pre.size >= 3:
        out[2] = _slope(v_pre, hop_s)
        out[3] = float(v_pre.max() - v_pre.min())
        out[4] = float(v_pre.mean() - f0_center_st)
    rms_pre = log_rms[max(a - npre, 0):a]
    if rms_pre.size >= 3:
        out[6] = _slope(rms_pre, hop_s)

    # --- reinicio de tono al retomar: el rasgo que separa frontera de suspensión
    v_post = _voiced_near(f0_st, b, back=False, hop_s=hop_s)
    if v_pre.size >= 3 and v_post.size >= 3:
        out[5] = float(v_post.mean() - v_pre.mean())
        out[8] = 1.0                    # medida válida, no un cero por falta de datos

    # --- alargamiento final: energía sostenida justo antes del corte
    # (proxy barato: cuánta energía media conserva el último tramo sonoro
    #  frente al resto de la parte hablada de la ventana)
    speech = log_rms[vad > 0.5]
    if speech.size >= 5 and rms_pre.size >= 3:
        base = float(np.median(speech))
        out[7] = float(rms_pre.mean() - base)

    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def speaker_f0_center(frames: np.ndarray) -> float:
    """Mediana de F0 sonoro (semitonos) de una grabación completa.

    Se usa para centrar el nivel tonal por hablante. Devuelve 0.0 si la
    grabación no tiene tramos sonoros suficientes.
    """
    if frames.ndim != 2 or frames.shape[1] < 2:
        return 0.0
    f0 = frames[:, 1]
    v = f0[f0 != 0.0]
    return float(np.median(v)) if v.size >= 10 else 0.0


def prosody_windows(frames: np.ndarray, starts_s, f0_center_st: float = 0.0,
                    hop_s: float = ACOUSTIC_HOP_S) -> np.ndarray:
    """Matriz (N, PROSODY_DIM) con los rasgos de frontera de cada ventana."""
    steps = int(round(WINDOW_S / hop_s))
    out = np.zeros((len(starts_s), PROSODY_DIM), np.float32)
    for i, t0 in enumerate(starts_s):
        a = int(round(t0 / hop_s))
        out[i] = boundary_features(frames[a:a + steps], f0_center_st, hop_s)
    return out
