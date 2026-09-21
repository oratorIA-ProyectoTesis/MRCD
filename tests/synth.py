"""Generador de escenas sintéticas con verdad conocida (solo para pruebas del pipeline)."""
from __future__ import annotations

import numpy as np

from core.constants import KIDX, KINESIC_DIM, SAMPLE_RATE
from core.extractors.linguistic import Word, normalize

SR = SAMPLE_RATE


def voiced(dur, f0_start, f0_end=None, sr=SR, amp=0.3, seed=0):
    """Segmento 'vocálico' armónico con F0 que va de f0_start a f0_end."""
    rng = np.random.default_rng(seed)
    n = int(dur * sr)
    f0 = np.linspace(f0_start, f0_end if f0_end is not None else f0_start, n)
    ph = 2 * np.pi * np.cumsum(f0) / sr
    x = sum((0.6 / k) * np.sin(k * ph) for k in range(1, 6))
    env = np.minimum(1, np.minimum(np.arange(n), n - np.arange(n)) / (0.02 * sr))
    return (amp * x * env + 0.002 * rng.standard_normal(n)).astype(np.float32)


def silence(dur, sr=SR, seed=1):
    return (0.0005 * np.random.default_rng(seed).standard_normal(int(dur * sr))).astype(np.float32)


def scene():
    """Escena guionizada: devuelve audio, palabras (ASR simulado), track cinésico y eventos esperados."""
    plan = [  # (texto, dur, f0_ini, f0_fin) ; texto None = silencio
        ("Hoy", 0.35, 140, 150), ("quiero", 0.40, 150, 145), ("hablar", 0.40, 145, 130), ("del", 0.25, 130, 125),
        ("tema.", 0.45, 125, 95),                     # frontera con tono descendente
        (None, 1.20, 0, 0),                           # -> rhetorical_pause (cara relajada)
        ("Eh,", 0.45, 120, 120),                      # -> filler_word
        ("pero", 0.30, 135, 140), ("pero", 0.30, 135, 140),   # -> repetition
        ("la", 0.20, 140, 138),
        (None, 0.90, 0, 0),                           # -> block (intra-constituyente + tensión perioral)
        ("casa", 0.40, 138, 130), ("es", 0.20, 130, 128),
        ("muuuy", 0.80, 150, 150),                    # -> prolongation (F0 plano 800 ms)
        ("grande,", 0.45, 150, 120), ("digo,", 0.35, 125, 120), ("enorme", 0.45, 130, 110),  # -> revision
        ("y", 0.15, 120, 118),
        (None, 0.70, 0, 0),                           # -> pausa tras función SIN tensión -> neutral_pause
        ("final", 0.40, 118, 100), ("aquí.", 0.40, 100, 85),
    ]
    audio, words, t = [silence(0.5)], [], 0.5
    for i, (txt, dur, a, b) in enumerate(plan):
        if txt is None:
            audio.append(silence(dur, seed=i))
        else:
            audio.append(voiced(dur, a, b, seed=i))
            words.append(Word(txt, normalize(txt), round(t, 3), round(t + dur, 3), 0.95,
                              txt[-1] if txt[-1] in ".,?" else ""))
        t += dur
    audio.append(silence(1.0)); t += 1.0
    audio = np.concatenate(audio)
    total = len(audio) / SR

    # track cinésico a 10 fps
    kt = np.arange(0, total, 0.1)
    kv = np.zeros((len(kt), KINESIC_DIM), np.float32)
    kv[:, KIDX["mouth_press_left"]] = kv[:, KIDX["mouth_press_right"]] = 0.08
    kv[:, KIDX["jaw_open"]] = 0.3
    kv[:, 0:3] = np.random.default_rng(3).normal(0, 0.5, (len(kt), 3))
    # tensión durante la pausa de bloqueo
    blk = [w for w in words if w.norm == "la"][0].end
    m = (kt >= blk) & (kt <= blk + 0.9)
    kv[m, KIDX["mouth_press_left"]] = kv[m, KIDX["mouth_press_right"]] = 0.55
    kv[m, KIDX["brow_down_avg"]] = 0.5
    det = np.ones(len(kt), bool)

    class Track:  # duck-typing de KinesicTrack
        times, vectors, detected = kt, kv, det
        detection_rate = 1.0

    expected = {"rhetorical_pause", "filler_word", "repetition", "block", "prolongation", "revision", "neutral_pause"}
    return audio, words, Track, expected


def fricative(dur, sr=SR, amp=0.08, seed=5):
    """/s/ sostenida: ruido filtrado > 3.5 kHz, envolvente estable."""
    from scipy.signal import butter, sosfilt
    n = int(dur * sr)
    x = np.random.default_rng(seed).standard_normal(n)
    x = sosfilt(butter(6, 3500, "highpass", fs=sr, output="sos"), x)
    env = np.minimum(1, np.minimum(np.arange(n), n - np.arange(n)) / (0.02 * sr))
    return (amp * x / (np.abs(x).max() + 1e-9) * env).astype(np.float32)


def build(plan, tense_after=None, relaxed=True):
    """plan: lista de (texto|None, dur, f0a, f0b) o ('<fric>', dur) dentro de palabra.
    tense_after: texto de palabra tras la cual la pausa lleva tensión perioral."""
    audio, words, t = [silence(0.5)], [], 0.5
    tense_spans = []
    for i, item in enumerate(plan):
        txt, dur = item[0], item[1]
        if txt is None:
            if words and tense_after and words[-1].text == tense_after:
                tense_spans.append((t, t + dur))
            audio.append(silence(dur, seed=i))
        elif isinstance(item[2], str):          # ("sssí", dur, "fric", vowel_dur)
            fr, vd = fricative(dur, seed=i), voiced(item[3], 140, 138, seed=i)
            audio.append(np.concatenate([fr, vd]))
            words.append(Word(txt, normalize(txt), round(t, 3), round(t + dur + item[3], 3), 0.95, txt[-1] if txt[-1] in ".,?" else ""))
            t += item[3]
        else:
            audio.append(voiced(dur, item[2], item[3], seed=i))
            words.append(Word(txt, normalize(txt), round(t, 3), round(t + dur, 3), 0.95, txt[-1] if txt[-1] in ".,?" else ""))
        t += dur
    audio.append(silence(1.0))
    audio = np.concatenate(audio)
    kt = np.arange(0, len(audio) / SR, 0.1)
    kv = np.zeros((len(kt), KINESIC_DIM), np.float32)
    base_mp = 0.08 if relaxed else 0.3
    kv[:, KIDX["mouth_press_left"]] = kv[:, KIDX["mouth_press_right"]] = base_mp
    for a, b in tense_spans:
        m = (kt >= a) & (kt <= b)
        kv[m, KIDX["mouth_press_left"]] = kv[m, KIDX["mouth_press_right"]] = 0.55
        kv[m, KIDX["brow_down_avg"]] = 0.5

    class Track:
        times, vectors, detected = kt, kv, np.ones(len(kt), bool)
        detection_rate = 1.0
    return audio, words, Track
