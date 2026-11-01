"""Construcción de tensores por ventana (3 s / 500 ms) a partir de las señales extraídas."""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from mrcd.constants import (KINESIC_SPEAKER_NORM_MIN_FRAMES,
    ACOUSTIC_HOP_S,
    BACKGROUND,
    CONTEXT_PRE_S,
    CONTEXT_S,
    KINESIC_DIM,
    KINESIC_STEPS,
    LABEL2ID,
    MAX_CONTEXT_WORDS,
    STRIDE_S,
    TAXONOMY,
    VIDEO_FPS,
    WINDOW_S,
)
from mrcd.extractors.prosody import prosody_windows, speaker_f0_center
from mrcd.sync.buffer import SyncBuffer

AC_STEPS = int(round(WINDOW_S / ACOUSTIC_HOP_S))  # 300
MAX_WORDS = 16
CONTEXT_STEPS = int(
    round(CONTEXT_S * VIDEO_FPS)
)  # 100 posiciones del ramal lingüístico


def kinesic_windows(k_times, k_vectors, starts_s, av_offset_ms: float = 0.0):
    """Usa el SyncBuffer (interpolación lineal) para producir la rejilla 30x12 de cada ventana."""
    buf = SyncBuffer(av_offset_ms=av_offset_ms, retain_s=1e9)
    for t, v in zip(k_times, k_vectors):
        if not np.all(np.isnan(v)):
            buf.push_kinesic(v, t * 1000)
    kin = np.zeros((len(starts_s), KINESIC_STEPS, KINESIC_DIM), np.float32)
    kmask = np.zeros((len(starts_s), KINESIC_STEPS), bool)
    has_video = np.zeros(len(starts_s), bool)
    for i, t0 in enumerate(starts_s):
        g, m, real, _ = buf._kinesic_grid(t0)
        kin[i], kmask[i], has_video[i] = g, m, real >= 24
    return kin, kmask, has_video


def speaker_kinesic_norm(
    kin: np.ndarray,
    kmask: np.ndarray,
    min_frames: int = KINESIC_SPEAKER_NORM_MIN_FRAMES,
) -> tuple[np.ndarray, bool]:
    """z-score por CANAL sobre todas las ventanas de UN hablante. -> (kin_norm, aplicado)

    Simétrico a `center_f0` para la rama acústica: lo que importa es el movimiento
    de la cara, no dónde está el reposo de cada persona. Sin esto, la rama cinésica
    gasta capacidad en identidad facial y, con folds disjuntos por hablante, esa
    capacidad es ruido puro en evaluación.

    ◆ El orden importa: se normaliza la SERIE (N, 30, 12) antes de cualquier
    agregación. Normalizar rasgos ya agregados anula la varianza entre hablantes
    por construcción y produce métricas degeneradas que parecen buenas.

    Solo se usan los pasos con `kmask` (frames reales); los interpolados o ausentes
    no deben mover la media del hablante. Si el hablante no llega a `min_frames`
    reales, se devuelve sin tocar y `aplicado=False`: normalizar con estadísticas de
    cuatro frames introduce más ruido del que quita.
    """
    m = kmask.astype(bool)
    if m.sum() < min_frames:
        return kin, False
    flat = kin[m]  # (n_frames_reales, 12)
    mu = flat.mean(0)
    sd = flat.std(0)
    if not np.all(np.isfinite(mu)) or np.any(sd < 1e-6):
        return kin, False  # canal constante: dividir amplificaría ruido
    return ((kin - mu) / sd).astype(np.float32), True


def center_f0(frame_matrix: np.ndarray, center_st: float) -> np.ndarray:
    """Centra el nivel tonal por hablante sin tocar los frames no sonoros.

    La columna 1 son semitonos relativos a 100 Hz, con 0.0 reservado para
    «no sonoro». Restar la mediana del hablante sólo en los frames sonoros
    evita que el modelo gaste capacidad en distinguir voces graves de agudas
    cuando lo que importa es el movimiento del tono, no su altura absoluta.
    """
    if center_st == 0.0 or frame_matrix.shape[1] < 2:
        return frame_matrix
    out = frame_matrix.copy()
    voiced = out[:, 1] != 0.0
    out[voiced, 1] -= center_st
    return out


def acoustic_windows(frame_matrix: np.ndarray, starts_s) -> np.ndarray:
    out = np.zeros((len(starts_s), AC_STEPS, frame_matrix.shape[1]), np.float32)
    for i, t0 in enumerate(starts_s):
        a = int(round(t0 / ACOUSTIC_HOP_S))
        seg = frame_matrix[a : a + AC_STEPS]
        out[i, : len(seg)] = seg
    return out


def linguistic_windows(
    words: list[dict],
    word_feats: np.ndarray,
    starts_s,
    max_words: int = MAX_CONTEXT_WORDS,
    context: bool = True,
):
    """Ramal lingüístico sobre una ventana de contexto más ancha que la de análisis.

    Con `context=True` la ventana abarca `CONTEXT_S` segundos (por defecto 10 s,
    centrados de modo que se vean ~5 s previos), no sólo los 3 s de análisis.
    Decidir si una pausa cierra una unidad de discurso —una pausa retórica— o la
    interrumpe exige ver si la cláusula anterior se completó, y eso suele quedar
    fuera de una ventana de 3 s.

    A los rasgos por palabra se añaden dos columnas dependientes de la ventana:

    - `zone`   : -1 antes de la ventana de análisis, 0 dentro, +1 después.
    - `dangling`: palabra funcional seguida de un hueco apreciable. Una pausa
      tras «de», «que» o «y» rompe un constituyente y apunta a suspensión, no a
      frontera retórica.

    `context=False` reproduce el comportamiento anterior (3 s, 16 palabras) y se
    conserva para comparar arquitecturas.
    """
    base_F = word_feats.shape[1] if len(word_feats) else 10
    F = base_F + 2
    ling = np.zeros((len(starts_s), max_words, F), np.float32)
    lpos = np.zeros((len(starts_s), max_words), np.int64)
    lmask = np.zeros((len(starts_s), max_words), bool)
    if not len(words):
        return ling, lpos, lmask

    centers = np.array([(w["start"] + w["end"]) / 2 for w in words])
    span = CONTEXT_S if context else WINDOW_S
    pre = CONTEXT_PRE_S if context else 0.0
    n_pos = CONTEXT_STEPS if context else KINESIC_STEPS
    # hueco posterior a cada palabra (columna 9 de word_features); si el
    # extractor cambió de forma, se recalcula desde los tiempos.
    gap_after = (
        word_feats[:, 9]
        if base_F > 9
        else np.array(
            [
                (words[i + 1]["start"] - words[i]["end"]) if i + 1 < len(words) else 0.0
                for i in range(len(words))
            ],
            np.float32,
        )
    )
    is_func = word_feats[:, 1] if base_F > 1 else np.zeros(len(words), np.float32)
    dangling = (is_func > 0.5) & (gap_after > 0.30)

    for i, t0 in enumerate(starts_s):
        lo, hi = t0 - pre, t0 - pre + span
        idx = np.where((centers >= lo) & (centers < hi))[0]
        if len(idx) > max_words:  # conserva las más cercanas al centro
            mid = t0 + WINDOW_S / 2
            idx = idx[np.argsort(np.abs(centers[idx] - mid))[:max_words]]
            idx.sort()
        n = len(idx)
        if not n:
            continue
        c = centers[idx]
        zone = np.where(c < t0, -1.0, np.where(c >= t0 + WINDOW_S, 1.0, 0.0))
        ling[i, :n, :base_F] = word_feats[idx]
        ling[i, :n, base_F] = zone
        ling[i, :n, base_F + 1] = dangling[idx].astype(np.float32)
        lpos[i, :n] = np.clip(((c - lo) * VIDEO_FPS).astype(int), 0, n_pos - 1)
        lmask[i, :n] = True
    return ling, lpos, lmask


def window_starts(duration_s: float) -> np.ndarray:
    if duration_s < WINDOW_S:
        return np.zeros(0)
    n = int(np.floor((duration_s - WINDOW_S) / STRIDE_S)) + 1
    return np.round(np.arange(n) * STRIDE_S, 6)


def labels_from_events(
    events: list[dict], starts_s, min_conf: float = 0.5
) -> tuple[np.ndarray, np.ndarray]:
    """Etiqueta primaria (int) + multi-hot (7) por ventana."""
    y = np.full(len(starts_s), LABEL2ID[BACKGROUND], np.int64)
    multi = np.zeros((len(starts_s), len(TAXONOMY)), np.int8)
    for i, a in enumerate(starts_s):
        b, best = a + WINDOW_S, 0.0
        for e in events:
            if (
                e.get("confidence", 1.0) < min_conf
                or e["category"] not in TAXONOMY
                or e.get("source", "auto") == "auto_low_conf"
            ):
                continue
            s, t = e["start_ms"] / 1000, e["end_ms"] / 1000
            ov = min(b, t) - max(a, s)
            if ov <= 0 or ov < min(0.30, 0.5 * (t - s)):
                continue
            multi[i, TAXONOMY.index(e["category"])] = 1
            sc = ov * e.get("confidence", 1.0)
            if sc > best:
                best, y[i] = sc, LABEL2ID[e["category"]]
    return y, multi


def read_human_events(
    eaf_path: str | Path, tier: str = "human_disfluency", *, strict: bool = False
) -> list[dict]:
    """Lee las anotaciones humanas (tier con vocabulario controlado) de un .eaf."""
    root = ET.parse(eaf_path).getroot()
    slots = {
        s.get("TIME_SLOT_ID"): int(s.get("TIME_VALUE"))
        for s in root.iter("TIME_SLOT")
        if s.get("TIME_VALUE")
    }
    out = []
    found = False
    for t in root.iter("TIER"):
        if t.get("TIER_ID") != tier:
            continue
        found = True
        if strict and list(t.iter("REF_ANNOTATION")):
            raise ValueError(f"El tier {tier} requiere intervalos temporales alineables")
        for a in t.iter("ALIGNABLE_ANNOTATION"):
            val = (a.findtext("ANNOTATION_VALUE") or "").strip()
            if strict and val not in TAXONOMY:
                raise ValueError(f"Etiqueta humana desconocida o vacía en {eaf_path}: {val!r}")
            if val in TAXONOMY:
                start = slots.get(a.get("TIME_SLOT_REF1"))
                end = slots.get(a.get("TIME_SLOT_REF2"))
                if strict and (start is None or end is None or start < 0 or end <= start):
                    raise ValueError(f"Intervalo humano inválido en {eaf_path}")
                out.append(
                    dict(
                        category=val,
                        start_ms=slots[a.get("TIME_SLOT_REF1")],
                        end_ms=slots[a.get("TIME_SLOT_REF2")],
                        confidence=1.0,
                        source="human",
                    )
                )
    if strict and not found:
        raise ValueError(f"Falta el tier humano {tier} en {eaf_path}")
    return out


def gold_windows(events: list[dict], starts) -> tuple[np.ndarray, np.ndarray]:
    """Una ventana por juicio del juez: la de centro más cercano al centro del intervalo juzgado.
    Devuelve (índices de ventana, etiqueta entera). Las ventanas no juzgadas quedan fuera del test."""
    idx, lab = [], []
    if len(starts) == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    centers = np.asarray(starts) + WINDOW_S / 2
    for e in events:
        if e["category"] not in LABEL2ID:
            continue
        c = (e["start_ms"] + e["end_ms"]) / 2000
        i = int(np.argmin(np.abs(centers - c)))
        idx.append(i)
        lab.append(LABEL2ID[e["category"]])
    return np.asarray(idx, np.int64), np.asarray(lab, np.int64)


def build_recording_windows(
    rec: dict, root: Path, label_source: str = "auto", av_offset_ms: float = 0.0
) -> dict:
    """Construye ventanas sin convertir cobertura desconocida en negativos.

    Para ``label_source='human'``, el manifiesto debe declarar por grabación
    ``human_annotation={"coverage": "complete", "provenance": "referencia"}``.
    ``complete`` declara revisión de toda la duración y de toda TAXONOMY;
    ``provenance`` identifica la fuente o registro de esa revisión. Un tier
    humano vacío es válido solo con esa declaración. No se modifica el
    ``label_source`` original del manifiesto ni se completa cobertura ausente.
    """
    human_events = None
    if label_source == "human":
        annotation = rec.get("human_annotation")
        if (
            not isinstance(annotation, dict)
            or annotation.get("coverage") != "complete"
            or not isinstance(annotation.get("provenance"), str)
            or not annotation["provenance"].strip()
        ):
            raise ValueError(
                f"Grabación {rec.get('recording_id', '<sin identificador>')}: "
                "label_source='human' requiere human_annotation con coverage='complete' "
                "y provenance no vacío; cobertura ausente, desconocida o parcial "
                "no permite generar ventanas negativas."
            )
        if not rec.get("eaf") or not (root / rec["eaf"]).is_file():
            raise ValueError("La cobertura humana declarada requiere un archivo EAF existente")
        human_events = read_human_events(root / rec["eaf"], strict=True)
        if any(e["end_ms"] > rec["duration_s"] * 1000 for e in human_events):
            raise ValueError("Un intervalo humano excede la duración declarada de la grabación")
    feats = np.load(root / rec["features_npz"], allow_pickle=False)
    words = json.loads((root / rec["words_json"]).read_text(encoding="utf-8"))
    starts = window_starts(rec["duration_s"])
    frames = feats["ac_frames"]
    f0_center = speaker_f0_center(frames)  # normalización tonal por hablante
    pros = prosody_windows(frames, starts, f0_center)  # rasgos de frontera prosódica
    ac = acoustic_windows(center_f0(frames, f0_center), starts)
    ling, lpos, lmask = linguistic_windows(words, feats["word_feats"], starts)
    kin, kmask, hv = kinesic_windows(
        feats["k_times"], feats["k_vectors"], starts, av_offset_ms
    )
    keep = None
    if label_source == "human":
        y, multi = labels_from_events(human_events, starts)
    elif label_source == "gold":
        gp = root / rec["events_json"].replace(".events.json", ".gold.json")
        events = json.loads(gp.read_text(encoding="utf-8")) if gp.exists() else []
        keep, ylab = gold_windows(events, starts)
        y = np.full(len(starts), LABEL2ID[BACKGROUND], np.int64)
        multi = np.zeros((len(starts), len(TAXONOMY)), np.int8)
        y[keep] = ylab
    else:
        events = json.loads((root / rec["events_json"]).read_text(encoding="utf-8"))
        y, multi = labels_from_events(events, starts)
    n = len(starts)
    out = dict(
        ac=ac,
        pros=pros,
        ling=ling,
        lpos=lpos,
        lmask=lmask,
        kin=kin,
        kmask=kmask,
        has_video=hv,
        y=y,
        multi=multi,
        start_ms=(starts * 1000).astype(np.int64),
        group=np.array([rec["speaker_id"]] * n),
        rec=np.array([rec["recording_id"]] * n),
    )
    if keep is not None:  # solo las ventanas efectivamente juzgadas entran al test
        uniq = np.unique(keep)
        out = {k: v[uniq] for k, v in out.items()}
    return out
