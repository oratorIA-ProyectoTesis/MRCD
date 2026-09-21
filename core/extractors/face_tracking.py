"""Seguimiento de identidad facial para vídeo con más de una persona en cuadro.

El problema
-----------
`result_to_vector` toma siempre `result.face_landmarks[0]`. MediaPipe **no
garantiza que el orden de las caras sea estable entre frames**: en una entrevista
con dos personas, el índice 0 puede ser el entrevistado en un frame y el
entrevistador en el siguiente. El vector 12-D resultante mezcla dos rostros y
cualquier rasgo derivado de él (tensión perioral, estabilidad de mirada) queda
contaminado sin que nada lo señale.

La solución
-----------
Detectar todas las caras por frame, asociarlas entre frames por solapamiento de
caja (IoU) para formar pistas con identidad propia, y quedarse con la pista
dominante: la que combina mayor área en pantalla y mayor persistencia. En una
entrevista grabada con plano principal sobre el entrevistado, esa es la suya.

Limitación conocida: si el montaje alterna planos cerrados de cada persona
durante tiempos comparables, «mayor área × persistencia» puede elegir al
entrevistador. `track_report()` expone las puntuaciones de cada pista para que
esa decisión sea auditable en lugar de silenciosa.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from core.constants import KINESIC_DIM
from core.extractors.kinesic import BLENDSHAPE_MAP, rotation_to_euler_deg

IOU_MATCH = 0.30          # solapamiento mínimo para considerar la misma cara
MAX_GAP_FRAMES = 5        # frames que una pista sobrevive sin detección


def bbox_iou(a: np.ndarray, b: np.ndarray) -> float:
    """IoU de dos cajas [x0, y0, x1, y1] en coordenadas normalizadas."""
    if a is None or b is None or np.isnan(a).any() or np.isnan(b).any():
        return 0.0
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    inter = (x1 - x0) * (y1 - y0)
    area_a = max(a[2] - a[0], 0) * max(a[3] - a[1], 0)
    area_b = max(b[2] - b[0], 0) * max(b[3] - b[1], 0)
    den = area_a + area_b - inter
    return float(inter / den) if den > 0 else 0.0


def bbox_area(b: np.ndarray) -> float:
    if b is None or np.isnan(b).any():
        return 0.0
    return float(max(b[2] - b[0], 0) * max(b[3] - b[1], 0))


def result_to_faces(result) -> list[tuple[np.ndarray, np.ndarray]]:
    """Todas las caras del frame como (vector 12-D, bbox), no sólo la primera."""
    out: list[tuple[np.ndarray, np.ndarray]] = []
    lms = getattr(result, "face_landmarks", None) or []
    mats = getattr(result, "facial_transformation_matrixes", None) or []
    blends = getattr(result, "face_blendshapes", None) or []
    for i, lm in enumerate(lms):
        vec = np.full(KINESIC_DIM, np.nan, np.float32)
        if i < len(mats):
            M = np.asarray(mats[i])
            vec[0:3] = rotation_to_euler_deg(M[:3, :3])
        if i < len(blends):
            bs = {c.category_name: c.score for c in blends[i]}
            for idx, names in BLENDSHAPE_MAP.items():
                vals = [bs[n] for n in names if n in bs]
                if vals:
                    vec[idx] = float(np.mean(vals))
        xs = np.array([p.x for p in lm]); ys = np.array([p.y for p in lm])
        out.append((vec, np.array([xs.min(), ys.min(), xs.max(), ys.max()], np.float32)))
    return out


@dataclass
class Track:
    """Una identidad facial seguida a lo largo del vídeo."""
    tid: int
    frames: list[int] = field(default_factory=list)
    times: list[float] = field(default_factory=list)
    vecs: list[np.ndarray] = field(default_factory=list)
    boxes: list[np.ndarray] = field(default_factory=list)
    last_frame: int = -1

    @property
    def last_box(self) -> np.ndarray:
        return self.boxes[-1]

    def score(self) -> float:
        """Persistencia x área típica. Premia al rostro grande y constante."""
        if not self.boxes:
            return 0.0
        areas = [bbox_area(b) for b in self.boxes]
        return float(len(self.frames) * np.median(areas))

    def add(self, fi: int, t: float, vec: np.ndarray, box: np.ndarray) -> None:
        self.frames.append(fi); self.times.append(t)
        self.vecs.append(vec); self.boxes.append(box); self.last_frame = fi


class FaceTracker:
    """Asociación voraz por IoU entre frames consecutivos."""

    def __init__(self, iou_match: float = IOU_MATCH, max_gap: int = MAX_GAP_FRAMES):
        self.iou_match, self.max_gap = iou_match, max_gap
        self.tracks: list[Track] = []
        self._next = 0

    def update(self, fi: int, t: float, faces: list[tuple[np.ndarray, np.ndarray]]) -> None:
        active = [tr for tr in self.tracks if fi - tr.last_frame <= self.max_gap]
        # caras grandes primero: en un plano de entrevista, la principal manda
        order = sorted(range(len(faces)), key=lambda i: -bbox_area(faces[i][1]))
        taken: set[int] = set()
        for i in order:
            vec, box = faces[i]
            best, best_iou = None, self.iou_match
            for tr in active:
                if id(tr) in taken:
                    continue
                v = bbox_iou(tr.last_box, box)
                if v >= best_iou:
                    best, best_iou = tr, v
            if best is None:
                best = Track(self._next); self._next += 1
                self.tracks.append(best)
            taken.add(id(best))
            best.add(fi, t, vec, box)

    def dominant(self) -> Track | None:
        return max(self.tracks, key=Track.score) if self.tracks else None

    def report(self) -> list[dict]:
        """Puntuación de cada pista, para que la elección sea auditable."""
        rows = [{"track_id": tr.tid, "frames": len(tr.frames),
                 "median_area": float(np.median([bbox_area(b) for b in tr.boxes])),
                 "t_start": float(tr.times[0]), "t_end": float(tr.times[-1]),
                 "score": tr.score()} for tr in self.tracks if tr.boxes]
        rows.sort(key=lambda r: -r["score"])
        total = sum(r["score"] for r in rows) or 1.0
        for r in rows:
            r["score_share"] = r["score"] / total
        return rows


def dominance_margin(rows: list[dict]) -> float:
    """Cuánto destaca la pista elegida sobre la segunda (0 = empate técnico).

    Un margen bajo en una entrevista significa que el montaje reparte el tiempo
    entre las dos personas y que la elección automática no es fiable: hay que
    mirarlo a mano antes de dar por buenos los rasgos cinésicos.
    """
    if not rows:
        return 0.0
    if len(rows) == 1:
        return 1.0
    return float(rows[0]["score_share"] - rows[1]["score_share"])
