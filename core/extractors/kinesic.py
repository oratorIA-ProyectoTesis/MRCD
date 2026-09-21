"""Extractor cinésico: MediaPipe Face Landmarker -> vector canónico 12-D.

Requiere el modelo `face_landmarker.task` (descarga oficial):
https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task

Salida por frame (10 fps): [head_pitch, head_yaw, head_roll, eye_look_in_left,
eye_look_out_right, eye_look_up_left, eye_look_down_right, mouth_press_left,
mouth_press_right, jaw_open, mouth_pucker, brow_down_avg] + bandera `detected`.

Nota metodológica: MediaPipe no entrega pitch/yaw/roll directamente; se derivan
de la matriz de transformación facial 4x4 (rotación 3x3 -> ángulos de Euler, orden XYZ).
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

from core.constants import KINESIC_DIM, VIDEO_FPS

BLENDSHAPE_MAP = {
    3: ("eyeLookInLeft",),
    4: ("eyeLookOutRight",),
    5: ("eyeLookUpLeft",),
    6: ("eyeLookDownRight",),
    7: ("mouthPressLeft",),
    8: ("mouthPressRight",),
    9: ("jawOpen",),
    10: ("mouthPucker",),
    11: ("browDownLeft", "browDownRight"),   # promedio
}
DEFAULT_MODEL = os.environ.get("MRCD_FACE_MODEL", "models/face_landmarker.task")


@dataclass
class KinesicTrack:
    times: np.ndarray        # (T,) s (timestamp de captura del frame)
    vectors: np.ndarray      # (T, 12) float32, NaN si no hubo detección
    detected: np.ndarray     # (T,) bool
    bboxes: np.ndarray       # (T, 4) x0,y0,x1,y1 normalizados, NaN si no hubo detección
    frontal: np.ndarray      # (T,) bool  |yaw| y |pitch| dentro del cono semi-frontal
    n_faces: np.ndarray | None = None   # (T,) rostros detectados (si num_faces > 1)

    @property
    def detection_rate(self) -> float:
        return float(self.detected.mean()) if len(self.detected) else 0.0


def rotation_to_euler_deg(R: np.ndarray) -> tuple[float, float, float]:
    """Matriz 3x3 -> (pitch, yaw, roll) en grados (convención XYZ)."""
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    if sy > 1e-6:
        pitch = np.arctan2(R[2, 1], R[2, 2])
        yaw = np.arctan2(-R[2, 0], sy)
        roll = np.arctan2(R[1, 0], R[0, 0])
    else:  # gimbal lock
        pitch = np.arctan2(-R[1, 2], R[1, 1]); yaw = np.arctan2(-R[2, 0], sy); roll = 0.0
    return tuple(float(np.degrees(a)) for a in (pitch, yaw, roll))  # type: ignore


def result_to_vector(result) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Convierte un FaceLandmarkerResult al vector 12-D (+ bbox)."""
    if not result.face_landmarks:
        return None, None
    vec = np.full(KINESIC_DIM, np.nan, dtype=np.float32)
    if result.facial_transformation_matrixes:
        M = np.asarray(result.facial_transformation_matrixes[0])
        vec[0:3] = rotation_to_euler_deg(M[:3, :3])
    bs = {c.category_name: c.score for c in result.face_blendshapes[0]} if result.face_blendshapes else {}
    for idx, names in BLENDSHAPE_MAP.items():
        vals = [bs[n] for n in names if n in bs]
        if vals:
            vec[idx] = float(np.mean(vals))
    lm = result.face_landmarks[0]
    xs = np.array([p.x for p in lm]); ys = np.array([p.y for p in lm])
    bbox = np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float32)
    return vec, bbox


class KinesicExtractor:
    def __init__(self, model_path: str = DEFAULT_MODEL, frontal_yaw_deg: float = 45.0,
                 frontal_pitch_deg: float = 35.0, num_faces: int = 1):
        import mediapipe as mp  # type: ignore
        from mediapipe.tasks import python as mp_python  # type: ignore
        from mediapipe.tasks.python import vision  # type: ignore

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"No existe {model_path}. Descárgalo con scripts/download_models.py o define MRCD_FACE_MODEL.")
        self._mp = mp
        opts = vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=num_faces,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.landmarker = vision.FaceLandmarker.create_from_options(opts)
        self.frontal_yaw = frontal_yaw_deg
        self.frontal_pitch = frontal_pitch_deg

    def process_video_tracked(self, video_path: str, target_fps: int = VIDEO_FPS,
                              max_seconds: float | None = None):
        """Como `process_video`, pero siguiendo la identidad del rostro dominante.

        Obligatorio cuando puede haber más de una persona en cuadro: MediaPipe no
        garantiza orden estable de caras entre frames, así que tomar siempre la
        cara [0] mezcla al entrevistado con el entrevistador.

        Devuelve `(KinesicTrack, report)`, donde `report` lista la puntuación de
        cada pista para poder auditar a quién se siguió.
        """
        import cv2

        from core.extractors.face_tracking import FaceTracker, dominance_margin, result_to_faces

        cap = cv2.VideoCapture(video_path)
        src_fps = cap.get(cv2.CAP_PROP_FPS) or target_fps
        step = max(1, int(round(src_fps / target_fps)))
        tracker = FaceTracker()
        grid_t, grid_n, i, fi = [], [], 0, 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if i % step == 0:
                t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0 if cap.get(cv2.CAP_PROP_POS_MSEC) > 0 else i / src_fps
                if max_seconds and t > max_seconds:
                    break
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
                res = self.landmarker.detect_for_video(img, int(t * 1000))
                faces = result_to_faces(res)
                tracker.update(fi, t, faces)
                grid_t.append(t); grid_n.append(len(faces)); fi += 1
            i += 1
        cap.release()

        rows = tracker.report()
        dom = tracker.dominant()
        n = len(grid_t)
        vecs = np.full((n, KINESIC_DIM), np.nan, np.float32)
        boxes = np.full((n, 4), np.nan, np.float32)
        det = np.zeros(n, bool)
        if dom is not None:
            pos = {f: k for k, f in enumerate(range(n))}
            for f, v, b in zip(dom.frames, dom.vecs, dom.boxes):
                k = pos.get(f)
                if k is not None:
                    vecs[k], boxes[k], det[k] = v, b, True
        with np.errstate(invalid="ignore"):
            frontal = det & (np.abs(vecs[:, 1]) <= self.frontal_yaw) & (np.abs(vecs[:, 0]) <= self.frontal_pitch)
        track = KinesicTrack(np.asarray(grid_t, np.float64), vecs, det, boxes, frontal,
                             np.asarray(grid_n, np.int8))
        report = {"tracks": rows, "selected": dom.tid if dom else None,
                  "dominance_margin": dominance_margin(rows),
                  "frames": n, "frames_with_dominant": int(det.sum())}
        return track, report

    def process_video(self, video_path: str, target_fps: int = VIDEO_FPS,
                      max_seconds: float | None = None) -> KinesicTrack:
        import cv2

        cap = cv2.VideoCapture(video_path)
        src_fps = cap.get(cv2.CAP_PROP_FPS) or target_fps
        step = max(1, int(round(src_fps / target_fps)))
        times, vecs, dets, boxes, nf = [], [], [], [], []
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if i % step == 0:
                t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0 if cap.get(cv2.CAP_PROP_POS_MSEC) > 0 else i / src_fps
                if max_seconds and t > max_seconds:
                    break
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
                res = self.landmarker.detect_for_video(img, int(t * 1000))
                v, b = result_to_vector(res)
                nf.append(len(res.face_landmarks or []))
                times.append(t)
                dets.append(v is not None)
                vecs.append(v if v is not None else np.full(KINESIC_DIM, np.nan, np.float32))
                boxes.append(b if b is not None else np.full(4, np.nan, np.float32))
            i += 1
        cap.release()
        vecs_a = np.asarray(vecs, np.float32).reshape(-1, KINESIC_DIM)
        det_a = np.asarray(dets, bool)
        with np.errstate(invalid="ignore"):
            frontal = det_a & (np.abs(vecs_a[:, 1]) <= self.frontal_yaw) & (np.abs(vecs_a[:, 0]) <= self.frontal_pitch)
        return KinesicTrack(np.asarray(times, np.float64), vecs_a, det_a,
                            np.asarray(boxes, np.float32).reshape(-1, 4), frontal, np.asarray(nf, np.int8))

    def close(self):
        self.landmarker.close()


def continuous_face_segments(track: KinesicTrack, min_len_s: float = 10.0,
                             max_gap_s: float = 0.5, require_frontal: bool = True) -> list[tuple[float, float]]:
    """Segmentos con detección facial continua (tolera huecos <= max_gap_s)."""
    ok = track.frontal if require_frontal else track.detected
    segs, start, last_ok = [], None, None
    for t, good in zip(track.times, ok):
        if good:
            if start is None:
                start = t
            last_ok = t
        elif start is not None and last_ok is not None and t - last_ok > max_gap_s:
            segs.append((start, last_ok)); start = None
    if start is not None and last_ok is not None:
        segs.append((start, last_ok))
    return [(float(a), float(b)) for a, b in segs if b - a >= min_len_s]
