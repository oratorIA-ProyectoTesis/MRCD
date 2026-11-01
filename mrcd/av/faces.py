"""Per-video face tracks with purely geometric, observable measures (no affect or intent).

For every frame and every detected face we keep scalars only (lip aperture, jaw opening, head pose, a landmark-ratio
descriptor used to verify that the same face is followed). Per-frame landmarks are NOT stored.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from mrcd.av.gate import FaceSeries
from mrcd.extractors.face_tracking import FaceTracker
from mrcd.extractors.kinesic import BLENDSHAPE_MAP, rotation_to_euler_deg  # noqa: F401

# MediaPipe face-mesh indices
I_UP_LIP, I_LO_LIP = 13, 14
I_MOUTH_L, I_MOUTH_R = 61, 291
I_TOP, I_CHIN = 10, 152
I_CHEEK_L, I_CHEEK_R = 234, 454
I_EYE_L, I_EYE_R = 33, 263
I_ALAR_L, I_ALAR_R = 98, 327

SHAPE_DIM = 5
# per-face vector: [lip_aperture, jaw_open, mouth_pucker, pitch, yaw, roll, shape(5)]
FACE_DIM = 6 + SHAPE_DIM


def face_scalars(xy: np.ndarray) -> tuple[float, np.ndarray]:
    """From landmark pixel coordinates (>=468, 2): (lip aperture / face height, shape descriptor (5,))."""
    d = lambda a, b: float(np.linalg.norm(xy[a] - xy[b]))  # noqa: E731
    face_h = max(d(I_TOP, I_CHIN), 1e-6)
    iod = max(d(I_EYE_L, I_EYE_R), 1e-6)
    lip = d(I_UP_LIP, I_LO_LIP) / face_h
    shape = np.array(
        [
            iod / face_h,
            d(I_MOUTH_L, I_MOUTH_R) / iod,
            d(I_ALAR_L, I_ALAR_R) / iod,
            d(I_CHEEK_L, I_CHEEK_R) / face_h,
            d(I_MOUTH_L, I_CHIN) / face_h,
        ],
        np.float32,
    )
    return lip, shape


def result_to_face_records(
    result, width: int, height: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    """All faces of a FaceLandmarkerResult as (FACE_DIM vector, bbox x0,y0,x1,y1 normalised)."""
    out = []
    lms = getattr(result, "face_landmarks", None) or []
    mats = getattr(result, "facial_transformation_matrixes", None) or []
    blends = getattr(result, "face_blendshapes", None) or []
    for i, lm in enumerate(lms):
        xn = np.array([p.x for p in lm])
        yn = np.array([p.y for p in lm])
        xy = np.stack([xn * width, yn * height], 1)
        lip, shape = face_scalars(xy)
        vec = np.full(FACE_DIM, np.nan, np.float32)
        vec[0] = lip
        vec[6:] = shape
        if i < len(mats):
            vec[3:6] = rotation_to_euler_deg(np.asarray(mats[i])[:3, :3])
        if i < len(blends):
            bs = {c.category_name: c.score for c in blends[i]}
            vec[1] = bs.get("jawOpen", np.nan)
            vec[2] = bs.get("mouthPucker", np.nan)
        out.append(
            (vec, np.array([xn.min(), yn.min(), xn.max(), yn.max()], np.float32))
        )
    return out


def tracks_to_arrays(
    tracker: FaceTracker,
    n: int,
    max_tracks: int = 6,
    min_frames: int = 10,
    yaw=45.0,
    pitch=35.0,
):
    """Dense (Tr, n, ...) arrays for the best tracks; `frontal` uses the repo's semi-frontal cone."""
    trs = [t for t in tracker.tracks if len(t.frames) >= min_frames]
    trs.sort(key=lambda t: -t.score())
    trs = trs[:max_tracks]
    T = len(trs)
    vec = np.full((T, n, FACE_DIM), np.nan, np.float32)
    present = np.zeros((T, n), bool)
    area = np.full((T, n), np.nan, np.float32)
    for k, t in enumerate(trs):
        for f, v, b in zip(t.frames, t.vecs, t.boxes):
            if f < n:
                vec[k, f], present[k, f] = v, True
                area[k, f] = max(b[2] - b[0], 0) * max(b[3] - b[1], 0)
    with np.errstate(invalid="ignore"):
        frontal = (
            present & (np.abs(vec[..., 4]) <= yaw) & (np.abs(vec[..., 3]) <= pitch)
        )
    return vec, present, frontal, area


def extract_video(video_path: str, model_path: str, num_faces: int = 4):
    """Decode every frame and run the face landmarker. Returns dict of arrays + metadata."""
    import cv2
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

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
    lm = vision.FaceLandmarker.create_from_options(opts)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w, h = (
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )
    tracker, n_faces, k = FaceTracker(), [], 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        img = mp.Image(
            image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
        )
        res = lm.detect_for_video(img, int(round(k * 1000.0 / fps)))
        faces = result_to_face_records(res, w, h)
        tracker.update(k, k / fps, faces)
        n_faces.append(len(faces))
        k += 1
    cap.release()
    lm.close()
    vec, present, frontal, area = tracks_to_arrays(tracker, k)
    return {
        "fps": float(fps),
        "n": k,
        "width": w,
        "height": h,
        "vec": vec,
        "present": present,
        "frontal": frontal,
        "area": area,
        "n_faces": np.asarray(n_faces, np.int8),
    }


def save_tracks(path: Path, d: dict) -> None:
    np.savez_compressed(path, **{k: np.asarray(v) for k, v in d.items()})


def load_tracks(path: Path) -> dict:
    z = np.load(path)
    return {k: z[k] for k in z.files}


def to_face_series(d: dict, env: np.ndarray) -> FaceSeries:
    """Adapt stored tracks to the gate's FaceSeries (lip = column 0, shape = columns 6:)."""
    vec = d["vec"]
    if vec.shape[0] == 0:
        n = int(d["n"])
        return FaceSeries(
            float(d["fps"]),
            np.zeros((0, n)),
            np.zeros((0, n), bool),
            np.zeros((0, n), bool),
            np.zeros((0, n, SHAPE_DIM)),
            env,
        )
    return FaceSeries(
        float(d["fps"]), vec[..., 0], d["present"], d["frontal"], vec[..., 6:], env
    )
