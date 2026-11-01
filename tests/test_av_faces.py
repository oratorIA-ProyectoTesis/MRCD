import numpy as np

from mrcd.av import faces as F
from mrcd.extractors.face_tracking import FaceTracker


def _lm(open_px=0.0, scale=1.0):
    xy = np.zeros((478, 2))
    xy[F.I_TOP] = (50, 0)
    xy[F.I_CHIN] = (50, 100 * scale)
    xy[F.I_EYE_L] = (30 * 1, 30)
    xy[F.I_EYE_R] = (70, 30)
    xy[F.I_UP_LIP] = (50, 70 * scale)
    xy[F.I_LO_LIP] = (50, 70 * scale + open_px * scale)
    xy[F.I_MOUTH_L] = (40, 72)
    xy[F.I_MOUTH_R] = (60, 72)
    xy[F.I_ALAR_L] = (45, 55)
    xy[F.I_ALAR_R] = (55, 55)
    xy[F.I_CHEEK_L] = (20, 50)
    xy[F.I_CHEEK_R] = (80, 50)
    return xy


def test_lip_aperture_is_geometric_and_scale_free():
    a0, s0 = F.face_scalars(_lm(0.0))
    a1, s1 = F.face_scalars(_lm(10.0))
    assert a0 == 0.0 and abs(a1 - 0.10) < 1e-6 and s0.shape == (F.SHAPE_DIM,)
    a2, s2 = F.face_scalars(_lm(10.0, scale=2.0))
    assert abs(a2 - a1) < 0.02  # face height scales with it


def test_tracks_to_arrays_marks_absence_and_frontal():
    tr = FaceTracker()
    box = np.array([0.2, 0.2, 0.6, 0.7], np.float32)
    for f in range(30):
        if f in (10, 11):
            tr.update(f, f / 25, [])
            continue
        v = np.full(F.FACE_DIM, 0.1, np.float32)
        v[4] = 60.0 if f == 20 else 5.0  # yaw
        v[3] = 0.0
        tr.update(f, f / 25, [(v, box)])
    vec, present, frontal, _ = F.tracks_to_arrays(tr, 30, min_frames=10)
    assert vec.shape == (1, 30, F.FACE_DIM) and present[0].sum() == 28
    assert (
        not present[0, 10] and np.isnan(vec[0, 10]).all()
    )  # absent frame is NaN, not 0
    assert present[0, 20] and not frontal[0, 20] and frontal[0, 21]
