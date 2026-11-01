import numpy as np

from mrcd.av.evidence import evidence_from_tracks, evidence_from_video
from mrcd.av.faces import FACE_DIM

FPS, SR = 25.0, 16000


def _tracks(n, rng, lip_shift=1):
    z = np.convolve(rng.normal(size=n), np.ones(5) / 5, mode="same")
    z = (z - z.mean()) / z.std()
    amp = np.repeat(10 ** (0.4 * z), int(SR / FPS))
    audio = (rng.normal(size=len(amp)) * amp * 0.05).astype(np.float32)
    vec = np.full((1, n, FACE_DIM), np.nan, np.float32)
    vec[0, :, 0] = 0.05 + 0.02 * np.roll(z, lip_shift)
    vec[0, :, 1] = 0.2 + 0.05 * np.roll(z, lip_shift)
    vec[0, :, 2] = 0.1
    vec[0, :, 3:6] = rng.normal(0, 2, size=(n, 3))
    vec[0, :, 6:] = np.array([0.4, 0.5, 0.3, 0.7, 0.2]) + 0.002 * rng.normal(size=(n, 5))
    pres = np.ones((1, n), bool)
    return {"fps": FPS, "n": n, "vec": vec, "present": pres, "frontal": pres.copy()}, audio


def test_gated_evidence_available_for_synchronised_spans_and_marked_experimental():
    rng = np.random.default_rng(0)
    n = 25 * 90
    tr, au = _tracks(n, rng)
    spans = [(4 + 7 * i, 9 + 7 * i) for i in range(10)]
    out, v = evidence_from_tracks(tr, au, SR, spans)
    assert v["reason"] == "ok"
    ok = [o for o in out if o["available"]]
    assert len(ok) >= 7 and all(o["experimental"] and "experimental" in o["note"] for o in out)
    assert set(ok[0]["features"]) and all(np.isfinite(list(ok[0]["features"].values())))


def test_no_face_gives_not_available_with_reason_never_zero():
    rng = np.random.default_rng(1)
    n = 25 * 60
    tr, au = _tracks(n, rng)
    tr["present"][:] = False
    tr["frontal"][:] = False
    out, _ = evidence_from_tracks(tr, au, SR, [(5, 10), (15, 20), (25, 30)])
    assert all(not o["available"] and o["reason"] for o in out)
    assert all("features" not in o for o in out)


def test_missing_video_is_not_available_with_reason():
    out, v = evidence_from_video("does_not_exist.mp4", np.zeros(16000, np.float32), SR, [(0, 1)], "x.task")
    assert out[0]["available"] is False and out[0]["reason"] == "video_file_not_found"
