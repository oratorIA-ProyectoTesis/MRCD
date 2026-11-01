"""Video as a SEPARATE modality: decode cost + face-landmark tracking cost only (Task 3.4).

Reads local mp4 clips (data/annotation_batch_01, paired with the wavs of the annotation batch), decodes every frame with OpenCV
and runs the MediaPipe FaceLandmarker (models/face_landmarker.task, local) in VIDEO mode. Measures time, fps, real-time factor
(wall / media duration), peak working set and face coverage. It does NOT measure fusion with audio, identity of the speaker or
audio/video synchrony: none of those is verified (the service contract declares ``kinesic_video`` available:false).
A missing/unreadable video is reported as ``available:false`` with the reason, never as zero.
usage: python video_cost.py --out FILE.json [--reps 3]
"""

import sys
import pathlib as _p

sys.path.insert(0, str(_p.Path(__file__).resolve().parents[1]))
import _pin_omp  # noqa: F401,E402
import argparse
import json
import time  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path[:0] = [str(REPO), str(HERE)]
import numpy as np  # noqa: E402
import _sys  # noqa: E402

DATA = REPO / "data/annotation_batch_01"
CLIPS = [
    "MTRsInkqv5E_s04",
    "p2ZWJP75kwA_s01",
    "ak4Oxthf_fc_s02",
    "ak4Oxthf_fc_s03",
    "does_not_exist_clip",
]


def run_clip(name, lm_factory, reps):
    import cv2
    import mediapipe as mp

    path = DATA / f"{name}.mp4"
    if not path.exists():
        return {
            "clip": name,
            "available": False,
            "unavailable_reason": f"video file not found: {path.name}",
        }
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return {
            "clip": name,
            "available": False,
            "unavailable_reason": "OpenCV could not open the file",
        }
    n, fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), cap.get(cv2.CAP_PROP_FPS)
    w, h = int(cap.get(3)), int(cap.get(4))
    cap.release()
    res = {
        "clip": name,
        "available": True,
        "sha256": _sys.sha256_file(path),
        "frames_declared": n,
        "fps": fps,
        "width": w,
        "height": h,
        "media_duration_s": n / fps if fps else None,
        "decode_only": [],
        "decode_plus_face": [],
        "decode_frames": None,
    }
    for _ in range(reps):
        cap = cv2.VideoCapture(str(path))
        t0, c0, k = time.perf_counter(), time.process_time(), 0
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            k += 1
        res["decode_only"].append(
            {"wall_s": time.perf_counter() - t0, "cpu_s": time.process_time() - c0}
        )
        res["decode_frames"] = k
        cap.release()
    for _ in range(reps):
        lm = lm_factory()
        cap = cv2.VideoCapture(str(path))
        t0, c0, k, with_face, per_frame = (
            time.perf_counter(),
            time.process_time(),
            0,
            0,
            [],
        )
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            t1 = time.perf_counter()
            img = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=cv2.cvtColor(fr, cv2.COLOR_BGR2RGB),
            )
            out = lm.detect_for_video(img, int(k * 1000 / fps))
            per_frame.append(time.perf_counter() - t1)
            with_face += int(len(out.face_landmarks) > 0)
            k += 1
        res["decode_plus_face"].append(
            {
                "wall_s": time.perf_counter() - t0,
                "cpu_s": time.process_time() - c0,
                "frames": k,
                "frames_with_face": with_face,
                "face_pct": 100 * with_face / max(k, 1),
                "per_frame_ms_median": 1000 * float(np.median(per_frame)),
                "per_frame_ms_p95": 1000 * float(np.percentile(per_frame, 95)),
            }
        )
        cap.release()
        lm.close()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=int, default=3)
    a = ap.parse_args()
    out = {
        "hardware": _sys.hardware(),
        "background": _sys.background_report(3),
        "date": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "clips": [],
    }
    try:
        import cv2
        import mediapipe as mp
        from mediapipe.tasks import python as mpt
        from mediapipe.tasks.python import vision

        out["versions"] = {"opencv": cv2.__version__, "mediapipe": mp.__version__}
        model = REPO / "models/face_landmarker.task"
        out["landmarker_model_sha256"] = _sys.sha256_file(model)

        def factory():
            return vision.FaceLandmarker.create_from_options(
                vision.FaceLandmarkerOptions(
                    base_options=mpt.BaseOptions(model_asset_path=str(model)),
                    running_mode=vision.RunningMode.VIDEO,
                    num_faces=1,
                )
            )

        for c in CLIPS:
            out["clips"].append(run_clip(c, factory, a.reps))
    except Exception as e:  # noqa: BLE001
        out["available"] = False
        out["unavailable_reason"] = f"{type(e).__name__}: {e}"
    out["mem"] = _sys.mem_mb()
    out["scope_note"] = (
        "Cost of decoding + face landmarks only. No audio/video fusion, no speaker identity, no synchrony were verified or measured; "
        "DIME AVI files are screen recordings without faces and were not used."
    )
    Path(a.out).write_text(json.dumps(out, indent=1, default=str), "utf8")
    print(json.dumps({c["clip"]: c.get("available") for c in out["clips"]}))


if __name__ == "__main__":
    main()
