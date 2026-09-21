#!/usr/bin/env python3
"""Descarga/precalienta los modelos necesarios (ejecutar en tu máquina, con internet)."""
from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FACE_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
            "face_landmarker/float16/latest/face_landmarker.task")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--whisper", default="small")
    ap.add_argument("--roberta", action="store_true")
    a = ap.parse_args()
    dst = ROOT / "models/face_landmarker.task"
    dst.parent.mkdir(exist_ok=True)
    if not dst.exists():
        print("MediaPipe face_landmarker.task ..."); urllib.request.urlretrieve(FACE_URL, dst)
    print(f"  ok {dst} ({dst.stat().st_size/1e6:.1f} MB)")
    print(f"Faster-Whisper '{a.whisper}' (int8) ...")
    from faster_whisper import WhisperModel
    WhisperModel(a.whisper, device="cpu", compute_type="int8"); print("  ok")
    from silero_vad import load_silero_vad
    load_silero_vad(onnx=True); print("Silero VAD ok (incluido en el paquete)")
    if a.roberta:
        from transformers import AutoModel, AutoTokenizer
        AutoTokenizer.from_pretrained("PlanTL-GOB-ES/roberta-base-bne"); AutoModel.from_pretrained("PlanTL-GOB-ES/roberta-base-bne")
        print("RoBERTa-BNE ok")


if __name__ == "__main__":
    sys.exit(main())
