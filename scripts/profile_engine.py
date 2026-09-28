"""Profile MRCD per stage (M1/P05) and measure the historical per-window extraction
against the prepared extract-once path on the same audio and words.

    python scripts/profile_engine.py audio16k.wav --start-s 25 --end-s 65 --ckpt models/ckpt_audio_text.pt
    python scripts/profile_engine.py audio16k.wav --random-init   # timing/equivalence only, no valid labels
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.constants import LABELS, SAMPLE_RATE  # noqa: E402
from core.dataset import window_starts  # noqa: E402
from core.engine import MRCDEngine, window_tensors  # noqa: E402
from core.extractors.linguistic import VerbatimASR  # noqa: E402
from core.inference import InferenceEngine  # noqa: E402
from core.live.window_builder import build_window  # noqa: E402


def random_engine() -> InferenceEngine:
    import torch

    from core.models.fusion import CrossModalFusion

    torch.manual_seed(0)
    net = CrossModalFusion(len(LABELS), mode="audio_text").eval()
    norm = {k: (np.zeros(w, np.float32), np.ones(w, np.float32)) for k, w in (("ac", 5), ("ling", 12), ("pros", 9))}
    return InferenceEngine(net, norm, list(LABELS), {"label_source": "random_init"}, "cpu")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("audio")
    ap.add_argument("--start-s", type=float, default=0.0)
    ap.add_argument("--end-s", type=float)
    ap.add_argument("--ckpt")
    ap.add_argument("--random-init", action="store_true")
    ap.add_argument("--whisper", default="small")
    ap.add_argument("--skip-legacy", action="store_true")
    ap.add_argument("--out")
    a = ap.parse_args()
    audio, sr = sf.read(
        a.audio,
        dtype="float32",
        start=int(a.start_s * SAMPLE_RATE),
        stop=None if a.end_s is None else int(a.end_s * SAMPLE_RATE),
        always_2d=True,
    )
    if sr != SAMPLE_RATE:
        raise SystemExit(f"se esperaba {SAMPLE_RATE} Hz; convierte antes con ffmpeg")
    audio = audio.mean(1)
    t = time.perf_counter()
    inference = random_engine() if a.random_init else InferenceEngine.from_checkpoint(Path(a.ckpt), "cpu")
    engine = MRCDEngine(inference, VerbatimASR(a.whisper, device="cpu"), init_s=time.perf_counter() - t)
    cold = engine.analyze(audio)
    prep = engine.prepare(audio)
    warm = engine.analyze(audio, prepared=prep)
    report = {
        "audio": a.audio,
        "region_s": [a.start_s, a.end_s],
        "duration_s": len(audio) / SAMPLE_RATE,
        "init_s": engine.init_s,
        "cold": cold["timings"],
        "cold_rtf": cold["rtf"],
        "warm_prepared": warm["timings"],
        "hardware": cold["hardware"],
        "n_events": len(cold["events"]),
        "labels_valid": not a.random_init,
    }
    if not a.skip_legacy:
        starts = window_starts(prep.duration_s)
        kw = {
            "context": bool(engine.ling_dim and engine.ling_dim > 10),
            "include_pros": engine.include_pros,
            "ling_dim": engine.ling_dim,
        }
        t = time.perf_counter()
        legacy = np.concatenate([inference.predict(build_window(audio, prep.words, s, **kw)) for s in starts])
        report["legacy_per_window_s"] = time.perf_counter() - t
        new = inference.predict(window_tensors(prep.frames, prep.words, starts, **kw))
        report["n_windows"] = len(starts)
        report["max_abs_prob_diff"] = float(np.abs(legacy - new).max()) if len(starts) else 0.0
    text = json.dumps(report, indent=1, default=str)
    print(text)
    if a.out:
        Path(a.out).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
