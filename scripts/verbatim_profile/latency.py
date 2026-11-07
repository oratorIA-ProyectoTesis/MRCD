"""Indicative end-to-end latency of profile cw2 (load, transcription, route) on a licence-safe synthetic WAV.
usage (crisper venv): python latency.py --device cuda|cpu [--runs 5] ; writes results/verbatim_profile_20261107/latency_<device>.json
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

WT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WT))
ap = argparse.ArgumentParser()
ap.add_argument("--device", required=True, choices=["cuda", "cpu"])
ap.add_argument("--runs", type=int, default=5)
a = ap.parse_args()

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402

if a.device == "cpu":
    torch.set_num_threads(2)
from mrcd import verbatim as V  # noqa: E402
from mrcd.extractors.linguistic import VerbatimASR  # noqa: E402

wav = WT / "mrcd" / "service" / "web" / "sample_es.wav"
audio, sr = sf.read(wav, dtype="float32")
assert sr == 16000 and audio.ndim == 1
t0 = time.perf_counter()
asr = VerbatimASR(profile="cw2", device=a.device)
load_s = time.perf_counter() - t0


def once():
    t = time.perf_counter()
    words = asr.transcribe_chunked(audio)
    t_asr = time.perf_counter() - t
    t = time.perf_counter()
    V.route_decisions(" ".join(w.text for w in words), {"repetition": 0, "filler_word": 0})
    return t_asr, time.perf_counter() - t, len(words)


once()  # warm-up, discarded
runs = []
for _ in range(a.runs):
    runs.append(once())
    if runs[-1][0] > 120:
        break
tot = [r[0] + r[1] for r in runs]
out = {
    "device": asr.device,
    "compute_type": asr.compute_type,
    "audio_s": round(len(audio) / sr, 3),
    "n_words": runs[-1][2],
    "model_load_s": round(load_s, 2),
    "n_timed_runs": len(runs),
    "transcribe_s": {"median": statistics.median(r[0] for r in runs), "min": min(r[0] for r in runs), "max": max(r[0] for r in runs)},
    "route_s_median": statistics.median(r[1] for r in runs),
    "end_to_end_s_median": statistics.median(tot),
    "real_time_factor_median": statistics.median(tot) / (len(audio) / sr),
    "torch": torch.__version__,
    "note": "indicative: shared machine (other agents), not a benchmark; classifier pass (P0 ASR) not included",
}
dest = WT / "results" / "verbatim_profile_20261107" / f"latency_{a.device}.json"
dest.write_text(json.dumps(out, indent=1), "utf8")
print(json.dumps(out, indent=1))
