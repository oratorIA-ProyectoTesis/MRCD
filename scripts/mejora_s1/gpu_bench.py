"""S1 task 2: CPU int8 vs CUDA ASR cost on the same units (serial, one config per process).

usage: OMP_NUM_THREADS=4 python gpu_bench.py --config p0_cpu|p0_cuda|p1_cpu|p1_cuda --out DIR [--rounds 2]
Decoding = ``VerbatimASR.transcribe_chunked(8 s, 2 s)`` (the offline engine path). Transcripts go to
``DIR/_texts`` (never committed); the committed JSON holds only timings, memory and token counts.
GPU memory is the delta of the device-wide used memory (NVML; WDDM has no per-process figure).
"""

import argparse
import ctypes
import json
import os
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _pin_omp  # noqa: F401,E402

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cierre3b"))
import _sys  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
INPUTS = REPO / "results/cierre_oe1_oe3_20261031/3b/cost/inputs"
CONFIGS = {
    "p0_cpu": ("p0", "cpu", "int8"),
    "p1_cpu": ("p1_disfluent_prompt", "cpu", "int8"),
    "p0_cuda": ("p0_cuda", "cuda", "float16"),
    "p1_cuda": ("p1_cuda", "cuda", "float16"),
    "p0_cuda_i8f16": ("p0", "cuda", "int8_float16"),  # not a registered profile: probe only
}


class Nvml:
    def __init__(self):
        self.ok = False
        try:
            self.lib = ctypes.WinDLL("nvml.dll")
            self.lib.nvmlInit_v2()
            self.h = ctypes.c_void_p()
            self.lib.nvmlDeviceGetHandleByIndex_v2(0, ctypes.byref(self.h))
            self.ok = True
        except Exception:  # noqa: BLE001
            pass

    def used_mb(self):
        class M(ctypes.Structure):
            _fields_ = [("total", ctypes.c_ulonglong), ("free", ctypes.c_ulonglong), ("used", ctypes.c_ulonglong)]
        m = M()
        self.lib.nvmlDeviceGetMemoryInfo(self.h, ctypes.byref(m))
        return m.used / 2**20


class Peak(threading.Thread):
    def __init__(self, nv):
        super().__init__(daemon=True)
        self.nv, self.peak, self.stop = nv, 0.0, False

    def run(self):
        while not self.stop:
            self.peak = max(self.peak, self.nv.used_mb())
            time.sleep(0.05)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, choices=sorted(CONFIGS))
    ap.add_argument("--out", required=True)
    ap.add_argument("--rounds", type=int, default=2)
    a = ap.parse_args()
    out = Path(a.out)
    (out / "_texts").mkdir(parents=True, exist_ok=True)
    prof, dev, ct = CONFIGS[a.config]
    t_proc = time.perf_counter()
    nv = Nvml()
    meta = {"config": a.config, "profile": prof, "device": dev, "compute_type": ct, "hardware": _sys.hardware(),
            "date": time.strftime("%Y-%m-%dT%H:%M:%S"), "omp": os.environ.get("OMP_NUM_THREADS"),
            "background_start": _sys.background_report(3)}
    meta["gate_ok"] = meta["background_start"]["system_cpu_pct"] <= 15.0
    gpu0 = nv.used_mb() if nv.ok else None
    from mrcd.extractors.linguistic import VerbatimASR

    peak = Peak(nv) if nv.ok else None
    if peak:
        peak.start()
    t0, c0 = time.perf_counter(), time.process_time()
    asr = VerbatimASR("small", device=dev, compute_type=ct, profile=prof)
    meta["load_s"] = time.perf_counter() - t0
    meta["load_cpu_s"] = time.process_time() - c0
    meta["actual"] = {"device": asr.device, "compute_type": asr.compute_type, "profile": asr.profile.name}
    meta["mem_after_load"] = _sys.mem_mb()
    meta["gpu_used_mb_before_load"] = gpu0
    meta["gpu_used_mb_after_load"] = nv.used_mb() if nv.ok else None
    files = sorted(INPUTS.glob("*.wav"), key=lambda p: sf.info(p).duration)
    rows = []
    for rnd in range(a.rounds + 1):  # round 0 = first pass (cold: includes kernel JIT/autotune), reported apart
        bg = _sys.system_cpu_load(2)
        for p in files:
            x, sr = sf.read(p, dtype="float32")
            assert sr == 16000 and x.ndim == 1
            t1, c1 = time.perf_counter(), time.process_time()
            words = asr.transcribe_chunked(x)
            dt = time.perf_counter() - t1
            rows.append({"file": p.stem, "round": rnd, "dur_s": len(x) / sr, "wall_s": dt,
                         "cpu_s": time.process_time() - c1, "rtf": dt / (len(x) / sr), "n_words": len(words),
                         "bg_cpu_pct_before_round": bg,
                         "gpu_used_mb": nv.used_mb() if nv.ok else None})
            if rnd == 1:
                (out / "_texts" / f"{a.config}__{p.stem}.txt").write_text(" ".join(w.text for w in words), "utf8")
    if peak:
        peak.stop = True
        peak.join()
        meta["gpu_peak_used_mb"] = peak.peak
    meta["mem_end"] = _sys.mem_mb()
    meta["background_end"] = _sys.system_cpu_load(2)
    meta["total_process_s"] = time.perf_counter() - t_proc
    meta["rows"] = rows
    (out / f"bench_{a.config}.json").write_text(json.dumps(meta, indent=1), "utf8")
    ok = [r for r in rows if r["round"] >= 1]
    print(a.config, "load", round(meta["load_s"], 2), "median RTF", round(float(np.median([r["rtf"] for r in ok])), 3))


if __name__ == "__main__":
    main()
