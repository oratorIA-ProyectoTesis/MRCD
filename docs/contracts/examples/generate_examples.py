"""Generate the REAL JSON fixtures of docs/contracts/examples/ by running the engine (no mocks).

Exact command used (repo root, Git Bash; OMP must be in the environment BEFORE numpy is imported):

    OMP_NUM_THREADS=4 .venv/Scripts/python.exe docs/contracts/examples/generate_examples.py \
        --source data/clips/FIwdDFht7FE.wav --start 60 --dur 20 --out docs/contracts/examples

HF_HUB_OFFLINE=1 is required on this machine (without it a second engine load in one process hung >10 min with no CPU use;
not investigated, the model is in the local cache). Outputs: batch_p0.json, batch_p0_windows.json, batch_p0_vc.json, batch_p1.json, error_short_audio.json,
live_rows_v1.json, live_rows_v2.json, live_rows_v3.json, provenance.json.
The audio excerpt is written next to the scratch directory and is NOT versioned (source audio must not be
redistributed). The only post-processing is replacing the user home directory by "<HOME>" in strings.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import sys
import time
import wave
from pathlib import Path

import faulthandler

import numpy as np

faulthandler.dump_traceback_later(900, exit=True)
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))


def cut(src: Path, start: float, dur: float, dst: Path) -> None:
    with wave.open(str(src), "rb") as w:
        assert (w.getframerate(), w.getnchannels(), w.getsampwidth()) == (16000, 1, 2)
        w.setpos(int(start * 16000))
        frames = w.readframes(int(dur * 16000))
    with wave.open(str(dst), "wb") as o:
        o.setnchannels(1)
        o.setsampwidth(2)
        o.setframerate(16000)
        o.writeframes(frames)


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def scrub(obj, home: str):
    if isinstance(obj, dict):
        return {k: scrub(v, home) for k, v in obj.items()}
    if isinstance(obj, list):
        return [scrub(v, home) for v in obj]
    if isinstance(obj, str):
        return obj.replace(home, "<HOME>").replace(home.replace("\\", "/"), "<HOME>")
    return obj


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--start", type=float, default=60.0)
    ap.add_argument("--dur", type=float, default=20.0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--clip", default=str(Path(os.environ.get("TEMP", ".")) / "mrcd_fixture_clip.wav"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    clip = Path(a.clip)
    cut(Path(a.source), a.start, a.dur, clip)
    home = str(Path.home())
    timings: dict[str, float] = {}

    def dump(name: str, data) -> None:
        (out / name).write_text(
            json.dumps(scrub(data, home), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )

    import mrcd
    from mrcd import DisfluencyEngine
    from mrcd.inference import predict

    ckpt = ROOT / "models" / "ckpt_audio_text.pt"
    t = time.perf_counter()
    eng = DisfluencyEngine.load(ckpt, head="vc")  # V0 output is unchanged; the head is opt-in extra output
    timings["load_engine_with_vc_s"] = time.perf_counter() - t

    t = time.perf_counter()
    res = eng.analyze_file(clip, keep_windows=True, terminal_window=True)
    timings["analyze_p0_vc_s"] = time.perf_counter() - t
    dump("batch_p0_vc.json", res.to_dict())
    dump("batch_p0_windows.json", [dataclasses.asdict(w) for w in res.windows])

    # One engine instance only (a second WhisperModel load in the same process hung on this machine).
    # Plain V0: same object with the opt-in head detached; P1: same ASR object with the profile swapped.
    from mrcd.asr_profiles import get_profile

    eng._head = None
    eng0 = eng
    t = time.perf_counter()
    r0 = eng0.analyze_file(clip)
    timings["analyze_p0_s"] = time.perf_counter() - t
    dump("batch_p0.json", r0.to_dict())

    eng._asr.profile = get_profile("p1_disfluent_prompt")
    t = time.perf_counter()
    r1 = eng.analyze_file(clip)
    timings["analyze_p1_s"] = time.perf_counter() - t
    dump("batch_p1.json", r1.to_dict())
    eng._asr.profile = get_profile("p0")

    try:
        eng0.analyze(np.zeros(16000 * 5, np.float32))
    except ValueError as exc:
        dump("error_short_audio.json", {"exception": "ValueError", "message": str(exc)})

    # Live path replay: same scorer as scripts/live_demo.py, but no microphone and ASR words taken from the offline
    # transcription (ticks of 0.5 s). It exercises LiveScorer rows, not the streaming ASR.
    from mrcd.extractors.linguistic import Word
    from mrcd.live.session import LiveScorer

    with wave.open(str(clip), "rb") as w:
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32768.0
    words = eng0._asr.transcribe_chunked(pcm, chunk_s=8.0)
    for cfg in ("live_cfg_v1", "live_cfg_v2", "live_cfg_v3"):
        t = time.perf_counter()
        sc = LiveScorer(
            lambda tens: predict(eng0._net, eng0._norm, tens, eng0._device, bs=1)[0],
            list(eng0.labels), config=cfg, f0_center="causal", context=eng0._context,
            include_pros=eng0._include_pros, ling_dim=eng0._ling_dim,
        )
        tick = 8000  # 0.5 s
        for k in range(0, len(pcm), tick):
            sc.push(pcm[k : k + tick])
            sc.step(words)
        sc.finalize(words)
        timings[f"replay_{cfg}_s"] = time.perf_counter() - t
        dump(f"live_rows_{cfg[-2:]}.json", {"config": cfg, "gaps": sc.gaps, "rows": sc.rows})

    dump(
        "provenance.json",
        {
            "command": "HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 .venv/Scripts/python.exe -u docs/contracts/examples/generate_examples.py "
            f"--source {a.source} --start {a.start} --dur {a.dur} --out {a.out} --clip <scratch>/clip.wav",
            "mrcd_version": mrcd.__version__,
            "source_clip": a.source,
            "excerpt": {"start_s": a.start, "dur_s": a.dur, "sha256": sha(clip)},
            "checkpoint_sha256": sha(ckpt),
            "omp_num_threads_env": os.environ.get("OMP_NUM_THREADS"),
            "wall_times_s_one_run_machine_shared_with_other_jobs_not_a_benchmark": {k: round(v, 2) for k, v in timings.items()},
            "live_replay_note": "no microphone; ASR words from offline transcribe_chunked(8 s, 2 s); 0.5 s ticks",
        },
    )
    print(json.dumps(timings, indent=1))


if __name__ == "__main__":
    main()
