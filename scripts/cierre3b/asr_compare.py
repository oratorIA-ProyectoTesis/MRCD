"""Paired ASR decoding of the SAME units under several conditions (Task 3.2). Resumable, one JSON per (condition, unit).

conditions (name -> how):
  off_p0 / off_p1            engine path: VerbatimASR.transcribe_chunked(chunk 8 s, overlap 2 s) with profile p0 / p1_disfluent_prompt
  str_<p>_t<tick>            live path simulated: ticks every <tick> s from 6.0 s (warm-up, as the live tracker) plus a final tick at the
                             end; each tick re-transcribes the last 13 s (CONTEXT_S+WINDOW_S) and merges with StreamingASR.merge_recent
  str_<p>_t<tick>_<variant>  same with a compatible variant (``edge_guard`` etc., see mrcd/live/streaming_asr.py); default behaviour untouched
usage: python asr_compare.py --out DIR --conds off_p0 off_p1 str_p0_t2 [--units-file F] [--limit N]
OMP_NUM_THREADS=4 is pinned before numpy (_pin_omp).
"""
import sys, pathlib as _p
sys.path.insert(0, str(_p.Path(__file__).resolve().parents[1]))
import _pin_omp  # noqa: F401,E402
import argparse, json, os, re, time  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path[:0] = [str(REPO), str(HERE)]
import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import _sys  # noqa: E402
from mrcd.constants import CONTEXT_S, SAMPLE_RATE, WINDOW_S  # noqa: E402
from mrcd.asr_profiles import get_profile  # noqa: E402
from mrcd.extractors.linguistic import VerbatimASR  # noqa: E402
from mrcd.live.streaming_asr import StreamingASR  # noqa: E402

MODEL = Path.home() / ".cache/huggingface/hub/models--Systran--faster-whisper-small/snapshots/536b0662742c02347bc0e980a01041f333bce120"
PROFILES = {"p0": "p0", "p1": "p1_disfluent_prompt"}
WARMUP_S = 6.0


def parse(cond):
    m = re.fullmatch(r"(off|str)_(p0|p1)(?:_t([0-9.]+))?(?:_(\w+))?", cond)
    if not m:
        raise SystemExit(f"bad condition {cond}")
    kind, prof, tick, variant = m.groups()
    return kind, PROFILES[prof], float(tick) if tick else None, variant


def stream(sasr, audio, tick, variant=None):
    from mrcd.live.streaming_asr import ChunkAlignedState

    keep = int(round((CONTEXT_S + WINDOW_S) * SAMPLE_RATE))
    dur = len(audio) / SAMPLE_RATE
    ticks = list(np.arange(WARMUP_S, dur, tick)) + [dur]
    words, calls = [], 0
    st = ChunkAlignedState() if variant == "ca" else None
    for T in ticks:
        n = int(round(T * SAMPLE_RATE))
        seg = audio[max(n - keep, 0):n]
        begin = (n - len(seg)) / SAMPLE_RATE
        words = sasr.step_chunk_aligned(st, seg, begin) if st else sasr.merge_recent(words, seg, begin)
        calls += 1
    return words, calls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--conds", nargs="+", required=True)
    ap.add_argument("--units-file", default=str(REPO / "results/cierre_oe1_oe3_20261031/3b/asr/units.json"))
    ap.add_argument("--subset-ids", default=None, help="json list of unit_ids to restrict to")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    units = json.loads(Path(a.units_file).read_text("utf8"))
    if a.subset_ids:
        ids = set(json.loads(Path(a.subset_ids).read_text("utf8"))); units = [u for u in units if u["unit_id"] in ids]
    if a.limit:
        units = units[:a.limit]
    t0 = time.perf_counter()
    asr = VerbatimASR(str(MODEL), device="cpu", compute_type="int8")
    sasr = StreamingASR(asr=asr)
    load_s = time.perf_counter() - t0
    meta = {"model_load_s": load_s, "model_info": asr.model_info(), "hardware": _sys.hardware(), "units": len(units),
            "code_sha256_mrcd": _sys.code_sha(REPO), "argv": sys.argv, "date": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "background_start": _sys.background_report(3), "profiles": {k: get_profile(v).decode_kwargs("es") for k, v in PROFILES.items()}}
    (out / f"meta_{int(time.time())}.json").write_text(json.dumps(meta, indent=1, default=str), "utf8")
    print("model load", round(load_s, 2), "s; units", len(units), flush=True)
    for cond in a.conds:
        kind, prof, tick, variant = parse(cond)
        asr.profile = get_profile(prof)
        d = out / cond; d.mkdir(exist_ok=True)
        tc = time.perf_counter(); n_done = 0
        for i, u in enumerate(units):
            p = d / f"{u['unit_id']}.json"
            if p.exists():
                continue
            audio, sr = sf.read(u["wav"], dtype="float32"); assert sr == SAMPLE_RATE and audio.ndim == 1
            t1 = time.perf_counter(); c1 = time.process_time()
            if kind == "off":
                words = asr.transcribe_chunked(audio, chunk_s=8.0, overlap_s=2.0); calls = 1
            else:
                words, calls = stream(sasr, audio, tick, variant)
            rec = {"unit_id": u["unit_id"], "condition": cond, "duration_s": len(audio) / SAMPLE_RATE, "asr_calls": calls,
                   "wall_s": time.perf_counter() - t1, "cpu_s": time.process_time() - c1,
                   "words": [w.to_dict() for w in words]}
            tmp = p.with_suffix(".tmp"); tmp.write_text(json.dumps(rec, ensure_ascii=False), "utf8"); tmp.replace(p)
            n_done += 1
            if n_done % 10 == 1:
                print(cond, i + 1, "/", len(units), round(time.perf_counter() - tc), "s", flush=True)
        print("DONE", cond, round(time.perf_counter() - tc), "s", "peak_mb", _sys.mem_mb()["peak_rss_mb"], flush=True)


if __name__ == "__main__":
    main()
