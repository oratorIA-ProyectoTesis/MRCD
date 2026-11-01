"""Streaming ASR (13 s re-transcription ring, mrcd.live.streaming_asr) vs offline whole-file ASR.

Offline reference = the engine path (transcribe_chunked 8 s / 2 s overlap), same model dir as the frozen
cache. Same model, same process, OMP_NUM_THREADS=4 (pinned before numpy by _pin_omp). Streaming is simulated
with ticks every ``--tick`` seconds (the live demo ticks every 0.5 s; a coarser tick keeps the CPU
budget small and is a stated deviation): at each tick the last 13 s of the audio heard so far are
re-transcribed and merged with ``StreamingASR.merge_recent``; the stream ends with one last tick at the
end of the audio (what ``LiveEngine`` does on stop). Reported per clip: WER between the offline text
and the FINAL streamed text (also vs. the frozen cache), word-time deltas of matched words, and how
many words changed between the text shown at a tick and the final text.

usage: python asr_streaming_parity.py --clips 8 --tick 2.0 --budget-min 18
"""

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import _pin_omp  # noqa: E402,F401

REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

SR = 16_000


def wer(ref, hyp):
    d = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        prev, d[0] = d[0], i
        for j, h in enumerate(hyp, 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (r != h))
            prev, d[j] = d[j], cur
    return d[-1] / max(len(ref), 1), d[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", type=int, default=8)
    ap.add_argument("--tick", type=float, default=2.0)
    ap.add_argument("--budget-min", type=float, default=18.0)
    ap.add_argument("--model", default=str(Path.home() / ".cache/huggingface/hub/models--Systran--faster-whisper-small/snapshots/536b0662742c02347bc0e980a01041f333bce120"))
    ap.add_argument(
        "--out",
        default=str(
            REPO
            / "results/ejecucion_material4_20261030/paridad/asr_streaming_parity_v2.json"
        ),
    )
    a = ap.parse_args()
    from mrcd.constants import CONTEXT_S, WINDOW_S
    from mrcd.extractors.linguistic import VerbatimASR
    from mrcd.live.streaming_asr import StreamingASR
    from mrcd.reproducibility import asr_environment

    man = json.loads(
        (REPO / "results/baselines_20261030/ameresco/manifest.json").read_text("utf8")
    )
    pilot = [m for m in man if m["in_pilot32"]][: a.clips]
    cache = REPO / "results/mejoras_20261030/asr_cache/ameresco"
    asr = VerbatimASR(a.model, device="cpu", compute_type="int8")  # same as scripts/improve/asr_cache.py
    sasr = StreamingASR(asr=asr)
    t_start, out = time.time(), []
    keep = int(round((CONTEXT_S + WINDOW_S) * SR))
    for m in pilot:
        if (time.time() - t_start) / 60 > a.budget_min:
            print("budget exhausted", flush=True)
            break
        audio, _ = sf.read(m["wav"], dtype="float32")
        t0 = time.time()
        off = asr.transcribe_chunked(audio, chunk_s=8.0, overlap_s=2.0)  # engine default (asr_chunk_s=8)
        t_off = time.time() - t0
        words, shown, n_calls = [], [], 0
        ticks = list(np.arange(6.0, len(audio) / SR, a.tick)) + [len(audio) / SR]
        t0 = time.time()
        for T in ticks:
            n = int(round(T * SR))
            seg = audio[max(n - keep, 0) : n]
            begin = (n - len(seg)) / SR
            words = sasr.merge_recent(words, seg, begin)
            shown.append((T, " ".join(w.norm for w in words)))
            n_calls += 1
        t_str = time.time() - t0
        ref = [w.norm for w in off]
        hyp = [w.norm for w in words]
        w_off_vs_str, errs = wer(ref, hyp)
        cached = [
            w["norm"]
            for w in json.loads((cache / f"{m['unit_id']}.json").read_text("utf8"))[
                "words"
            ]
        ]
        w_off_vs_cache, _ = wer(cached, ref)
        w_str_vs_cache, _ = wer(cached, hyp)
        # time deltas of identically-ordered matching words (simple alignment by edit path is overkill:
        # use the common prefix/suffix-free greedy match on equal norm within 1 s)
        deltas, j = [], 0
        for w in off:
            for k in range(j, min(j + 4, len(words))):
                if words[k].norm == w.norm and abs(words[k].start - w.start) < 1.0:
                    deltas.append(words[k].start - w.start)
                    j = k + 1
                    break
        # how much of the text shown at the first tick that covers 8 s changed by the end
        first = next(((T, s) for T, s in shown if T >= 8.0), shown[-1])
        w_first_vs_final, _ = wer(hyp, first[1].split())
        out.append(
            {
                "unit_id": m["unit_id"],
                "duration_s": len(audio) / SR,
                "n_ref_words": len(ref),
                "n_stream_words": len(hyp),
                "wer_offline_vs_streaming_final": w_off_vs_str,
                "edit_errors": errs,
                "wer_cache_vs_offline_rerun": w_off_vs_cache,
                "wer_cache_vs_streaming_final": w_str_vs_cache,
                "matched_words": len(deltas),
                "start_delta_abs_median_s": float(np.median(np.abs(deltas)))
                if deltas
                else None,
                "start_delta_abs_max_s": float(np.max(np.abs(deltas)))
                if deltas
                else None,
                "wer_text_at_first_tick_ge_8s_vs_final": w_first_vs_final,
                "ticks": n_calls,
                "t_offline_s": t_off,
                "t_streaming_s": t_str,
                "offline_text": " ".join(ref),
                "streaming_text": " ".join(hyp),
            }
        )
        print(
            m["unit_id"],
            {
                k: out[-1][k]
                for k in (
                    "wer_offline_vs_streaming_final",
                    "wer_cache_vs_offline_rerun",
                    "t_streaming_s",
                )
            },
            flush=True,
        )
    res = {
        "tick_s": a.tick,
        "model": a.model,
        "environment": asr_environment(),
        "asr": {"device": asr.device, "compute_type": asr.compute_type},
        "clips": out,
        "mean_wer_offline_vs_streaming": float(
            np.mean([c["wer_offline_vs_streaming_final"] for c in out])
        )
        if out
        else None,
        "total_errors": sum(c["edit_errors"] for c in out),
        "total_ref_words": sum(c["n_ref_words"] for c in out),
        "elapsed_s": time.time() - t_start,
    }
    Path(a.out).write_text(json.dumps(res, indent=1, ensure_ascii=False), "utf8")
    print({k: v for k, v in res.items() if k not in ("clips", "environment")})


if __name__ == "__main__":
    main()
