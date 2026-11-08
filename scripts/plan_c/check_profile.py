"""Post-hoc check of the opt-in profile ``cw2_ct2`` (added after RESULTS_C acceptance). EXPLORATORY, DIHANA folds 1-4.

Runs ``VerbatimASR(profile="cw2_ct2")`` on the 48 Plan C units and compares its word sequence (normalised, filled-pause
tags mapped as in profile ``cw2``) with the HF_full output passed through the same mapping. Also a smoke run on one
synthetic long input (concatenated sample turns, > 30 s) to exercise the windowing; no metric is computed on it.
Writes cache/profile_check.json (aggregates only).

Env: repo .venv + CUDA 12 libraries on PATH; MRCD_CW2_CT2_DIR=<ct2_models_dir>/cw2_fp16.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "verbatim_ft"))
import dihana_io as dio  # noqa: E402

from mrcd import verbatim  # noqa: E402
from mrcd.extractors.linguistic import VerbatimASR, normalize  # noqa: E402

CACHE = REPO / "results" / "plan_c_20261108" / "cache"


def ref_norm(raw: str) -> list[str]:
    toks, _ = verbatim.map_tags(raw)
    return [n for n in (normalize(t) for t in toks) if n]


def main() -> None:
    units = json.loads((CACHE / "sample.json").read_text("utf8"))["units"]
    dio.assert_folds(u["fold"] for u in units)
    turns = dio.load_turns()
    audio = {u["unit_id"]: dio.load_audio(turns[u["unit_id"]]) for u in units}
    t0 = time.perf_counter()
    asr = VerbatimASR(profile="cw2_ct2")
    load_s = time.perf_counter() - t0
    for u in units[:2]:
        asr.transcribe(audio[u["unit_id"]])
    same, sec, aud, mono = 0, 0.0, 0.0, 0
    for u in units:
        a = audio[u["unit_id"]]
        tic = time.perf_counter()
        words = asr.transcribe_chunked(a, chunk_s=8.0)
        sec += time.perf_counter() - tic
        aud += len(a) / dio.SR
        hf = json.loads((CACHE / "HF_full" / f"{u['unit_id']}.json").read_text("utf8"))[
            "text"
        ]
        same += [w.norm for w in words] == ref_norm(hf)
        mono += all(w.start <= w.end for w in words) and all(
            x.start <= y.start + 1e-6 for x, y in zip(words, words[1:])
        )
    long_audio = np.concatenate([audio[u["unit_id"]] for u in units[:12]])
    tic = time.perf_counter()
    lw = asr.transcribe_chunked(long_audio, chunk_s=8.0)
    out = {
        "profile": "cw2_ct2",
        "describe": asr.profile.describe(),
        "load_s": load_s,
        "n": len(units),
        "word_sequence_identical_to_HF_full": int(same),
        "monotonic_times": int(mono),
        "rtf_sum": sec / aud,
        "long_smoke": {
            "audio_s": len(long_audio) / dio.SR,
            "sec": time.perf_counter() - tic,
            "n_words": len(lw),
            "max_end_s": max((w.end for w in lw), default=0.0),
        },
        "dropped_tags_total": len(asr.dropped_tags),
    }
    (CACHE / "profile_check.json").write_text(json.dumps(out, indent=1), "utf8")
    print(json.dumps({k: v for k, v in out.items() if k != "describe"}))


if __name__ == "__main__":
    main()
