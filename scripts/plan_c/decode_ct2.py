"""Plan C arms CT2_fp16 and CT2_int8f16 (PROTOCOL_EXPLORATORY_C.md section 3). EXPLORATORY, DIHANA folds 1-4.

Upstream ctranslate2 ``Whisper.generate`` with the SAME decoder prompt, suppress list and token budget as the HF arms
(read from cache/decoder_spec.json written by decode_hf.py), the same HF ``WhisperFeatureExtractor`` features and the
same transformers tokenizer for decoding, so that only the runtime/conversion differs from HF_greedy. The generate call
mirrors ``crisperwhisper.engine.CT2Engine.generate`` (max_length budget, suppress_blank=False).

Env: <venv_cw2> with the CUDA 12 libraries of <venv_gpu> on PATH (ctranslate2 4.8.2 is a
CUDA 12 build). usage: python scripts/plan_c/decode_ct2.py CT2_fp16|CT2_int8f16
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "verbatim_ft"))
import dihana_io as dio  # noqa: E402

OUT = REPO / "results" / "plan_c_20261108"
CACHE = OUT / "cache"
MODELS = Path(os.environ.get("MRCD_CT2_MODELS") or Path(__file__).resolve().parents[2].parent / "_models_ct2")
ARMS = {
    "CT2_fp16": ("cw2_fp16", "float16"),
    "CT2_int8f16": ("cw2_int8f16", "int8_float16"),
}
SNAP = (
    Path.home()
    / ".cache/huggingface/hub/models--nyralabs--CrisperWhisper2.0_large/snapshots"
    / "f4334f6e8193f2691212d49b20fa12d370e13896"
)
N_WARMUP, N_DET = 2, 8


def write(p: Path, rec: dict) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(rec, ensure_ascii=False), "utf8")
    tmp.replace(p)


def main(arm: str) -> None:
    import ctranslate2
    import numpy as np
    from transformers import AutoTokenizer, WhisperFeatureExtractor

    sub, ct = ARMS[arm]
    spec = json.loads((CACHE / "decoder_spec.json").read_text("utf8"))
    units = json.loads((CACHE / "sample.json").read_text("utf8"))["units"]
    dio.assert_folds(u["fold"] for u in units)
    turns = dio.load_turns()
    audio = {u["unit_id"]: dio.load_audio(turns[u["unit_id"]]) for u in units}
    fe = WhisperFeatureExtractor.from_pretrained(str(SNAP))
    tok = AutoTokenizer.from_pretrained(str(SNAP))
    special = set(spec["all_special_ids"])
    prompt = spec["prompt_ids"]
    ml = max(spec["max_new_tokens"] * 2, spec["max_new_tokens"] + len(prompt) - 1)

    t0 = time.perf_counter()
    model = ctranslate2.models.Whisper(
        str(MODELS / sub), device="cuda", compute_type=ct
    )
    load_s = time.perf_counter() - t0

    def run(a):
        f = fe(a, sampling_rate=dio.SR, return_tensors="np").input_features.astype(
            np.float32
        )
        sv = ctranslate2.StorageView.from_array(np.ascontiguousarray(f))
        r = model.generate(
            sv,
            [prompt],
            max_length=ml,
            beam_size=1,
            suppress_blank=False,
            suppress_tokens=spec["suppress_tokens"],
            return_scores=False,
            return_no_speech_prob=False,
        )
        ids = [int(t) for t in r[0].sequences_ids[0] if int(t) not in special]
        return tok.decode(ids)

    for u in units[:N_WARMUP]:
        run(audio[u["unit_id"]])
    for u in units:
        p = CACHE / arm / f"{u['unit_id']}.json"
        if p.exists():
            continue
        a = audio[u["unit_id"]]
        tic = time.perf_counter()
        text = run(a)
        write(
            p,
            {
                "unit_id": u["unit_id"],
                "arm": arm,
                "sec": time.perf_counter() - tic,
                "audio_s": len(a) / dio.SR,
                "text": text,
            },
        )
    det = sum(
        run(audio[u["unit_id"]])
        == json.loads((CACHE / arm / f"{u['unit_id']}.json").read_text("utf8"))["text"]
        for u in units[:N_DET]
    )
    meta = {
        "arm": arm,
        "compute_type": ct,
        "model_dir": str(MODELS / sub),
        "load_s": load_s,
        "ctranslate2": ctranslate2.__version__,
        "determinism_identical": int(det),
        "determinism_n": N_DET,
    }
    write(CACHE / f"meta_{arm}.json", meta)
    print(json.dumps(meta))


if __name__ == "__main__":
    main(sys.argv[1])
