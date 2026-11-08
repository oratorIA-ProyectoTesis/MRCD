"""Plan C arms HF_full and HF_greedy (PROTOCOL_EXPLORATORY_C.md section 3). EXPLORATORY, DIHANA folds 1-4.

Env: <venv_cw2> (crisperwhisper 2.0.3, transformers backend, CUDA float16). Resumable: one JSON per unit
under results/plan_c_20261108/cache/<arm>/ (git-ignored). Also writes cache/decoder_spec.json (prompt ids, suppress list,
special ids) so that the CT2 arms use exactly the same decoder prompt.

usage: HF_HUB_OFFLINE=1 python scripts/plan_c/decode_hf.py
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
REV = "f4334f6e8193f2691212d49b20fa12d370e13896"
SNAP = (
    Path.home()
    / ".cache/huggingface/hub/models--nyralabs--CrisperWhisper2.0_large/snapshots"
    / REV
)
KW = dict(language="es", mode="verbatim", word_timestamps=False, sr=16000)
MAX_NEW = 256
N_WARMUP, N_DET = 2, 8


def write(p: Path, rec: dict) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(rec, ensure_ascii=False), "utf8")
    tmp.replace(p)


def main() -> None:
    import torch
    from crisperwhisper import CrisperWhisperModel
    from crisperwhisper.prompt import PromptBuilder

    units = json.loads((CACHE / "sample.json").read_text("utf8"))["units"]
    dio.assert_folds(u["fold"] for u in units)
    turns = dio.load_turns()
    audio = {u["unit_id"]: dio.load_audio(turns[u["unit_id"]]) for u in units}

    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    m = CrisperWhisperModel(
        str(SNAP), backend="transformers", device="cuda", compute_type="float16"
    )
    load_s = time.perf_counter() - t0
    eng = m._engine
    prompt = PromptBuilder(eng, language="es").verbatim()
    spec = {
        "prompt_ids": [int(t) for t in prompt],
        "suppress_tokens": [int(t) for t in eng.default_suppress_tokens],
        "all_special_ids": sorted(int(t) for t in eng.all_special_ids),
        "eot_id": int(eng.eot_id),
        "max_new_tokens": MAX_NEW,
        "n_mels": int(eng.n_mels),
    }
    write(CACHE / "decoder_spec.json", spec)

    def full(a):
        torch.manual_seed(13)
        return m.transcribe(a, **KW).text

    def greedy(a):
        torch.manual_seed(13)
        feats = eng.extract_features(a)
        ids = eng.generate(feats, [prompt], max_length=MAX_NEW, beam_size=1)[0]
        return eng.decode_tokens(ids)

    arms = {"HF_full": full, "HF_greedy": greedy}
    meta = {
        "load_s": load_s,
        "torch": torch.__version__,
        "device": torch.cuda.get_device_name(0),
        "arms": {},
    }
    for arm, fn in arms.items():
        for u in units[:N_WARMUP]:  # untimed warm-up
            fn(audio[u["unit_id"]])
        torch.cuda.synchronize()
        for u in units:
            p = CACHE / arm / f"{u['unit_id']}.json"
            if p.exists():
                continue
            a = audio[u["unit_id"]]
            tic = time.perf_counter()
            text = fn(a)
            torch.cuda.synchronize()
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
            fn(audio[u["unit_id"]])
            == json.loads((CACHE / arm / f"{u['unit_id']}.json").read_text("utf8"))[
                "text"
            ]
            for u in units[:N_DET]
        )
        meta["arms"][arm] = {"determinism_identical": int(det), "determinism_n": N_DET}
        print(arm, "done", flush=True)
    meta["peak_gpu_mem_gb"] = torch.cuda.max_memory_allocated() / 1e9
    write(CACHE / "meta_hf.json", meta)
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
