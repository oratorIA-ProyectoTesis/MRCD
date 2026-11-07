"""Second-corpus CrisperWhisper 2.0 large decode (ADDENDUM section 4).

Same options as decode_crisper.py (verbatim, es, GPU float16, torch seed 13).
usage (venv /path/to/_venv_crisper): python decode_chm150_crisper.py
"""

import json
import os
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
REPO = Path(__file__).resolve().parents[2]
DEST = REPO / "results" / "asr_survival_robustness_20261104"
CORPUS = REPO / "data" / "external" / "CHM150" / "CHM150"
MODEL = "nyralabs/CrisperWhisper2.0_large"
KW = dict(language="es", mode="verbatim", word_timestamps=False, sr=16000)

import soundfile as sf  # noqa: E402
import torch  # noqa: E402
from crisperwhisper import CrisperWhisperModel  # noqa: E402


def main():
    ids = json.loads((DEST / "chm_frame.json").read_text("utf8"))["ids"]
    refs = json.loads(
        (REPO / "results/external_chm150_20261102/reference.json").read_text("utf8")
    )
    assert len(ids) == 500
    m = CrisperWhisperModel(
        MODEL, backend="transformers", device="cuda", compute_type="float16"
    )
    d = DEST / "runs" / "chm150" / "cw_large"
    d.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    for k, uid in enumerate(ids):
        p = d / f"{uid}.json"
        if p.exists():
            continue
        audio, sr = sf.read(
            CORPUS / "data" / refs[uid]["audio_relpath"], dtype="float32"
        )
        assert sr == 16000 and audio.ndim == 1
        torch.manual_seed(13)
        tic = time.perf_counter()
        r = m.transcribe(audio, **KW)
        rec = {
            "utt_id": uid,
            "model": MODEL,
            "n_samples": int(len(audio)),
            "sec": time.perf_counter() - tic,
            "decode_kwargs": KW,
            "text": r.text,
        }
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(rec, ensure_ascii=False), "utf8")
        tmp.replace(p)
        if k % 50 == 0:
            print(
                f"cw_large chm {k + 1}/{len(ids)} {time.perf_counter() - t0:.0f}s",
                flush=True,
            )
    print(f"DONE cw_large chm {time.perf_counter() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
