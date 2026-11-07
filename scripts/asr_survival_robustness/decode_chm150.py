"""Second-corpus decodes with faster-whisper (ADDENDUM section 4): no-prompt variant on the CHM150 sub-frame.

usage (mrcd venv): python decode_chm150.py small_pnone   # CPU int8, OMP 4, Systran small
                   python decode_chm150.py lv3_pnone     # GPU float16, Systran large-v3
Same settings as the frozen CHM150 protocol (decode_kwargs('p_none'), 8 s chunking, ctranslate2 seed 13).
"""

import os
import sys

if "numpy" in sys.modules:
    raise SystemExit("numpy imported before the OMP pin")
os.environ["OMP_NUM_THREADS"] = "4"
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
_NV = REPO.parent / "_venv_gpu" / "Lib" / "site-packages" / "nvidia"
for sub in ("cublas", "cudnn", "cuda_nvrtc"):
    d = _NV / sub / "bin"
    if d.is_dir():
        os.add_dll_directory(str(d))
        os.environ["PATH"] = str(d) + os.pathsep + os.environ["PATH"]
import soundfile as sf  # noqa: E402

from mrcd.asr_recovery.decode import transcribe_chunked  # noqa: E402

HUB = Path.home() / ".cache/huggingface/hub"
SNAPS = {
    "small_pnone": (
        HUB
        / "models--Systran--faster-whisper-small/snapshots/536b0662742c02347bc0e980a01041f333bce120",
        "cpu",
        "int8",
    ),
    "lv3_pnone": (
        HUB
        / "models--Systran--faster-whisper-large-v3/snapshots/edaa852ec7e145841d8ffdb056a99866b5f0a478",
        "cuda",
        "float16",
    ),
}
DEST = REPO / "results" / "asr_survival_robustness_20261104"
CORPUS = REPO / "data" / "external" / "CHM150" / "CHM150"


class Shim:
    def __init__(self, model):
        self.model, self.language = model, "es"


def main(system):
    import ctranslate2
    from faster_whisper import WhisperModel

    snap, dev, ct = SNAPS[system]
    assert snap.is_dir(), snap
    ids = json.loads((DEST / "chm_frame.json").read_text("utf8"))["ids"]
    refs = json.loads(
        (REPO / "results/external_chm150_20261102/reference.json").read_text("utf8")
    )
    assert len(ids) == 500
    asr = Shim(WhisperModel(str(snap), device=dev, compute_type=ct))
    d = DEST / "runs" / "chm150" / system
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
        ctranslate2.set_random_seed(13)
        tic = time.perf_counter()
        words = transcribe_chunked(asr, audio, "p_none")
        rec = {
            "utt_id": uid,
            "model": f"{system} {dev} {ct}",
            "n_samples": int(len(audio)),
            "sec": time.perf_counter() - tic,
            "words": [w.to_dict() for w in words],
        }
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(rec, ensure_ascii=False), "utf8")
        tmp.replace(p)
        if k % 50 == 0:
            print(
                f"{system} {k + 1}/{len(ids)} {time.perf_counter() - t0:.0f}s",
                flush=True,
            )
    print(f"DONE {system} {time.perf_counter() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
