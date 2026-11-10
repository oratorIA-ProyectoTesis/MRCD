"""Plan F: engine unit features of the validation turns recomputed with the FT words (resumable, atomic writes).

Needed only for the DESCRIPTIVE prolongation reading of the protocol (section 3/4): the shipped ``o2026_11`` prolongation head,
unchanged, fed with ``DisfluencyEngine.unit_features(audio, ft_words)`` (the service path). Run after ``decode_ft.py`` has finished
(one heavy job at a time). Writes ``cache/feat_ft/<uid>.npz`` (git-ignored). Computes no metric.
"""

from __future__ import annotations

import io
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402


def main() -> None:
    C.protocol_sha()
    import numpy as np
    import soundfile as sf

    from mrcd.engine import DisfluencyEngine

    rows = C.val_pool()
    d = C.CACHE / "feat_ft"
    d.mkdir(parents=True, exist_ok=True)
    todo = [u["unit_id"] for u in rows if not (d / f"{u['unit_id']}.npz").exists()]
    print(f"{len(rows)} turns, {len(todo)} to do", flush=True)
    if not todo:
        print("DONE", flush=True)
        return
    # the engine ASR is irrelevant here (words are given); p0 is the cheapest valid profile to construct
    eng = DisfluencyEngine.load(
        C.WT / "models" / "ckpt_audio_text.pt", device="cpu", asr_profile="p0"
    )
    t0 = time.perf_counter()
    for k, uid in enumerate(todo):
        audio, sr = sf.read(C.wav_path(uid), dtype="float32")
        assert sr == C.SR and audio.ndim == 1
        f = eng.unit_features(audio, C.ft_words(uid))
        import json

        buf = io.BytesIO()
        np.savez(
            buf,
            probs=np.asarray(f["probs"], np.float32),
            pooled=np.asarray(f["pooled"], np.float32),
            pause=np.frombuffer(json.dumps(f["pause"]).encode("utf8"), np.uint8),
            n_samples=np.asarray([int(f["n_samples"])]),
        )
        tmp = d / f"{uid}.tmp"
        tmp.write_bytes(buf.getvalue())
        tmp.replace(d / f"{uid}.npz")
        if k % 50 == 0:
            print(f"{k + 1}/{len(todo)} {time.perf_counter() - t0:.0f}s", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
