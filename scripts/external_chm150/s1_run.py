"""CHM150 step 1 (after the protocol freeze): ASR P0 / P1, V0 engine unit decision, P1 text route, per utterance.

Usage: python -I s1_run.py SHARD NSHARDS     (resumable, atomic writes to OUT/runs/<utt>.json, LOCAL ONLY)
Refuses to run if PROTOCOL.md or any frozen file differs from PROTOCOL.sha256.json.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _pin  # noqa: E402,F401  (MUST be first)

import json  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

from freeze import verify_frozen  # noqa: E402

V0_UNIT_S = (
    3.0  # DIHANA unit recipe (scripts/baselines/run_mrcd.py): pad to one 3 s window
)


def v0_unit(audio, asr, net, norm, labels):
    """Verbatim recipe of scripts/baselines/run_mrcd.py (the DIHANA V0 baseline): pad short units with zeros at the
    END to 3 s, 0.5 s grid + terminal window, decision = any window argmax == class."""
    from mrcd.dataset import acoustic_windows, center_f0, linguistic_windows
    from mrcd.extractors.acoustic import extract_acoustic
    from mrcd.extractors.linguistic import word_features
    from mrcd.extractors.prosody import prosody_windows, speaker_f0_center
    from mrcd.inference import predict

    a, b = 0.0, len(audio) / 16000
    padded = False
    if b - a < V0_UNIT_S - 1e-8:
        need = int(round((a + V0_UNIT_S) * 16000)) - len(audio)
        if need > 0:
            audio = np.concatenate([audio, np.zeros(need, np.float32)])
        b = a + V0_UNIT_S
        padded = True
    starts = np.arange(a, b - 3 + 1e-9, 0.5)
    if b - 3 - starts[-1] > 1e-6:
        starts = np.append(starts, b - 3)
    words = asr.transcribe_chunked(audio, chunk_s=8.0)
    track = extract_acoustic(audio, use_silero=True)
    frames = track.frame_matrix()
    c = speaker_f0_center(frames)
    wd = [w.to_dict() for w in words]
    ling, lpos, lmask = linguistic_windows(
        wd, word_features(words), starts, context=True
    )
    t = {
        "ac": acoustic_windows(center_f0(frames, c), starts),
        "pros": prosody_windows(frames, starts, c),
        "ling": ling[:, :, :12],
        "lpos": lpos,
        "lmask": lmask,
    }
    probs = predict(net, norm, t, "cpu")
    argmax = [labels[int(i)] for i in probs.argmax(1)]
    return {
        "padded": padded,
        "n_windows": len(starts),
        "window_labels": argmax,
        "max_prob": {lab: float(probs[:, labels.index(lab)].max()) for lab in labels},
        "model_pred": {lab: int(lab in argmax) for lab in labels},
        "asr_words": wd,
        "vad": track.vad_backend,
    }


def main(shard: int, nshard: int):
    verify_frozen()
    import ctranslate2
    import torch

    torch.set_num_threads(4)
    from mrcd.asr_profiles import P1_PROMPT as PROFILE_P1_PROMPT
    from mrcd.asr_recovery.decode import P1_PROMPT, decode_kwargs, transcribe_chunked
    from mrcd.extractors.linguistic import VerbatimASR
    from mrcd.inference import load_ckpt
    from mrcd.service import s3_features as S3F

    assert P1_PROMPT == PROFILE_P1_PROMPT, (
        "P1 prompt of the experiment differs from the service profile"
    )
    refs = json.loads((_pin.OUT / "reference.json").read_text("utf8"))
    ids = sorted(refs)
    mine = [u for i, u in enumerate(ids) if i % nshard == shard]
    out_dir = _pin.OUT / "runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    asr = VerbatimASR(str(_pin.WHISPER), device="cpu", compute_type="int8")
    net, norm, labels, _ = load_ckpt(_pin.CKPT, "cpu")
    S3F.load_route()
    t0 = time.perf_counter()
    for k, uid in enumerate(mine):
        p = out_dir / f"{uid}.json"
        if p.exists():
            continue
        audio, sr = sf.read(
            _pin.CORPUS_ROOT / "data" / refs[uid]["audio_relpath"], dtype="float32"
        )
        assert sr == 16000 and audio.ndim == 1
        rec = {
            "utt_id": uid,
            "n_samples": int(len(audio)),
            "omp_num_threads": os.environ["OMP_NUM_THREADS"],
        }
        for v in ("P0", "P1"):
            ctranslate2.set_random_seed(_pin.SEED)
            tic = time.perf_counter()
            words = transcribe_chunked(asr, audio, v)
            rec[f"asr_{v}"] = {
                "sec": time.perf_counter() - tic,
                "words": [w.to_dict() for w in words],
            }
        tic = time.perf_counter()
        ev = S3F.text_route_events(rec["asr_P1"]["words"])
        rec["route_rep_ctx_p1"] = {
            "decision": int(bool(ev)),
            "max_score": max((e["score"] for e in ev), default=None),
            "n_events": len(ev),
            "sec": time.perf_counter() - tic,
        }
        ctranslate2.set_random_seed(_pin.SEED)
        tic = time.perf_counter()
        rec["v0"] = v0_unit(audio, asr, net, norm, labels)
        rec["v0"]["sec"] = time.perf_counter() - tic
        rec["decode_kwargs"] = {
            v: {
                kk: (list(vv) if isinstance(vv, tuple) else vv)
                for kk, vv in decode_kwargs(v).items()
            }
            for v in ("P0", "P1")
        }
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(rec, ensure_ascii=False), "utf8")
        tmp.replace(p)
        if k % 50 == 0:
            print(
                f"shard {shard} {k + 1}/{len(mine)} {time.perf_counter() - t0:.0f}s",
                flush=True,
            )
    print("DONE", shard, flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]), int(sys.argv[2]))
