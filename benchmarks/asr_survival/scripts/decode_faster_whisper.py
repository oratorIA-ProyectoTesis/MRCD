"""Decode the frame with faster-whisper under one of the fixed prompt variants (P0, P1, C1, p_none).

Writes ``<out-dir>/<turn_id>.json`` with ``{"turn_id", "variant", "n_samples", "sec", "words": [...]}``,
the input format of ``survival_bench.py --kind words``. Settings are those of the frozen protocol: 8 s chunks with
2 s overlap, beam 5, temperature 0, no conditioning on previous text. Resumable, atomic writes.

You supply the audio: 16 kHz mono PCM WAV files named ``<turn_id>.wav`` (convert the DIHANA ``.ul`` files
yourself; ``scripts/baselines/dihana_decode.py`` in this repository shows how). ``--model`` is a faster-whisper
snapshot directory or a model name known to faster-whisper.

usage: python decode_faster_whisper.py --wav-dir WAVS --model NAME_OR_DIR --variant P0 --out-dir DIR
           [--device cpu|cuda] [--compute-type int8|float16]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import soundfile as sf
from survival_bench import load_frame

from mrcd.asr_recovery.decode import decode_kwargs, transcribe_chunked


class Shim:
    """The two attributes ``transcribe_chunked`` reads from a ``VerbatimASR``."""

    def __init__(self, model):
        self.model = model
        self.language = "es"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav-dir", required=True, type=Path)
    ap.add_argument("--model", required=True)
    ap.add_argument("--variant", required=True, choices=("P0", "P1", "C1", "p_none"))
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--compute-type", default="int8")
    a = ap.parse_args()
    from faster_whisper import WhisperModel

    asr = Shim(WhisperModel(a.model, device=a.device, compute_type=a.compute_type))
    a.out_dir.mkdir(parents=True, exist_ok=True)
    frame = load_frame()
    for k, r in enumerate(frame):
        uid = r["turn_id"]
        p = a.out_dir / f"{uid}.json"
        if p.exists():
            continue
        audio, sr = sf.read(a.wav_dir / f"{uid}.wav", dtype="float32")
        assert sr == 16000 and audio.ndim == 1, f"{uid}: need 16 kHz mono"
        tic = time.perf_counter()
        words = transcribe_chunked(asr, audio, a.variant)
        rec = {
            "turn_id": uid,
            "variant": a.variant,
            "n_samples": int(len(audio)),
            "sec": time.perf_counter() - tic,
            "decode_kwargs": {
                kk: (list(v) if isinstance(v, tuple) else v)
                for kk, v in decode_kwargs(a.variant).items()
            },
            "words": [w.to_dict() for w in words],
        }
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(rec, ensure_ascii=False), "utf8")
        tmp.replace(p)
        if k % 50 == 0:
            print(f"{a.variant} {k + 1}/{len(frame)}", flush=True)


if __name__ == "__main__":
    main()
