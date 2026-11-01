"""Functioning check (NOT an evaluation): V0 on the utt74 WAV, printed next to the reference event.

Same V0 recipe as scripts/baselines/run_mrcd.py (zero-pad to 3 s if shorter, 0.5 s grid + terminal window,
whisper small int8 chunked ASR, rule 'any window argmax == class'). n=1, one segment of dialogue d12.
Usage: python scripts/rep_rev/utt74_check.py
"""

import os

os.environ["OMP_NUM_THREADS"] = "4"
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import hashlib  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

REPO = Path(r"C:\zArnes-tesis\mrcd")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "baselines"))
import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

WAV = (
    REPO
    / "material"
    / "MRCD_Material_4_Corpus_20261030"
    / "DIME"
    / "muestras"
    / "utt74.wav"
)
REF = (
    REPO
    / "results"
    / "ejecucion_material4_20261030"
    / "importadores"
    / "utt74_example.json"
)
OUT = (
    REPO
    / "results"
    / "ejecucion_material4_20261030"
    / "experimentos"
    / "utt74_check.json"
)


def main():
    import torch
    from bl_common import CKPT, WHISPER

    from mrcd.annotation.auto_labeler import (
        detect_fillers,
        detect_prolongations,
        detect_repetitions,
        detect_revisions,
    )
    from mrcd.dataset import acoustic_windows, center_f0, linguistic_windows
    from mrcd.extractors.acoustic import extract_acoustic
    from mrcd.extractors.linguistic import VerbatimASR, word_features
    from mrcd.extractors.prosody import prosody_windows, speaker_f0_center
    from mrcd.inference import load_ckpt, predict

    torch.set_num_threads(4)
    audio, sr = sf.read(str(WAV), dtype="float32")
    assert sr == 16000 and audio.ndim == 1
    dur = len(audio) / sr
    ref = json.loads(REF.read_text("utf8"))
    ev = [a for a in ref["original_annotations"] if a["proposed_class"] == "revision"][
        0
    ]
    a, b, padded = 0.0, dur, False
    if b - a < 3 - 1e-8:
        audio = np.concatenate(
            [audio, np.zeros(int(round(3 * sr)) - len(audio), np.float32)]
        )
        b, padded = 3.0, True
    starts = np.arange(a, b - 3 + 1e-9, 0.5)
    if b - 3 - starts[-1] > 1e-6:
        starts = np.append(starts, b - 3)
    net, norm, labels, _ = load_ckpt(CKPT, "cpu")
    asr = VerbatimASR(str(WHISPER), device="cpu", compute_type="int8")
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
    fil = detect_fillers(track, words, "es")
    rules = {
        "filler_word": fil,
        "prolongation": detect_prolongations(track, words, "es", fil),
        "repetition": detect_repetitions(words),
        "revision": detect_revisions(words),
    }
    out = {
        "NOT_AN_EVALUATION": "route check, n=1, segment of dialogue d12",
        "wav_sha256": hashlib.sha256(WAV.read_bytes()).hexdigest(),
        "duration_s": dur,
        "zero_padded_to_3s": padded,
        "reference_event": {
            "class": ev["proposed_class"],
            "start_s": ev["start_s"],
            "end_s": ev["end_s"],
            "reference_text": ref["original_text"],
        },
        "asr_words": [
            (w["text"], round(w["start"], 2), round(w["end"], 2)) for w in wd
        ],
        "windows": [
            {
                "start": float(s),
                "end": float(s) + 3.0,
                "argmax": labels[int(p.argmax())],
                "probs": {l: round(float(p[i]), 3) for i, l in enumerate(labels)},
            }
            for s, p in zip(starts, probs)
        ],
        "v0_unit_decision": {
            l: int(l in {labels[int(i)] for i in probs.argmax(1)})
            for l in labels
            if l != "fluent"
        },
        "rule_detectors": {
            k: [
                (
                    round(float(x.start), 2),
                    round(float(x.end), 2),
                    getattr(x, "label", ""),
                )
                for x in v
            ]
            for k, v in rules.items()
        },
        "OMP_NUM_THREADS": os.environ["OMP_NUM_THREADS"],
        "reading": "V0 windows are 3 s long and the unit is 3.5 s or shorter: time localisation of the 0.538-1.813 s event is not available from window labels.",
    }
    OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf8")
    print(
        json.dumps(
            {
                k: out[k]
                for k in (
                    "duration_s",
                    "zero_padded_to_3s",
                    "reference_event",
                    "asr_words",
                    "v0_unit_decision",
                    "rule_detectors",
                )
            },
            ensure_ascii=False,
        )
    )
    for w in out["windows"]:
        print(w["start"], w["argmax"], w["probs"])


if __name__ == "__main__":
    main()
