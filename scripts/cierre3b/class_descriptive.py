"""DESCRIPTIVE only: how the MRCD window argmax (8 checkpoint classes) changes with the ASR path (Task 3.2).
No accuracy claim: there is no human label for these windows. Same acoustic frames for every condition; only the ASR words change.
usage: python class_descriptive.py --pred DIR --subset ids.json --out FILE.json
"""

import sys
import pathlib as _p

sys.path.insert(0, str(_p.Path(__file__).resolve().parents[1]))
import _pin_omp  # noqa: F401,E402
import argparse
import collections
import json  # noqa: E402
from pathlib import Path  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
from mrcd.constants import STRIDE_S, WINDOW_S  # noqa: E402
from mrcd.dataset import acoustic_windows, center_f0, linguistic_windows  # noqa: E402
from mrcd.extractors.acoustic import extract_acoustic  # noqa: E402
from mrcd.extractors.linguistic import Word, word_features  # noqa: E402
from mrcd.extractors.prosody import prosody_windows, speaker_f0_center  # noqa: E402
from mrcd.inference import load_ckpt, predict  # noqa: E402

CONDS = ["off_p0", "str_p0_t2", "str_p0_t2_ca", "off_p1", "str_p1_t2"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--subset", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ids = json.loads(Path(a.subset).read_text("utf8"))
    units = {
        u["unit_id"]: u
        for u in json.loads((Path(a.pred).parent / "units.json").read_text("utf8"))
    }
    net, norm, labels, _ = load_ckpt(REPO / "models/ckpt_audio_text.pt", "cpu")
    ling_dim = net.ling.proj[0].in_features
    counts = {c: collections.Counter() for c in CONDS}
    flips = {c: [0, 0] for c in CONDS}
    for uid in ids:
        audio, _ = sf.read(units[uid]["wav"], dtype="float32")
        frames = extract_acoustic(audio, use_silero=True).frame_matrix()
        center = speaker_f0_center(frames)
        starts = np.arange(0.0, max(len(audio) / 16000 - WINDOW_S, 0) + 1e-9, STRIDE_S)
        pred = {}
        for c in CONDS:
            p = Path(a.pred) / c / f"{uid}.json"
            if not p.exists():
                continue
            words = [Word(**w) for w in json.loads(p.read_text("utf8"))["words"]]
            ling, lpos, lmask = linguistic_windows(
                [w.to_dict() for w in words],
                word_features(words),
                starts,
                context=ling_dim > 10,
            )
            t = {
                "ac": acoustic_windows(center_f0(frames, center), starts),
                "ling": ling[:, :, :ling_dim],
                "lpos": lpos,
                "lmask": lmask,
                "pros": prosody_windows(frames, starts, center),
            }
            pred[c] = predict(net, norm, t, "cpu", bs=64).argmax(1)
        for c, v in pred.items():
            counts[c].update(labels[i] for i in v)
            if c != "off_p0" and "off_p0" in pred:
                flips[c][0] += int((v != pred["off_p0"]).sum())
                flips[c][1] += len(v)
    out = {
        "note": "descriptive; no human window labels; counts of window argmax per class; flips = windows whose argmax differs from off_p0",
        "labels": list(labels),
        "counts": {c: dict(v) for c, v in counts.items()},
        "flip_vs_off_p0": {
            c: {"flipped": f[0], "windows": f[1], "rate": f[0] / f[1] if f[1] else None}
            for c, f in flips.items()
        },
    }
    Path(a.out).write_text(json.dumps(out, indent=1), "utf8")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
