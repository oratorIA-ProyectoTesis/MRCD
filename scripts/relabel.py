#!/usr/bin/env python3
"""Re-etiqueta desde las señales YA extraídas (sin repetir ASR ni MediaPipe).

Úsalo cuando cambian las reglas de core/annotation/auto_labeler.py: reconstruye las pistas
desde data/features/*.npz + *.words.json, regenera events.json, el .eaf y el manifest.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.annotation.auto_labeler import label_recording, window_labels, write_eaf  # noqa: E402
from core.constants import ACOUSTIC_HOP_S, LABELS, TAXONOMY  # noqa: E402
from core.contracts import validate_raw_features  # noqa: E402
from core.extractors.acoustic import AcousticTrack, _complement, _mask_to_segments  # noqa: E402
from core.extractors.linguistic import Word  # noqa: E402

MANIFEST = ROOT / "data/dataset_manifest.json"


class _Kin:
    def __init__(self, t, v, d):
        self.times, self.vectors, self.detected = t, v, d
        self.detection_rate = float(d.mean()) if len(d) else 0.0


def rebuild(rid_npz: Path, duration_s: float, wav: Path | None = None) -> tuple[AcousticTrack, _Kin | None]:
    z = np.load(rid_npz, allow_pickle=False)
    validate_raw_features(z, path=str(rid_npz))
    times, f0, rms, vad = z["ac_times"], z["f0"], z["rms"], z["vad"]
    frames = z["ac_frames"]
    if "hf_ratio" in z:
        hf = z["hf_ratio"]
    elif wav is not None and wav.exists():          # recomputar solo la energía HF (barato)
        import librosa
        from core.extractors.acoustic import load_wav
        audio, sr = load_wav(str(wav))
        hop = int(sr * ACOUSTIC_HOP_S)
        S = np.abs(librosa.stft(audio, n_fft=512, hop_length=hop, center=True)) ** 2
        fr = librosa.fft_frequencies(sr=sr, n_fft=512)
        hf = (S[fr >= 3000].sum(0) / (S.sum(0) + 1e-10)).astype(np.float32)[: len(times)]
        hf = np.pad(hf, (0, max(0, len(times) - len(hf))))
    else:
        hf = None
    speech = _mask_to_segments(vad == 1, times, min_len=0.0, merge_gap=0.0)
    ac = AcousticTrack(times, rms, f0, frames[:, 2], vad, speech, _complement(speech, duration_s), "cached", hf)
    kin = _Kin(z["k_times"], z["k_vectors"], z["k_detected"]) if len(z["k_times"]) else None
    return ac, kin


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variety", default="es-PE")
    ap.add_argument("--recall-mode", action="store_true", default=True)
    a = ap.parse_args()
    man = json.loads(MANIFEST.read_text(encoding="utf-8"))
    for r in man["recordings"]:
        rid = r["recording_id"]
        ac, kin = rebuild(ROOT / r["features_npz"], r["duration_s"], ROOT / r["wav"])
        words = [Word(**w) for w in json.loads((ROOT / r["words_json"]).read_text(encoding="utf-8"))]
        cands = label_recording(ac, words, kin, a.variety, recall_mode=a.recall_mode)
        (ROOT / r["events_json"]).write_text(json.dumps([c.to_dict() for c in cands], ensure_ascii=False, default=float),
                                             encoding="utf-8")
        media = {str(ROOT / r["wav"]): "audio/x-wav"}
        if r.get("mp4"):
            media[str(ROOT / r["mp4"])] = "video/mp4"
        write_eaf(ROOT / r["eaf"], cands, words, ac.silence_segments, media)
        rows = window_labels(cands, r["duration_s"])
        r["event_counts"] = {c: sum(x.category == c and x.source == "auto" for x in cands) for c in TAXONOMY}
        r["low_conf_counts"] = {c: sum(x.category == c and x.source != "auto" for x in cands) for c in TAXONOMY}
        r["window_primary_counts"] = {l: sum(w["primary"] == l for w in rows) for l in LABELS}
        print(f"[{rid}] {r['event_counts']}")
    MANIFEST.write_text(json.dumps(man, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    tot = {c: sum(r["event_counts"][c] for r in man["recordings"]) for c in TAXONOMY}
    print("TOTAL:", tot)


if __name__ == "__main__":
    main()
