#!/usr/bin/env python3
"""Prueba de humo END-TO-END sobre un corpus SINTÉTICO con verdad conocida.

Objetivo: verificar que extract_and_label -> manifest/EAF -> build_windows -> run_ablation
corre completo y que el benchmark es capaz de DETECTAR una ganancia multimodal cuando
existe por construcción: aquí, tras una palabra funcional, la pausa es 'block' si hay
tensión perioral y 'neutral_pause' si no (audio y texto idénticos en ambos casos).
No produce evidencia sobre datos reales.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import scripts.extract_and_label as xl  # noqa: E402
from core.constants import KIDX, KINESIC_DIM  # noqa: E402
from core.extractors.linguistic import Word, normalize  # noqa: E402
from tests.synth import silence, voiced  # noqa: E402

SR = 16000
FUNC = ["la", "de", "el", "en", "con", "y"]
CONTENT = ["casa", "tema", "proyecto", "idea", "datos", "equipo", "modelo", "problema", "ciudad", "gente"]


def make_recording(seed: int):
    rng = np.random.default_rng(seed)
    base = rng.uniform(95, 190)
    audio, words, truth, t = [silence(0.4, seed=seed)], [], [], 0.4
    tense_spans = []

    def say(txt, dur, a, b):
        nonlocal t
        audio.append(voiced(dur, a, b, seed=int(rng.integers(1e6)))); 
        words.append(Word(txt, normalize(txt), round(t, 3), round(t + dur, 3), 0.95, txt[-1] if txt[-1] in ".," else ""))
        t += dur

    def pause(dur):
        nonlocal t
        audio.append(silence(dur, seed=int(rng.integers(1e6)))); s = t; t += dur; return s, t

    for _ in range(14):
        for _ in range(int(rng.integers(2, 4))):
            say(rng.choice(CONTENT), rng.uniform(0.3, 0.5), base * rng.uniform(1, 1.15), base * rng.uniform(0.95, 1.1))
        kind = rng.choice(["block", "neutral", "rhet", "filler", "none"], p=[.3, .3, .15, .15, .1])
        if kind in ("block", "neutral"):
            say(rng.choice(FUNC), 0.2, base, base * 0.98)
            a, b = pause(rng.uniform(0.7, 1.2))
            truth.append(("block" if kind == "block" else "neutral_pause", a, b))
            if kind == "block":
                tense_spans.append((a, b))
        elif kind == "rhet":
            say(rng.choice(CONTENT) + ".", 0.45, base * 1.05, base * 0.75)
            a, b = pause(rng.uniform(0.9, 1.6)); truth.append(("rhetorical_pause", a, b))
        elif kind == "filler":
            s0 = t; say("eh,", rng.uniform(0.35, 0.5), base, base); truth.append(("filler_word", s0, t))
        say(rng.choice(CONTENT), 0.4, base, base * 0.95)
        pause(0.2)
    audio.append(silence(0.8)); t += 0.8
    audio = np.concatenate(audio)
    kt = np.arange(0, len(audio) / SR, 0.1)
    kv = np.zeros((len(kt), KINESIC_DIM), np.float32)
    kv[:, 0:3] = rng.normal(0, 0.8, (len(kt), 3))
    kv[:, KIDX["jaw_open"]] = rng.uniform(0.1, 0.5, len(kt))
    kv[:, [KIDX["mouth_press_left"], KIDX["mouth_press_right"]]] = rng.uniform(0.02, 0.12, (len(kt), 1))
    for a, b in tense_spans:
        m = (kt >= a) & (kt <= b)
        kv[m, KIDX["mouth_press_left"]] = kv[m, KIDX["mouth_press_right"]] = rng.uniform(0.45, 0.7)
        kv[m, KIDX["brow_down_avg"]] = rng.uniform(0.4, 0.6)
    return audio, words, (kt, kv), truth


class FakeASR:
    def __init__(self): self.words = []
    def transcribe(self, audio): return self.words


class FakeKx:
    def __init__(self): self.track = None
    def process_video(self, path): return self.track


def main(n_speakers: int = 6):
    seg_dir = ROOT / "data/segments"; seg_dir.mkdir(parents=True, exist_ok=True)
    asr, kx = FakeASR(), FakeKx()
    manifest = {"recordings": []}
    for s in range(n_speakers):
        audio, words, (kt, kv), truth = make_recording(100 + s)
        rid = f"synth_spk{s:02d}"
        sf.write(seg_dir / f"{rid}.wav", audio, SR, subtype="PCM_16")
        rec = dict(recording_id=rid, speaker_id=rid, wav=f"data/segments/{rid}.wav", mp4=f"data/segments/{rid}.mp4",
                   duration_s=len(audio) / SR, license="synthetic")

        class T:
            times, vectors, detected = kt, kv, np.ones(len(kt), bool)
            detection_rate = 1.0
        asr.words, kx.track = words, T
        entry = xl.process(rec, asr, kx, "es-PE")
        # reemplazar eventos auto por la VERDAD sintética (evita la circularidad en esta prueba)
        ev = [dict(category=c, start_ms=int(a * 1000), end_ms=int(b * 1000), confidence=1.0, source="synthetic_truth")
              for c, a, b in truth]
        (ROOT / entry["events_json"]).write_text(json.dumps(ev), encoding="utf-8")
        manifest["recordings"].append(entry)
    (ROOT / "data/dataset_manifest.json").write_text(json.dumps(manifest, indent=2, default=float), encoding="utf-8")
    print("corpus sintético listo")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 6)
