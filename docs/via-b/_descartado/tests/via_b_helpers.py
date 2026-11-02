"""Synthetic builders shared by the Via B tests (no real recordings, no engine output)."""

from __future__ import annotations

import json
import wave
from pathlib import Path

from mrcd.annotation_eval.eaf import Ev, build_eaf, write_eaf
from mrcd.annotation_eval.importer import import_corpus

CORE = ("filler_word", "prolongation", "repetition")
TIER_OF = {
    "filler_word": "filler_word",
    "prolongation": "prolongation",
    "repetition": "repetition",
}


def write_wav(path: Path, seconds: float, sr: int = 16000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(b"\x00\x00" * int(sr * seconds))


def speaker_events(k: int, dur_s: float = 30.0) -> dict[str, list[tuple[float, float]]]:
    """Deterministic, speaker-dependent reference events (seconds); at most 5 per class, never overlapping."""
    ev: dict[str, list[tuple[float, float]]] = {}
    for ci, c in enumerate(CORE):
        n = 5 if c == "filler_word" else 4
        ev[c] = [
            (
                1.0 + 5.0 * i + 0.4 * ci + 0.01 * k,
                1.0 + 5.0 * i + 0.4 * ci + 0.01 * k + 0.30,
            )
            for i in range(n)
        ]
    return ev


def make_adjudicated_corpus(
    tmp: Path,
    n_speakers: int = 12,
    dur_s: float = 30.0,
    practice: bool = False,
    gate: bool = True,
) -> tuple[Path, dict]:
    root = tmp / "root"
    adj, media = root / "adj", root / "media"
    for k in range(1, n_speakers + 1):
        rec = f"VB{'P' if practice else 'T'}_S{k:02d}_PRES1"
        write_wav(media / f"{rec}.wav", dur_s)
        evs = [
            Ev(c, "ADJ", c, int(round(s * 1000)), int(round(e * 1000)))
            for c, ivs in speaker_events(k).items()
            for s, e in ivs
        ]
        tree = build_eaf(
            rec,
            ["ADJ"],
            f"{rec}.wav",
            int(dur_s * 1000),
            events=evs,
            coverage={c: "complete" for c in CORE},
        )
        write_eaf(tree, adj / f"{rec}.eaf")
    agreement = {"classes": {c: {"passes_gate": gate} for c in CORE}}
    out = tmp / "corpus"
    manifest = import_corpus(
        adj, media, out, root=root, agreement=agreement, include_practice=practice
    )
    return out, manifest


def predictions(
    corpus_events: dict[str, dict[str, list[tuple[float, float]]]],
    system: str,
    *,
    keep: float = 1.0,
    extra_fp: int = 0,
    supported=CORE,
    failed: tuple[str, ...] = (),
    shift: float = 0.0,
) -> dict:
    """Predictions from the (synthetic) reference: keep a fraction of events, optionally shifted, plus FPs."""
    recs = {}
    for rec, per in corpus_events.items():
        if rec in failed:
            recs[rec] = {"status": "failed", "events": []}
            continue
        evs = []
        for c, ivs in per.items():
            if c not in supported:
                continue
            n_keep = int(round(len(ivs) * keep))
            for s, e in ivs[:n_keep]:
                evs.append(
                    {"class": c, "start_s": s + shift, "end_s": e + shift, "score": 0.9}
                )
            for i in range(extra_fp):
                evs.append(
                    {"class": c, "start_s": 25.0 + i, "end_s": 25.3 + i, "score": 0.5}
                )
        recs[rec] = {"status": "ok", "events": evs}
    return {
        "system": system,
        "version": "test",
        "supported_classes": list(supported),
        "recordings": recs,
    }


def reference_events(n_speakers: int = 12) -> dict:
    return {f"VBT_S{k:02d}_PRES1": speaker_events(k) for k in range(1, n_speakers + 1)}


def dump(path: Path, obj) -> Path:
    path.write_text(json.dumps(obj), encoding="utf-8")
    return path
