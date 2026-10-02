"""MRCD analysis use case: prepare a recording once, then window, predict and decode.

Extraction (ASR + acoustic track) runs at most once per input/configuration and is
shared by every window and by the rule baseline. Output times are integer
milliseconds from the start of the analysed audio, intervals are [start, end).
"""

from __future__ import annotations

import hashlib
import json
import pickle
import platform
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from core.constants import BACKGROUND, CONTEXT_S, SAMPLE_RATE, SMOKE_HEADER, STRIDE_S, WINDOW_S
from core.dataset import acoustic_windows, center_f0, linguistic_windows, window_starts
from core.extractors.acoustic import AcousticTrack, extract_acoustic
from core.extractors.linguistic import Word, word_features
from core.extractors.prosody import prosody_windows, speaker_f0_center
from core.live.streaming_asr import dedupe_words

ENGINE_VERSION = "engine-1.0.0"
PAUSES = ("rhetorical_pause", "neutral_pause")
LEXICAL = ("filler_word", "repetition", "revision")
MIN_AUDIO_S = CONTEXT_S + WINDOW_S
# Reproducible ASR. Decoding in int8 turns ~1e-5 platform differences in the log-mel
# features (NumPy FFT) into different words on Windows and Linux; float32 on CPU gives
# identical transcriptions on both (measured). GPU or int8 stay available but change results.
ASR_DEFAULTS = {"asr_device": "cpu", "asr_compute_type": "float32"}


@dataclass
class PreparedRecording:
    track: AcousticTrack
    words: list[Word]
    duration_s: float
    timings: dict

    @property
    def frames(self) -> np.ndarray:
        return self.track.frame_matrix()


def cache_key(**parts) -> str:
    """Hash of media, region, code and parameters; never just a file name."""
    return hashlib.sha256(json.dumps({"engine": ENGINE_VERSION, **parts}, sort_keys=True).encode()).hexdigest()


def transcribe(asr, audio: np.ndarray, chunk_s: float = 0.0, overlap_s: float = 2.0) -> list[Word]:
    """chunk_s=0 transcribes in one pass (training condition); otherwise overlapped
    chunks are merged by time+text so seam duplicates collapse but real repeats stay."""
    if not chunk_s:
        return asr.transcribe(audio)
    words: list[Word] = []
    duration = len(audio) / SAMPLE_RATE
    for t0 in np.arange(0.0, max(duration - overlap_s, 1e-9), chunk_s - overlap_s):
        a = int(t0 * SAMPLE_RATE)
        local = asr.transcribe(audio[a : a + int(chunk_s * SAMPLE_RATE)])
        words = dedupe_words(
            words, [Word(w.text, w.norm, w.start + t0, w.end + t0, w.prob, w.punct_after) for w in local]
        )
    return words


def prepare(
    audio: np.ndarray, asr, *, asr_chunk_s: float = 0.0, use_silero: bool = True, cache: Path | None = None
) -> PreparedRecording:
    if cache and cache.exists():
        prep = pickle.loads(cache.read_bytes())
        prep.timings = {**prep.timings, "cache_hit": True}
        return prep
    t = time.perf_counter()
    words = transcribe(asr, audio, asr_chunk_s)
    t_asr = time.perf_counter() - t
    t = time.perf_counter()
    track = extract_acoustic(audio, use_silero=use_silero)
    prep = PreparedRecording(
        track,
        words,
        len(audio) / SAMPLE_RATE,
        {"asr_s": t_asr, "acoustic_s": time.perf_counter() - t, "cache_hit": False},
    )
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_bytes(pickle.dumps(prep))
    return prep


def window_tensors(
    frames: np.ndarray,
    words: list[Word],
    starts,
    *,
    context: bool = True,
    include_pros: bool = True,
    ling_dim: int | None = None,
) -> dict[str, np.ndarray]:
    """All windows at once, in the offline feature ordering: pros, ac, ling, lpos, lmask."""
    starts = np.asarray(starts, np.float64)
    center = speaker_f0_center(frames)
    ling, lpos, lmask = linguistic_windows([w.to_dict() for w in words], word_features(words), starts, context=context)
    if ling_dim is not None:
        if ling_dim > ling.shape[-1]:
            raise ValueError(f"El checkpoint espera ling={ling_dim}, pero el extractor produce {ling.shape[-1]}")
        ling = ling[:, :, :ling_dim]
    out = {"ac": acoustic_windows(center_f0(frames, center), starts), "ling": ling, "lpos": lpos, "lmask": lmask}
    if include_pros:
        out["pros"] = prosody_windows(frames, starts, center)
    return out


def decode(
    probs: np.ndarray,
    starts,
    labels: list[str],
    thresholds: dict | None = None,
    score_kind: str = "softmax_uncalibrated",
) -> list[dict]:
    """Merge contiguous windows of the same label into non-overlapping events.

    Each window owns the STRIDE_S slice it opens, so events tile without overlap;
    the interval is the detector's resolution, not the 3 s of evidence behind it.
    A window whose top score is below its class threshold becomes `uncertain`
    (abstention) instead of an event.
    """
    thresholds = thresholds or {}
    rows = []
    for t0, p in zip(starts, probs):
        i = int(p.argmax())
        if labels[i] == BACKGROUND:
            rows.append(None)
            continue
        decision = "event" if p[i] >= thresholds.get(labels[i], 0.0) else "uncertain"
        rows.append((labels[i], decision, float(p[i]), float(t0)))
    events: list[dict] = []
    for row in rows:
        last = events[-1] if events else None
        if (
            row
            and last
            and (last["label"], last["decision"]) == row[:2]
            and row[3] * 1000 - last["end_ms"] <= STRIDE_S * 500
        ):
            last["end_ms"] = round((row[3] + STRIDE_S) * 1000)
            last["score"], last["n_windows"] = (max(last["score"], row[2]), last["n_windows"] + 1)
        elif row:
            events.append(
                {
                    "label": row[0],
                    "decision": row[1],
                    "score": row[2],
                    "score_kind": score_kind,
                    "n_windows": 1,
                    "start_ms": round(row[3] * 1000),
                    "end_ms": round((row[3] + STRIDE_S) * 1000),
                }
            )
    return events


def refine(events: list[dict], prep: PreparedRecording) -> list[dict]:
    """Experimental boundary hypothesis (M3): snap lexical events to ASR words and
    pause events to the longest VAD silence inside them; other classes unchanged."""
    vad = prep.frames[:, 3] > 0.5
    for ev in events:
        a, b = ev["start_ms"] / 1000, ev["end_ms"] / 1000
        if ev["label"] in LEXICAL:
            inside = [w for w in prep.words if a <= (w.start + w.end) / 2 < b]
            if inside:
                ev["start_ms"], ev["end_ms"] = (round(inside[0].start * 1000), round(inside[-1].end * 1000))
        elif ev["label"] in PAUSES:
            lo, hi = int(a * 100), min(int(b * 100), len(vad))
            best, run_start = (0, 0), None
            for k in range(lo, hi + 1):
                silent = k < hi and not vad[k]
                if silent and run_start is None:
                    run_start = k
                if not silent and run_start is not None:
                    best, run_start = max(best, (k - run_start, run_start)), None
            if best[0]:
                ev["start_ms"], ev["end_ms"] = best[1] * 10, (best[1] + best[0]) * 10
        ev["boundary_source"] = "refined_asr_vad"
    return events


def event_text(words: list[Word], start_ms: int, end_ms: int) -> str:
    return " ".join(w.text for w in words if start_ms <= (w.start + w.end) * 500 < end_ms)


class MRCDEngine:
    """Load once per worker process; `analyze` is not thread-safe."""

    def __init__(
        self,
        inference,
        asr,
        *,
        asr_chunk_s: float = 0.0,
        thresholds: dict | None = None,
        refine_boundaries: bool = False,
        use_silero: bool = True,
        temperature: float = 1.0,
        init_s: float = 0.0,
    ):
        if inference.net.mode == "trimodal":
            raise ValueError("El producto analiza audio/texto; la rama de video aún no está integrada.")
        self.inference, self.asr, self.init_s, self.calls = inference, asr, init_s, 0
        self.config = {
            "asr_chunk_s": asr_chunk_s,
            "thresholds": thresholds or {},
            "refine_boundaries": refine_boundaries,
            "use_silero": use_silero,
            "temperature": temperature,
            "engine_version": ENGINE_VERSION,
            "mode": inference.net.mode,
        }
        net = inference.net
        self.ling_dim = net.ling.proj[0].in_features if net.ling is not None else None
        self.include_pros = net.pros is not None

    @classmethod
    def load(
        cls,
        checkpoint: str | Path,
        *,
        whisper_size: str = "small",
        device: str | None = None,
        asr_device: str = ASR_DEFAULTS["asr_device"],
        asr_compute_type: str = ASR_DEFAULTS["asr_compute_type"],
        **config,
    ):
        import torch

        from core.extractors.linguistic import VerbatimASR
        from core.inference import InferenceEngine

        t = time.perf_counter()
        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        engine = cls(
            InferenceEngine.from_checkpoint(Path(checkpoint), dev),
            VerbatimASR(whisper_size, device=asr_device, compute_type=asr_compute_type),
            **config,
        )
        engine.config.update(whisper_size=whisper_size, asr_device=asr_device, asr_compute_type=asr_compute_type)
        engine.init_s = time.perf_counter() - t
        return engine

    def prepare(self, audio: np.ndarray, cache: Path | None = None) -> PreparedRecording:
        return prepare(
            audio, self.asr, asr_chunk_s=self.config["asr_chunk_s"], use_silero=self.config["use_silero"], cache=cache
        )

    def analyze(
        self, audio: np.ndarray, *, prepared: PreparedRecording | None = None, cache: Path | None = None
    ) -> dict:
        audio = np.asarray(audio, np.float32).reshape(-1)
        duration = len(audio) / SAMPLE_RATE
        if duration < MIN_AUDIO_S:
            raise ValueError(
                f"Audio de {duration:.1f} s; MRCD necesita al menos {MIN_AUDIO_S:.0f} s "
                "porque el ramal lingüístico mira 10 s de contexto."
            )
        t_all = time.perf_counter()
        prep = prepared or self.prepare(audio, cache)
        timings = dict(prep.timings)
        t = time.perf_counter()
        starts = window_starts(prep.duration_s)
        tensors = window_tensors(
            prep.frames,
            prep.words,
            starts,
            context=bool(self.ling_dim and self.ling_dim > 10),
            include_pros=self.include_pros,
            ling_dim=self.ling_dim,
        )
        timings["windows_s"] = time.perf_counter() - t
        t = time.perf_counter()
        probs = self.inference.predict(tensors)
        # temperature is fitted on dev with evaluation.fit_temperature, never on test
        if (temperature := self.config["temperature"]) != 1.0:
            z = np.log(np.clip(probs, 1e-12, 1)) / temperature
            probs = np.exp(z - z.max(1, keepdims=True))
            probs /= probs.sum(1, keepdims=True)
        timings["inference_s"] = time.perf_counter() - t
        t = time.perf_counter()
        events = decode(
            probs,
            starts,
            list(self.inference.labels),
            self.config["thresholds"],
            "softmax_uncalibrated" if temperature == 1.0 else "softmax_temperature_dev",
        )
        if self.config["refine_boundaries"]:
            events = refine(events, prep)
        for ev in events:
            ev["text"] = event_text(prep.words, ev["start_ms"], ev["end_ms"])
        timings["decode_s"] = time.perf_counter() - t
        # a preparation computed by the caller for this input still counts; a cache hit does not
        reused = 0.0 if prep.timings.get("cache_hit") else prep.timings["asr_s"] + prep.timings["acoustic_s"]
        timings["total_s"] = time.perf_counter() - t_all + (reused if prepared is not None else 0.0)
        timings["cold"], self.calls = self.calls == 0, self.calls + 1
        provenance = self.inference.provenance
        return {
            "events": events,
            "words": [w.to_dict() for w in prep.words],
            "duration_ms": round(duration * 1000),
            "timings": timings,
            "init_s": self.init_s,
            "rtf": timings["total_s"] / duration,
            "hardware": hardware(self.inference.device),
            "config": self.config,
            "provenance": provenance,
            "warning": provenance.get("warning")
            or {"human": None, "auto": SMOKE_HEADER}.get(
                provenance.get("label_source"), "Checkpoint sin evidencia humana verificada"
            ),
        }


def hardware(device: str) -> dict:
    info = {"platform": platform.platform(), "processor": platform.processor(), "device": device}
    try:
        import psutil

        info["rss_mb"] = round(psutil.Process().memory_info().rss / 2**20)
    except ImportError:
        info["rss_mb"] = None
    return info
