"""Comparator adapters run by the worker: MRCD (engine), rules (B0) and GPT audio (G).

Each adapter returns {"events": [...], "segments": {segment_id: status}, ...provenance}.
Events use integer ms on the analysis-copy clock and are owned by the segment that
contains their midpoint. Heavy models are loaded once per worker process.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import subprocess
import urllib.error
import urllib.request
from pathlib import Path

from app import media, media_dir
from core.contracts import FEATURE_SCHEMA_VERSION, RAW_FEATURE_SCHEMA_VERSION
from core.engine import MRCDEngine, cache_key, event_text, prepare

ROOT = Path(__file__).resolve().parents[1]
GPT_PROMPT = ROOT / "config" / "gpt_prompt_v1.txt"
_loaded: dict = {}


def _asr(size: str):
    if ("asr", size) not in _loaded:
        from core.extractors.linguistic import VerbatimASR

        _loaded[("asr", size)] = VerbatimASR(size)
    return _loaded[("asr", size)]


def prepared(recording: dict, audio, config: dict, stage=lambda name: None):
    """ASR + acoustic track once per media/configuration, shared by MRCD, rules and window export."""
    cfg = {"asr_chunk_s": float(config.get("asr_chunk_s", 0.0)), "use_silero": config.get("use_silero", True)}
    size = config.get("whisper_size", "small")
    key = cache_key(media=recording["data"]["media"]["analysis"]["sha256"], whisper=size,
                    raw_schema=RAW_FEATURE_SCHEMA_VERSION, **cfg)
    stage("prepare_features")
    return prepare(audio, _asr(size), cache=media_dir(recording) / "prepared" / f"{key}.pkl", **cfg)


def _own(job, events: list[dict], prefix: str) -> list[dict]:
    segs = job.recording["data"]["segments"]
    for i, ev in enumerate(events):
        ev.update(event_id=f"{prefix}-e{i:04d}", segment_id=media.owner(segs, ev["start_ms"], ev["end_ms"]))
    return events


def _commit() -> str | None:
    """Code version for provenance: git when available, MRCD_COMMIT in images, else unknown."""
    if os.environ.get("MRCD_COMMIT"):
        return os.environ["MRCD_COMMIT"]
    try:
        run = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    except OSError:
        return None
    return run.stdout.strip() or None


def mrcd(job) -> dict:
    ckpt = Path(
        job.config.get("checkpoint") or os.environ.get("MRCD_CHECKPOINT", ROOT / "models" / "ckpt_audio_text.pt")
    )
    cfg = {
        k: job.config[k]
        for k in ("asr_chunk_s", "thresholds", "refine_boundaries", "use_silero", "temperature")
        if k in job.config
    }
    size = job.config.get("whisper_size", "small")
    key = ("mrcd", str(ckpt), size, json.dumps(cfg, sort_keys=True))
    if key not in _loaded:
        import torch

        from core.inference import InferenceEngine

        device = "cuda" if torch.cuda.is_available() else "cpu"
        engine = MRCDEngine(InferenceEngine.from_checkpoint(ckpt, device), _asr(size), **cfg)
        engine.config["whisper_size"] = size
        _loaded[key] = engine
    engine = _loaded[key]
    prep = prepared(job.recording, job.audio, job.config, job.stage)
    job.stage("detect")
    result = engine.analyze(job.audio, prepared=prep)
    result["events"] = _own(job, result["events"], job.run_id)
    result["segments"] = {s["id"]: "succeeded" for s in job.recording["data"]["segments"]}
    result["model_version"] = {
        "checkpoint": ckpt.name,
        "checkpoint_sha256": media.sha256(ckpt),
        "commit": _commit(),
        "config": engine.config,
        "feature_schema": FEATURE_SCHEMA_VERSION,
        "training": engine.inference.provenance,
    }
    return result


def rules(job) -> dict:
    from core.annotation.auto_labeler import label_recording

    prep = prepared(job.recording, job.audio, job.config, job.stage)
    job.stage("detect")
    cands = [
        c
        for c in label_recording(prep.track, prep.words, variety=job.config.get("variety", "es-PE"))
        if c.source == "auto"
    ]
    events = [
        {
            "label": c.category,
            "start_ms": round(c.start * 1000),
            "end_ms": round(c.end * 1000),
            "decision": "event",
            "score": c.confidence,
            "score_kind": "heuristic_confidence",
            "evidence": c.evidence,
            "text": event_text(prep.words, round(c.start * 1000), round(c.end * 1000)),
        }
        for c in sorted(cands, key=lambda c: c.start)
    ]
    return {
        "events": _own(job, events, job.run_id),
        "words": [w.to_dict() for w in prep.words],
        "timings": prep.timings,
        "segments": {s["id"]: "succeeded" for s in job.recording["data"]["segments"]},
        "warning": "Heurísticas con audio y ASR, sin video; referencia automática.",
        "model_version": {"commit": _commit(), "config": job.config},
    }


def gpt(job) -> dict:
    """GPT audio as an operational end-to-end comparator: one call per segment,
    responses cached by audio+model+prompt hash, no automatic paid retries."""
    if not job.project.get("external_processing"):
        raise PermissionError("El proyecto no autoriza enviar audio a servicios externos")
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise PermissionError("Falta OPENAI_API_KEY en el entorno del worker")
    model, prompt = (job.config.get("model", "gpt-audio-1.5"), GPT_PROMPT.read_text(encoding="utf-8"))
    analysis = job.media_dir / "analysis.wav"
    events, legitimate, records, status = [], [], [], {}
    for seg in job.recording["data"]["segments"]:
        job.stage(f"gpt:{seg['id']}")
        try:
            blob = media.clip_wav(analysis, seg["audio_start_ms"], seg["audio_end_ms"])
            fp = hashlib.sha256(blob + model.encode() + prompt.encode()).hexdigest()
            cache = job.media_dir / "gpt" / f"{seg['id']}-{fp[:16]}.json"
            if not cache.exists():
                body = {
                    "model": model,
                    "modalities": ["text"],
                    "store": False,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "input_audio",
                                    "input_audio": {"data": base64.b64encode(blob).decode(), "format": "wav"},
                                },
                            ],
                        }
                    ],
                }
                req = urllib.request.Request(
                    "https://api.openai.com/v1/chat/completions",
                    data=json.dumps(body).encode(),
                    headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                )
                with urllib.request.urlopen(req, timeout=180) as resp:
                    raw = json.load(resp)
                cache.parent.mkdir(parents=True, exist_ok=True)
                cache.write_text(json.dumps({"request_hash": fp, "response": raw}), encoding="utf-8")
            raw = json.loads(cache.read_text(encoding="utf-8"))["response"]
            choice = raw["choices"][0]
            if choice.get("finish_reason") != "stop":
                raise ValueError("respuesta incompleta")
            obj = json.loads(choice["message"]["content"])
            span = seg["audio_end_ms"] - seg["audio_start_ms"]
            for group, out in (("events", events), ("legitimate_uses", legitimate)):
                for ev in obj[group]:
                    a, b = (round(float(ev["start"]) * 1000), round(float(ev["end"]) * 1000))
                    if not 0 <= a < b <= span or (
                        group == "events" and ev.get("decision") not in ("event", "uncertain")
                    ):
                        raise ValueError("intervalo o decisión fuera del esquema; no se corrige en silencio")
                    a, b = a + seg["audio_start_ms"], b + seg["audio_start_ms"]
                    if seg["core_start_ms"] <= (a + b) / 2 < seg["core_end_ms"]:
                        out.append(
                            {
                                "label": ev.get("label"),
                                "start_ms": a,
                                "end_ms": b,
                                "text": ev.get("text"),
                                "decision": ev.get("decision", "event"),
                                "evidence": ev.get("evidence"),
                                "score": None,
                                "score_kind": None,
                            }
                        )
            records.append(
                {
                    "segment_id": seg["id"],
                    "usage": raw.get("usage"),
                    "model_returned": raw.get("model"),
                    "limitations": obj.get("limitations", []),
                }
            )
            status[seg["id"]] = "succeeded"
        except (urllib.error.URLError, ValueError, KeyError, TypeError) as exc:
            status[seg["id"]] = f"failed: {exc}"[:300]
    return {
        "events": _own(job, events, job.run_id),
        "legitimate": legitimate,
        "segments": status,
        "records": records,
        "model_version": {"model": model, "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest()},
        "warning": "Tiempos propuestos por el modelo, no verificados.",
    }


SYSTEMS = {"mrcd": mrcd, "rules": rules, "gpt": gpt}
