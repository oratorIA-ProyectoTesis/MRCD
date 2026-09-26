#!/usr/bin/env python3
"""Extrae las 3 señales por segmento y genera el pre-etiquetado + manifest + EAF.

Entrada : data/segments/index.json (de data_mining.py) o --wav/--mp4 sueltos.
Salida  : data/features/<id>.npz, <id>.words.json, <id>.events.json, <id>.eaf
          data/dataset_manifest.json (manifest unificado)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.annotation.auto_labeler import label_recording, window_labels, write_eaf  # noqa: E402
from core.constants import (H, KINESIC_FIELDS, LABELS, STRIDE_S, TAXONOMY, WINDOW_S)  # noqa: E402
from core.contracts import (RAW_FEATURE_SCHEMA_VERSION, feature_contract,
                            raw_feature_contract, validate_raw_features)  # noqa: E402
from core.extractors.acoustic import extract_acoustic, load_wav  # noqa: E402
from core.extractors.linguistic import VerbatimASR, WORD_FEAT_NAMES, word_features  # noqa: E402

FEAT = ROOT / "data/features"
MANIFEST = ROOT / "data/dataset_manifest.json"


def signal_worker(rec: dict, with_video: bool) -> tuple:
    """Proceso hijo: acústica + cinésica (CPU). Devuelve (rid, ac, kin, timings)."""
    t = {}
    audio, sr = load_wav(str(ROOT / rec["wav"]))
    s = time.time(); ac = extract_acoustic(audio, sr); t["acoustic_s"] = round(time.time() - s, 2)
    kin = None
    if with_video and rec.get("mp4") and (ROOT / rec["mp4"]).exists():
        from core.extractors.kinesic import KinesicExtractor
        kx = KinesicExtractor(num_faces=3)
        try:
            s = time.time()
            kin, t["face_tracking"] = kx.process_video_tracked(str(ROOT / rec["mp4"]))
            t["kinesic_s"] = round(time.time() - s, 2)
        finally:
            kx.close()
    return rec["recording_id"], ac, kin, t


def insertion_rate(words, ac, variety: str) -> dict:
    """Muletillas transcritas dentro de tramos que el VAD marca como silencio (posibles alucinaciones
    inducidas por el initial_prompt)."""
    from core.extractors.linguistic import lexicon_for
    lex = lexicon_for(variety)
    fill = [w for w in words if w.norm in lex]
    sus = 0
    for w in fill:
        m = (ac.times >= w.start) & (ac.times < w.end)
        if m.any() and (ac.vad[m] == 0).mean() > 0.5:
            sus += 1
    return dict(filler_tokens=len(fill), in_silence=sus, rate=round(sus / len(fill), 3) if fill else 0.0)


def process(rec: dict, asr, kx, variety: str, roberta=None, pre: tuple | None = None, recall_mode: bool = False) -> dict:
    t0 = time.time()
    rid = rec["recording_id"]
    audio, sr = load_wav(str(ROOT / rec["wav"]))
    dur = len(audio) / sr
    timings = {}

    if pre is not None:
        ac, kin, pt = pre; timings.update({k: v for k, v in pt.items() if k != "face_tracking"})
        face_tracking = pt.get("face_tracking")
    else:
        s = time.time(); ac = extract_acoustic(audio, sr); timings["acoustic_s"] = round(time.time() - s, 2)
        kin = None
        face_tracking = None
    s = time.time(); words = asr.transcribe(audio); timings["asr_s"] = round(time.time() - s, 2)
    # Control de coherencia VAD vs ASR: si el VAD marca como silencio > 50 % del tiempo
    # cubierto por palabras, se descarta y se recalcula con VAD por energía.
    if words:
        cov = np.zeros(len(ac.times), bool)
        for w in words:
            cov[(ac.times >= w.start) & (ac.times < w.end)] = True
        miss = float((ac.vad[cov] == 0).mean()) if cov.any() else 0.0
        if miss > 0.5:
            print(f"[{rid}] VAD incoherente con ASR ({miss:.0%} de palabras en 'silencio'); usando VAD por energía")
            ac = extract_acoustic(audio, sr, use_silero=False)
    wf = word_features(words, variety)
    if roberta is not None and words:
        s = time.time(); wf = np.concatenate([wf, roberta.encode(words)], 1); timings["roberta_s"] = round(time.time() - s, 2)
    if pre is None and kx is not None and rec.get("mp4"):
        s = time.time()
        if hasattr(kx, "process_video_tracked"):
            kin, face_tracking = kx.process_video_tracked(str(ROOT / rec["mp4"]))
        else:  # injectable legacy test adapter
            kin = kx.process_video(str(ROOT / rec["mp4"]))
        timings["kinesic_s"] = round(time.time() - s, 2)

    cands = label_recording(ac, words, kin, variety, recall_mode=recall_mode)
    rows = window_labels(cands, dur)

    FEAT.mkdir(parents=True, exist_ok=True)
    npz = FEAT / f"{rid}.npz"
    raw_config = {"asr_model": getattr(asr, "model_size", "n/a"), "variety": variety,
                  "roberta": roberta is not None, "vad_backend": ac.vad_backend,
                  "face_tracking_method": "dominant_iou" if face_tracking is not None else "none_or_external"}
    np.savez_compressed(
        npz, raw_feature_schema=np.array(RAW_FEATURE_SCHEMA_VERSION),
        raw_contract=np.array(json.dumps(raw_feature_contract(), sort_keys=True)),
        raw_extractor=np.array("extract_and_label"),
        raw_extractor_config=np.array(json.dumps(raw_config, sort_keys=True)),
        ac_frames=ac.frame_matrix(), ac_times=ac.times, f0=ac.f0, rms=ac.rms, vad=ac.vad,
        hf_ratio=ac.hf_ratio if ac.hf_ratio is not None else np.zeros(0, np.float32),
        word_feats=wf.astype(np.float32) if len(wf) else np.zeros((0, len(WORD_FEAT_NAMES)), np.float32),
        k_times=kin.times if kin is not None else np.zeros(0), 
        k_vectors=kin.vectors if kin is not None else np.zeros((0, 12), np.float32),
        k_detected=kin.detected if kin is not None else np.zeros(0, bool),
    )
    (FEAT / f"{rid}.words.json").write_text(json.dumps([w.to_dict() for w in words], ensure_ascii=False), encoding="utf-8")
    events = [c.to_dict() for c in cands]
    (FEAT / f"{rid}.events.json").write_text(json.dumps(events, ensure_ascii=False, default=float), encoding="utf-8")
    media = {str(ROOT / rec["wav"]): "audio/x-wav"}
    if rec.get("mp4"):
        media[str(ROOT / rec["mp4"])] = "video/mp4"
    eaf = FEAT / f"{rid}.eaf"
    write_eaf(eaf, cands, words, ac.silence_segments, media)

    counts = {c: sum(e["category"] == c and e.get("source") == "auto" for e in events) for c in TAXONOMY}
    low_conf = {c: sum(e["category"] == c and e.get("source") == "auto_low_conf" for e in events) for c in TAXONOMY}
    wcounts = {l: sum(r["primary"] == l for r in rows) for l in LABELS}
    out = dict(rec, duration_s=round(dur, 3), n_words=len(words), vad_backend=ac.vad_backend,
               kinesic_detection_rate=round(kin.detection_rate, 3) if kin is not None else None,
               n_windows=len(rows), event_counts=counts, low_conf_counts=low_conf, window_primary_counts=wcounts,
               asr_insertions=insertion_rate(words, ac, variety),
               features_npz=str(npz.relative_to(ROOT)), words_json=str((FEAT / f"{rid}.words.json").relative_to(ROOT)),
               events_json=str((FEAT / f"{rid}.events.json").relative_to(ROOT)), eaf=str(eaf.relative_to(ROOT)),
               timings=timings, realtime_factor=round((time.time() - t0) / max(dur, 1e-6), 3),
               label_source="auto", asr_model=getattr(asr, "model_size", "n/a"),
               asr_device=getattr(asr, "device", "n/a"),
               face_tracking_method="dominant_iou" if face_tracking is not None else "none_or_external",
               face_tracking=face_tracking, raw_feature_schema=RAW_FEATURE_SCHEMA_VERSION,
               raw_extractor_config=raw_config)
    print(f"[{rid}] {dur:.0f}s | palabras={len(words)} | eventos={counts} | ventanas={len(rows)} | RTF={out['realtime_factor']}")
    return out


def main():
    from concurrent.futures import ProcessPoolExecutor, as_completed
    import os

    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default=str(ROOT / "data/segments/index.json"))
    ap.add_argument("--only", nargs="*", help="recording_ids a procesar")
    ap.add_argument("--max", type=int)
    ap.add_argument("--whisper", default="small", help="tiny|base|small|medium|large-v3")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--variety", default="es-PE")
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--roberta", action="store_true", help="añadir embeddings RoBERTa-BNE a los rasgos por palabra")
    ap.add_argument("--recall-mode", action="store_true", help="candidatos de baja confianza SOLO para revisión humana")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--force", action="store_true", help="reprocesar aunque exista el .npz")
    args = ap.parse_args()

    recs = json.loads(Path(args.index).read_text(encoding="utf-8"))
    if args.only:
        recs = [r for r in recs if r["recording_id"] in set(args.only)]
    recs = recs[: args.max] if args.max else recs
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8")) if MANIFEST.exists() else {"recordings": []}
    done = {r["recording_id"]: i for i, r in enumerate(manifest["recordings"])}
    replacing = {r["recording_id"] for r in recs} if args.force else set()
    for entry in manifest["recordings"]:
        if entry["recording_id"] in replacing:
            continue
        path = ROOT / entry.get("features_npz", f"data/features/{entry['recording_id']}.npz")
        if not path.exists():
            raise ValueError(f"{path}: raw features referenced by manifest are missing; re-extract with --force")
        with np.load(path, allow_pickle=False) as existing:
            validate_raw_features(existing, path=str(path))
    if not args.force:   # reanudable
        recs = [r for r in recs if not ((FEAT / f"{r['recording_id']}.npz").exists() and r["recording_id"] in done)]
    print(f"[extract_and_label] {len(recs)} segmentos pendientes | workers={args.workers}")
    if not recs:
        return

    asr = VerbatimASR(args.whisper, device=args.device)
    print(f"[asr] Faster-Whisper {args.whisper} en {asr.device} ({asr.compute_type})")
    rob = None
    if args.roberta:
        from core.extractors.linguistic import SyntacticEncoder
        rob = SyntacticEncoder()

    def save_manifest():
        manifest["schema"] = dict(
            version="0.2.0", window_s=WINDOW_S, stride_s=STRIDE_S, taxonomy=list(TAXONOMY), labels=list(LABELS),
            kinesic_fields=list(KINESIC_FIELDS), word_feature_names=list(WORD_FEAT_NAMES),
            heuristic_thresholds=H, whisper_model=args.whisper, variety=args.variety, recall_mode=args.recall_mode,
            feature_contract=feature_contract(),
            note="Etiquetas 'auto' = candidatas heurísticas para revisión humana en ELAN; no son ground truth.")
        MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=float), encoding="utf-8")

    by_id = {r["recording_id"]: r for r in recs}
    # Acústica + cinésica en paralelo (CPU); ASR secuencial en el proceso principal (GPU/CPU)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(signal_worker, r, not args.no_video) for r in recs]
        for fut in as_completed(futs):
            try:
                rid, ac, kin, t = fut.result()
                entry = process(by_id[rid], asr, None, args.variety, rob, pre=(ac, kin, t), recall_mode=args.recall_mode)
            except Exception as exc:
                print(f"[error] {type(exc).__name__}: {str(exc)[:200]}")
                continue
            if rid in done:
                manifest["recordings"][done[rid]] = entry
            else:
                done[rid] = len(manifest["recordings"]); manifest["recordings"].append(entry)
            save_manifest()          # checkpoint tras cada segmento
    print(f"Manifest: {MANIFEST} ({len(manifest['recordings'])} grabaciones)")


if __name__ == "__main__":
    main()
