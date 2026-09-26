#!/usr/bin/env python3
"""Ingesta de UN vídeo concreto para estudio de caso, sin tocar el corpus.

Diferencias respecto a `data_mining.py`, y por qué:

- **No filtra por multi-rostro.** El corpus descarta segmentos con dos caras
  porque allí buscamos hablantes aislados. En una entrevista eso tiraría el
  vídeo entero. Aquí se usa seguimiento de identidad (`process_video_tracked`)
  para quedarse con el rostro dominante y descartar al interlocutor.
- **No segmenta ni recorta por continuidad facial.** El estudio de caso quiere
  el flujo completo, no trozos seleccionados.
- **Escribe en `data/case/`**, fuera de `data/segments/`, para que el corpus de
  entrenamiento no quede contaminado con material de evaluación.

Salidas:
    data/case/<vid>.wav              audio WAV mono 16 kHz PCM s16le
    data/case/<vid>_10fps.mp4        vídeo a 10 fps
    data/case/<vid>.npz              señales acústicas + cinésicas + rasgos léxicos
    data/case/<vid>.words.json       transcripción literal con tiempos por palabra
    data/case/<vid>.face_report.json a quién se siguió y con qué margen
    data/case/<vid>.meta.json        metadatos e índice del caso

Uso:
    python scripts/ingest_single_video.py --url https://www.youtube.com/watch?v=XXXX
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.constants import SAMPLE_RATE, VIDEO_FPS  # noqa: E402
from core.contracts import (RAW_FEATURE_SCHEMA_VERSION, raw_feature_contract,
                            validate_raw_features)  # noqa: E402
from core.extractors.acoustic import extract_acoustic, load_wav  # noqa: E402
from core.extractors.linguistic import VerbatimASR, word_features  # noqa: E402

CASE = ROOT / "data/case"


def _ffmpeg() -> str:
    for c in ("ffmpeg", str(ROOT / "bin/ffmpeg.exe")):
        try:
            subprocess.run([c, "-version"], capture_output=True, check=True)
            return c
        except Exception:
            continue
    raise RuntimeError("No se encontró ffmpeg en el PATH.")


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True, capture_output=True)


def download(url: str) -> tuple[Path, dict]:
    import yt_dlp

    CASE.mkdir(parents=True, exist_ok=True)
    opts = dict(
        quiet=True,
        no_warnings=True,
        noplaylist=True,
        restrictfilenames=True,
        format="bestvideo[height<=1080]+bestaudio/best[height<=1080]",
        merge_output_format="mp4",
        outtmpl=str(CASE / "%(id)s_src.%(ext)s"),
    )
    with yt_dlp.YoutubeDL(opts) as y:
        info = y.extract_info(url, download=True)
    src = next(CASE.glob(f"{info['id']}_src.*"))
    meta = {
        "video_id": info["id"],
        "url": url,
        "title": info.get("title"),
        "channel": info.get("uploader"),
        "duration_s": info.get("duration"),
        "license": info.get("license"),
        "upload_date": info.get("upload_date"),
    }
    return src, meta


def to_tracks(src: Path, vid: str) -> tuple[Path, Path]:
    ff = _ffmpeg()
    wav, mp4 = CASE / f"{vid}.wav", CASE / f"{vid}_10fps.mp4"
    if not wav.exists():
        run(
            [
                ff,
                "-y",
                "-i",
                str(src),
                "-vn",
                "-ac",
                "1",
                "-ar",
                str(SAMPLE_RATE),
                "-c:a",
                "pcm_s16le",
                str(wav),
            ]
        )
    if not mp4.exists():
        run(
            [
                ff,
                "-y",
                "-i",
                str(src),
                "-an",
                "-r",
                str(VIDEO_FPS),
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "23",
                str(mp4),
            ]
        )
    return wav, mp4


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--whisper", default="small")
    ap.add_argument("--variety", default="es-PE")
    ap.add_argument(
        "--max-seconds",
        type=float,
        default=None,
        help="recorta el análisis (útil para una prueba rápida)",
    )
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    CASE.mkdir(parents=True, exist_ok=True)
    print(f"[ingesta] descargando {a.url}")
    src, meta = download(a.url)
    vid = meta["video_id"]
    print(f"[ingesta] {meta['title']} · {meta['duration_s']} s · {meta['channel']}")

    npz_path = CASE / f"{vid}.npz"
    if npz_path.exists() and not a.force:
        with np.load(npz_path, allow_pickle=False) as existing:
            validate_raw_features(existing, path=str(npz_path))
        print(f"[ingesta] {npz_path} ya existe (usa --force para rehacer)")
        return

    wav, mp4 = to_tracks(src, vid)
    print(f"[ingesta] pistas listas: {wav.name}, {mp4.name}")

    # ---------------------------------------------------------------- acústica
    audio, _sr = load_wav(str(wav))
    ac = extract_acoustic(audio)
    dur = len(audio) / SAMPLE_RATE
    speech = float(ac.vad.mean())
    print(f"[ingesta] acústica: {dur:.0f} s · VAD={ac.vad_backend} · voz={speech:.0%}")

    # ------------------------------------------------------------- ASR literal
    print("[ingesta] ASR verbatim (puede tardar)")
    asr = VerbatimASR(a.whisper)
    words = asr.transcribe(audio)
    wf = word_features(words, a.variety)
    print(f"[ingesta] {len(words)} palabras transcritas")

    # -------------------------------------------- cinésica con seguimiento
    print("[ingesta] cinésica con seguimiento de identidad")
    from core.extractors.kinesic import KinesicExtractor

    kx = KinesicExtractor(num_faces=3)  # detecta varias, sigue una
    try:
        ktrack, freport = kx.process_video_tracked(str(mp4), max_seconds=a.max_seconds)
    finally:
        kx.close()

    margin = freport["dominance_margin"]
    print(
        f"[ingesta] pistas faciales: {len(freport['tracks'])} · elegida: "
        f"{freport['selected']} · margen: {margin:.2f}"
    )
    for r in freport["tracks"][:4]:
        print(
            f"    pista {r['track_id']}: {r['frames']} frames · área mediana "
            f"{r['median_area']:.3f} · cuota {r['score_share']:.0%}"
        )
    if margin < 0.25 and len(freport["tracks"]) > 1:
        print(
            "[ingesta] AVISO: margen de dominancia bajo. El montaje reparte el tiempo "
            "entre dos personas y la elección automática puede no ser la correcta. "
            "Revisa .face_report.json antes de usar los rasgos cinésicos."
        )
    cover = freport["frames_with_dominant"] / max(freport["frames"], 1)
    print(f"[ingesta] cobertura del rostro seguido: {cover:.0%} de los frames")

    np.savez_compressed(
        npz_path,
        raw_feature_schema=np.array(RAW_FEATURE_SCHEMA_VERSION),
        raw_contract=np.array(json.dumps(raw_feature_contract(), sort_keys=True)),
        raw_extractor=np.array("ingest_single_video"),
        raw_extractor_config=np.array(json.dumps({"asr_model": a.whisper, "variety": a.variety,
                                                  "vad_backend": ac.vad_backend,
                                                  "face_tracking_method": "dominant_iou"}, sort_keys=True)),
        ac_frames=ac.frame_matrix(),
        ac_times=ac.times,
        f0=ac.f0,
        rms=ac.rms,
        vad=ac.vad,
        hf_ratio=ac.hf_ratio,
        word_feats=wf,
        k_times=ktrack.times,
        k_vectors=ktrack.vectors,
        k_detected=ktrack.detected,
        k_frontal=ktrack.frontal,
        k_nfaces=ktrack.n_faces,
    )
    (CASE / f"{vid}.words.json").write_text(
        json.dumps(
            [
                {
                    "start": w.start,
                    "end": w.end,
                    "text": w.text,
                    "norm": w.norm,
                    "prob": w.prob,
                    "punct_after": w.punct_after,
                }
                for w in words
            ],
            ensure_ascii=False,
            indent=1,
        ),
        encoding="utf-8",
    )
    (CASE / f"{vid}.face_report.json").write_text(
        json.dumps(freport, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    meta.update(
        {
            "duration_s": dur,
            "speech_ratio": speech,
            "n_words": len(words),
            "vad_backend": ac.vad_backend,
            "face_coverage": cover,
            "dominance_margin": margin,
            "recording_id": vid,
            "features_npz": str(npz_path.relative_to(ROOT)).replace("\\", "/"),
            "words_json": f"data/case/{vid}.words.json",
            "wav": f"data/case/{vid}.wav",
            "mp4": f"data/case/{vid}_10fps.mp4",
            "speaker_id": f"case_{vid}",
            "raw_feature_schema": RAW_FEATURE_SCHEMA_VERSION,
        }
    )
    (CASE / f"{vid}.meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False, default=float), encoding="utf-8"
    )
    print(f"[ingesta] listo -> {npz_path}")


if __name__ == "__main__":
    main()
