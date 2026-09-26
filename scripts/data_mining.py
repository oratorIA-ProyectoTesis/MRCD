#!/usr/bin/env python3
"""Minería audiovisual AUTÓNOMA en español (yt-dlp + ffmpeg + MediaPipe + Whisper).

Subcomandos:
  seeds          descarga config/seeds.txt (en orden)
  discover-auto  busca con yt-dlp hasta sumar 15–20 videos NUEVOS que pasen los filtros
  extract        WAV 16 kHz/16-bit/mono + MP4 10 fps + filtros de calidad + segmentación + recorte dinámico
  all            seeds -> discover-auto -> extract

Reanudable e idempotente: estado en data/raw/state.json (dedupe por video_id).
Trazabilidad: data/raw/sources_log.csv (licencia, canal, URL, fecha) y data/raw/rejected.csv (motivo).
No redistribuir audio ni video: solo rasgos derivados.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.constants import SAMPLE_RATE, VIDEO_FPS  # noqa: E402

DATA = ROOT / "data"
RAW, CLIPS, SEGS = DATA / "raw", DATA / "clips", DATA / "segments"
STATE = RAW / "state.json"
SOURCES_LOG, REJECTED = RAW / "sources_log.csv", RAW / "rejected.csv"
LOG_FIELDS = ["video_id", "url", "title", "channel", "license", "upload_date", "duration_s", "origin", "logged_at"]
REJ_FIELDS = ["video_id", "url", "stage", "reason", "logged_at"]


# ------------------------------------------------------------------ utilidades
def _ffmpeg() -> str:
    exe = shutil.which("ffmpeg")
    if not exe:
        sys.exit("ffmpeg no está en el PATH (Windows: winget install Gyan.FFmpeg).")
    return exe


def run(cmd: list[str]) -> None:
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode != 0:
        raise RuntimeError(f"Falló: {' '.join(cmd)}\n{r.stderr[-1500:]}")


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def load_state() -> dict:
    return json.loads(STATE.read_text(encoding="utf-8")) if STATE.exists() else {}


def save_state(st: dict) -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    tmp = STATE.with_suffix(".tmp"); tmp.write_text(json.dumps(st, indent=1, ensure_ascii=False), encoding="utf-8")
    tmp.replace(STATE)


def append_csv(path: Path, fields: list[str], row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in fields})


def reject(st: dict, vid: str, url: str, stage: str, reason: str) -> None:
    print(f"   ✗ {vid}: {stage} -> {reason}")
    st[vid] = dict(st.get(vid, {}), status="rejected", stage=stage, reason=reason, url=url)
    append_csv(REJECTED, REJ_FIELDS, dict(video_id=vid, url=url, stage=stage, reason=reason, logged_at=now()))
    save_state(st)


def cfg_load(path: str) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


# ------------------------------------------------------------------- descarga
YDL_BASE = dict(quiet=True, no_warnings=True, noplaylist=True, ignoreerrors=False, restrictfilenames=True)


def fetch_info(url: str) -> dict | None:
    import yt_dlp
    try:
        with yt_dlp.YoutubeDL(dict(YDL_BASE, skip_download=True)) as y:
            return y.extract_info(url, download=False)
    except Exception as exc:
        print(f"   ! metadatos no disponibles para {url}: {str(exc)[:160]}")
        return None


def download_one(info: dict, st: dict, origin: str, cfg: dict) -> bool:
    """Descarga un video (<=1080p, preferencia 720p). Devuelve True si quedó aceptado para extracción."""
    import yt_dlp

    vid, url = info["id"], info.get("webpage_url") or info.get("original_url")
    if st.get(vid, {}).get("status") in {"downloaded", "extracted", "rejected"}:
        return st[vid]["status"] != "rejected"
    dur = info.get("duration") or 0
    if not (cfg["min_duration_s"] <= dur <= cfg["max_duration_s"]):
        reject(st, vid, url, "metadata", f"duración {dur}s fuera de [{cfg['min_duration_s']}, {cfg['max_duration_s']}]")
        return False
    append_csv(SOURCES_LOG, LOG_FIELDS, dict(video_id=vid, url=url, title=info.get("title", ""), channel=info.get("channel", ""),
                                             license=info.get("license") or "no declarada (licencia estándar de YouTube)",
                                             upload_date=info.get("upload_date", ""), duration_s=dur, origin=origin, logged_at=now()))
    opts = dict(YDL_BASE, format="bv*[height<=1080]+ba/b[height<=1080]/b", format_sort=["res:720", "fps", "vcodec:h264"],
                merge_output_format="mp4", outtmpl=str(RAW / "%(id)s.%(ext)s"), writeinfojson=True)
    try:
        with yt_dlp.YoutubeDL(opts) as y:
            y.download([url])
    except Exception as exc:
        reject(st, vid, url, "download", str(exc)[:200]); return False
    if not (RAW / f"{vid}.mp4").exists():
        reject(st, vid, url, "download", "archivo mp4 no encontrado tras la descarga"); return False
    st[vid] = dict(status="downloaded", url=url, origin=origin, title=info.get("title", ""),
                   license=info.get("license") or "", channel=info.get("channel", ""))
    save_state(st)
    print(f"   ✓ {vid} descargado ({dur}s) — {info.get('title', '')[:60]}")
    return True


def cmd_seeds(cfg: dict) -> None:
    st = load_state()
    urls = [u.strip() for u in (ROOT / "config/seeds.txt").read_text(encoding="utf-8").splitlines()
            if u.strip() and not u.strip().startswith("#")]
    print(f"[seeds] {len(urls)} semillas")
    for u in urls:
        info = fetch_info(u)
        if info is None:
            vid = u.split("v=")[-1][:11]
            reject(st, vid, u, "metadata", "no accesible (privado, borrado o restringido por región)"); continue
        download_one(info, st, "seed", cfg)


def cmd_discover(cfg: dict, whisper: str) -> None:
    import yt_dlp

    st = load_state()
    accepted_new = sum(1 for v in st.values() if v.get("origin") == "search" and v.get("status") in {"downloaded", "extracted"})
    print(f"[discover-auto] ya aceptados por búsqueda: {accepted_new}; objetivo {cfg['target_new_videos']}")
    for q in cfg["queries"]:
        if accepted_new >= cfg["target_new_videos"]:
            break
        with yt_dlp.YoutubeDL(dict(YDL_BASE, extract_flat="in_playlist", ignoreerrors=True)) as y:
            res = y.extract_info(f"ytsearch{cfg['search_per_query']}:{q}", download=False) or {}
        for e in res.get("entries") or []:
            if not e or not e.get("id") or e["id"] in st:
                continue
            info = fetch_info(e.get("url") or f"https://www.youtube.com/watch?v={e['id']}")
            if info is None:
                continue
            if download_one(info, st, "search", cfg):
                # filtros de calidad inmediatos para contar solo videos útiles
                ok = extract_video(info["id"], st, cfg, whisper)
                accepted_new += int(ok)
            if accepted_new >= cfg["target_new_videos"]:
                break
    if accepted_new < cfg["min_new_videos"]:
        print(f"[discover-auto] AVISO: solo {accepted_new} videos nuevos pasaron los filtros "
              f"(mínimo {cfg['min_new_videos']}). Añade consultas en config/sources.yaml.")


# -------------------------------------------------------------------- extracción
_ASR = None


def _asr(model: str):
    global _ASR
    if _ASR is None:
        from core.extractors.linguistic import VerbatimASR
        _ASR = VerbatimASR(model)
    return _ASR


def extract_tracks(src: Path) -> tuple[Path, Path]:
    ff = _ffmpeg()
    CLIPS.mkdir(parents=True, exist_ok=True)
    wav, mp4 = CLIPS / f"{src.stem}.wav", CLIPS / f"{src.stem}_10fps.mp4"
    if not wav.exists():
        run([ff, "-y", "-i", str(src), "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE), "-c:a", "pcm_s16le", str(wav)])
    if not mp4.exists():
        run([ff, "-y", "-i", str(src), "-an", "-vf", f"fps={VIDEO_FPS}", "-c:v", "libx264", "-preset", "veryfast",
             "-crf", "18", str(mp4)])
    return wav, mp4


def smooth_boxes(times: np.ndarray, boxes: np.ndarray, win_s: float) -> np.ndarray:
    """Media móvil (ventana win_s) sobre centro y tamaño de la caja; ignora NaN."""
    out = np.full_like(boxes, np.nan)
    for i, t in enumerate(times):
        m = (np.abs(times - t) <= win_s / 2) & ~np.isnan(boxes[:, 0])
        if m.any():
            out[i] = boxes[m].mean(0)
    return out


def split_on_jumps(times, boxes, a, b, W, jump_frac, jump_win) -> list[tuple[float, float]]:
    """Corta [a,b] donde el centro de la caja salta > jump_frac*W en < jump_win s (cambio de cámara)."""
    m = (times >= a) & (times <= b) & ~np.isnan(boxes[:, 0])
    t, bx = times[m], boxes[m]
    if len(t) < 2:
        return [(a, b)]
    cx = (bx[:, 0] + bx[:, 2]) / 2 * W
    cuts = [a]
    for i in range(1, len(t)):
        j = i - 1
        while j > 0 and t[i] - t[j - 1] <= jump_win:
            j -= 1
        if abs(cx[i] - cx[j]) > jump_frac * W and t[i] - t[j] <= jump_win + 1e-6:
            cuts.append(float(t[i]))
    cuts.append(b)
    return [(cuts[k], cuts[k + 1]) for k in range(len(cuts) - 1)]


def square_crop(box, W, H, scale):
    cx, cy = (box[0] + box[2]) / 2 * W, (box[1] + box[3]) / 2 * H
    side = int(min(max((box[2] - box[0]) * W, (box[3] - box[1]) * H) * scale, W, H)) // 2 * 2
    x0 = int(np.clip(cx - side / 2, 0, W - side)); y0 = int(np.clip(cy - side / 2 + 0.12 * side, 0, H - side))
    return x0, y0, side


def write_dynamic_crop(mp4: Path, out: Path, a: float, b: float, times, sboxes, scale: float, min_side: int) -> dict:
    """Recorte cuadrado que sigue al orador (cajas suavizadas), escalado sin distorsión a >= min_side."""
    import cv2

    cap = cv2.VideoCapture(str(mp4))
    W, H = int(cap.get(3)), int(cap.get(4))
    valid = ~np.isnan(sboxes[:, 0])
    sides = [square_crop(bx, W, H, scale)[2] for bx in sboxes[valid & (times >= a) & (times <= b)]]
    out_side = max(min_side, int(np.median(sides)) // 2 * 2) if sides else min_side
    cap.set(cv2.CAP_PROP_POS_MSEC, a * 1000)
    tmp = out.with_suffix(".tmp.mp4")
    wr = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), VIDEO_FPS, (out_side, out_side))
    n, last = 0, None
    while True:
        ok, frame = cap.read()
        t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        if not ok or t > b + 1e-3:
            break
        i = int(np.argmin(np.abs(times - t)))
        bx = sboxes[i] if valid[i] else last
        if bx is None:
            continue
        last = bx
        x0, y0, side = square_crop(bx, W, H, scale)
        crop = frame[y0:y0 + side, x0:x0 + side]
        wr.write(cv2.resize(crop, (out_side, out_side), interpolation=cv2.INTER_CUBIC if side < out_side else cv2.INTER_AREA))
        n += 1
    wr.release(); cap.release()
    # re-codificar a H.264 para compatibilidad con ELAN / reproductores
    ff = _ffmpeg()
    run([ff, "-y", "-i", str(tmp), "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-r", str(VIDEO_FPS), str(out)])
    tmp.unlink(missing_ok=True)
    return dict(frames=n, out_side=out_side, src_size=[W, H])


def extract_video(vid: str, st: dict, cfg: dict, whisper: str) -> bool:
    """Filtros de calidad + segmentación + recorte. True si produce >= 1 segmento."""
    from core.extractors.acoustic import extract_acoustic, load_wav
    from core.extractors.kinesic import KinesicExtractor, continuous_face_segments

    q = cfg["quality"]
    ent = st.get(vid, {})
    if ent.get("status") == "extracted":
        return True
    if ent.get("status") != "downloaded":
        return False
    url = ent.get("url", "")
    src = RAW / f"{vid}.mp4"
    try:
        wav, mp4 = extract_tracks(src)
    except Exception as exc:
        reject(st, vid, url, "ffmpeg", str(exc)[:200]); return False

    audio, sr = load_wav(str(wav))
    # (e) habla vs música: proporción de habla según Silero
    ac = extract_acoustic(audio[: sr * 600], sr)          # primeros 10 min bastan para el filtro
    speech = sum(b - a for a, b in ac.speech_segments) / max(len(audio[: sr * 600]) / sr, 1e-6)
    if ac.vad_backend == "silero" and speech < q["min_speech_ratio"]:
        reject(st, vid, url, "speech_ratio", f"{speech:.2f} < {q['min_speech_ratio']} (música/silencio dominante)"); return False
    # (b) idioma sobre los primeros 30 s de HABLA (evita intros musicales)
    onset = ac.speech_segments[0][0] if ac.speech_segments else 0.0
    lang, prob = _asr(whisper).detect_language(audio[int(onset * sr):])
    if lang != q["language"] or prob < q["min_language_prob"]:
        reject(st, vid, url, "language", f"{lang} p={prob:.2f}"); return False
    # (c)(d) rostro frontal y multi-rostro (pasada rápida 5 fps, hasta 2 rostros)
    kx = KinesicExtractor(num_faces=2)
    try:
        track, face_report = kx.process_video_tracked(str(mp4), target_fps=5)
    finally:
        kx.close()
    frontal_rate = float(track.frontal.mean()) if len(track.frontal) else 0.0
    if frontal_rate < q["min_frontal_face_rate"]:
        reject(st, vid, url, "face", f"tasa frontal {frontal_rate:.2f} < {q['min_frontal_face_rate']}"); return False

    import cv2
    cap = cv2.VideoCapture(str(mp4)); W = int(cap.get(3)); cap.release()
    sboxes = smooth_boxes(track.times, track.bboxes, q["crop_smooth_s"])
    base = continuous_face_segments(track, min_len_s=q["min_segment_s"], max_gap_s=q["max_face_gap_s"], require_frontal=True)
    segs = [s for a, b in base for s in split_on_jumps(track.times, track.bboxes, a, b, W, q["jump_frac"], q["jump_window_s"])]
    segs = [(a, b) for a, b in segs if b - a >= q["min_segment_s"]]
    # trocear segmentos largos (<= max_segment_s) para ELAN, reanudación y estratificación por orador
    mx = q.get("max_segment_s", 300)
    segs = [(a + k * mx, min(b, a + (k + 1) * mx)) for a, b in segs for k in range(int(np.ceil((b - a) / mx)))]
    segs = [(a, b) for a, b in segs if b - a >= q["min_segment_s"]]
    ff = _ffmpeg()
    SEGS.mkdir(parents=True, exist_ok=True)
    info_p = RAW / f"{vid}.info.json"
    info = json.loads(info_p.read_text(encoding="utf-8")) if info_p.exists() else {}
    entries, dropped = [], []
    for k, (a, b) in enumerate(segs):
        m = (track.times >= a) & (track.times <= b)
        multi = float((track.n_faces[m] >= 2).mean()) if track.n_faces is not None and m.any() else 0.0
        if multi > q["max_multi_face_rate"]:
            dropped.append(f"seg{k}: {multi:.0%} con 2+ rostros"); continue
        sid = f"{vid}_s{k:02d}"
        seg_wav, seg_mp4 = SEGS / f"{sid}.wav", SEGS / f"{sid}.mp4"
        if not seg_wav.exists():
            run([ff, "-y", "-ss", f"{a:.3f}", "-to", f"{b:.3f}", "-i", str(wav), "-c:a", "pcm_s16le", str(seg_wav)])
        crop_info = {}
        if not seg_mp4.exists():
            crop_info = write_dynamic_crop(mp4, seg_mp4, a, b, track.times, sboxes, q["crop_scale"], q["out_min_side"])
        det = float(track.detected[m].mean()) if m.any() else 0.0
        entries.append(dict(recording_id=sid, source_video=vid, speaker_id=vid, url=url, title=info.get("title", ""),
                            channel=info.get("channel", ""), license=info.get("license") or "no declarada",
                            upload_date=info.get("upload_date", ""), offset_s=round(a, 3), duration_s=round(b - a, 3),
                            wav=seg_wav.relative_to(ROOT).as_posix(), mp4=seg_mp4.relative_to(ROOT).as_posix(),
                            face_detection_rate=round(det, 3), frontal_rate=round(float(track.frontal[m].mean()), 3),
                            multi_face_rate=round(multi, 3), crop=crop_info, language=lang, language_prob=round(prob, 3),
                            speech_ratio=round(speech, 3), face_tracking_method="dominant_iou",
                            face_track_id=face_report["selected"],
                            face_dominance_margin=round(face_report["dominance_margin"], 3),
                            face_review_required=face_report["dominance_margin"] < 0.25))
    if not entries:
        reject(st, vid, url, "segments", "sin segmentos válidos" + (f" ({'; '.join(dropped)})" if dropped else "")); return False
    update_index(entries)
    st[vid] = dict(st[vid], status="extracted", n_segments=len(entries), frontal_rate=round(frontal_rate, 3),
                   dropped_segments=dropped)
    save_state(st)
    print(f"   ✓ {vid}: {len(entries)} segmentos ({sum(e['duration_s'] for e in entries):.0f}s útiles)")
    return True


def update_index(entries: list[dict]) -> None:
    idx_path = SEGS / "index.json"
    idx = json.loads(idx_path.read_text(encoding="utf-8")) if idx_path.exists() else []
    known = {e["recording_id"]: i for i, e in enumerate(idx)}
    for e in entries:
        if e["recording_id"] in known:
            idx[known[e["recording_id"]]] = e
        else:
            idx.append(e)
    idx_path.write_text(json.dumps(idx, ensure_ascii=False, indent=2), encoding="utf-8")


def cmd_extract(cfg: dict, whisper: str) -> None:
    st = load_state()
    todo = [v for v, e in st.items() if e.get("status") == "downloaded"]
    order = sorted(todo, key=lambda v: (st[v].get("origin") != "seed", v))
    print(f"[extract] {len(order)} videos pendientes")
    for v in order:
        try:
            extract_video(v, st, cfg, whisper)
        except Exception as exc:
            reject(st, v, st[v].get("url", ""), "extract", f"{type(exc).__name__}: {str(exc)[:180]}")


def summary() -> dict:
    st = load_state()
    out = {"extracted": 0, "downloaded": 0, "rejected": 0}
    for e in st.values():
        out[e.get("status", "?")] = out.get(e.get("status", "?"), 0) + 1
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["seeds", "discover-auto", "extract", "all", "status"])
    ap.add_argument("--config", default=str(ROOT / "config/sources.yaml"))
    ap.add_argument("--whisper", default="small")
    a = ap.parse_args()
    cfg = cfg_load(a.config)
    if a.stage in ("seeds", "all"):
        cmd_seeds(cfg)
    if a.stage in ("extract", "all"):
        cmd_extract(cfg, a.whisper)            # procesa semillas primero
    if a.stage in ("discover-auto", "all"):
        cmd_discover(cfg, a.whisper)
    if a.stage in ("extract", "all"):
        cmd_extract(cfg, a.whisper)
    print("[estado]", summary())


if __name__ == "__main__":
    main()
