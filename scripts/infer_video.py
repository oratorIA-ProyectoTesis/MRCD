#!/usr/bin/env python3
"""Inferencia de los tres modos sobre un vídeo ingerido, con artefacto JSON.

Produce `data/results_<vid>.json` con una entrada por ventana y otra por evento.

Ventana vs. evento
------------------
Con ventanas de 3 s y paso de 0.5 s, cada instante del audio cae en unas seis
ventanas. Una sola muletilla aparece por tanto en ~6 ventanas consecutivas.
Contar ventanas multiplicaría los eventos por seis y daría una tasa de
disfluencias falsa. El resumen cuenta EVENTOS: rachas contiguas de ventanas con
la misma clase predicha, fundidas en un intervalo.

Uso:
    python scripts/infer_video.py --vid Ka_okSSytes
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.constants import (BACKGROUND, KIDX, PROSODY_FIELDS, STRIDE_S, VIDEO_FPS,
                            WINDOW_S)  # noqa: E402
from core.dataset import (acoustic_windows, center_f0, kinesic_windows, linguistic_windows,
                          window_starts)  # noqa: E402
from core.extractors.prosody import prosody_windows, speaker_f0_center  # noqa: E402
from core.models.fusion import MODES  # noqa: E402
from core.inference import InferenceEngine, load_ckpt, predict  # noqa: E402, F401
from core.contracts import validate_raw_features  # noqa: E402

CASE = ROOT / "data/case"
PIDX = {n: i for i, n in enumerate(PROSODY_FIELDS)}


def load_reporting_engine(path: Path, expected_mode: str, device: str,
                          expected_labels: list[str] | None = None) -> InferenceEngine:
    """Reject renamed or legacy checkpoints before composing a comparative report."""
    engine = InferenceEngine.from_checkpoint(path, device)
    if engine.net.mode != expected_mode:
        raise ValueError(f"{path}: checkpoint mode {engine.net.mode!r} does not match requested {expected_mode!r}")
    if engine.provenance.get("contract_status") != "versioned":
        raise ValueError(f"{path}: unversioned checkpoint cannot be reported; retrain with train_checkpoints.py")
    if expected_labels is not None and list(engine.labels) != list(expected_labels):
        raise ValueError(f"{path}: checkpoint label order differs across modes")
    return engine


def mouth_press_peak(kin_w: np.ndarray, kmask_w: np.ndarray) -> float:
    """Pico de tensión labial sobre la línea base de la propia ventana.

    Media de mouth_press izquierda/derecha; se resta el percentil 10 de la
    ventana para medir el ASCENSO de tensión, no el tono muscular de reposo,
    que varía entre personas.
    """
    if kmask_w is None or not kmask_w.any():
        return 0.0
    mp = kin_w[kmask_w][:, [KIDX["mouth_press_left"], KIDX["mouth_press_right"]]].mean(1)
    if mp.size < 3 or not np.isfinite(mp).any():
        return 0.0
    mp = mp[np.isfinite(mp)]
    return float(mp.max() - np.percentile(mp, 10))


def gaze_stability(kin_w: np.ndarray, kmask_w: np.ndarray) -> float:
    """Desviación típica de la mirada. Valores bajos = mirada fija (posible bloqueo)."""
    if kmask_w is None or not kmask_w.any():
        return float("nan")
    cols = [KIDX["eye_look_in_left"], KIDX["eye_look_out_right"],
            KIDX["eye_look_up_left"], KIDX["eye_look_down_right"]]
    g = kin_w[kmask_w][:, cols]
    g = g[np.isfinite(g).all(1)]
    return float(g.std(0).mean()) if len(g) >= 3 else float("nan")


def visual_metrics(kin_w: np.ndarray, kmask_w: np.ndarray, has_video: bool) -> tuple[float | None, float | None]:
    """Do not report visual measurements below the real-frame coverage gate."""
    if not has_video:
        return None, None
    gaze = gaze_stability(kin_w, kmask_w)
    return round(mouth_press_peak(kin_w, kmask_w), 4), (round(gaze, 4) if np.isfinite(gaze) else None)


def window_text(words: list[dict], t0: float, t1: float) -> str:
    return " ".join(w["text"] for w in words
                    if (w["start"] + w["end"]) / 2 >= t0 and (w["start"] + w["end"]) / 2 < t1).strip()


def merge_events(starts: np.ndarray, cls: np.ndarray, probs: np.ndarray,
                 labels: list[str]) -> list[dict]:
    """Funde ventanas contiguas de la misma clase en un solo evento."""
    ev, i, n = [], 0, len(cls)
    while i < n:
        if labels[cls[i]] == BACKGROUND:
            i += 1
            continue
        j = i
        while j + 1 < n and cls[j + 1] == cls[i] and starts[j + 1] - starts[j] <= STRIDE_S * 1.5:
            j += 1
        ev.append({"class": labels[cls[i]],
                   "start": float(starts[i]), "end": float(starts[j] + WINDOW_S),
                   "n_windows": j - i + 1,
                   "peak_prob": float(probs[i:j + 1, cls[i]].max())})
        i = j + 1
    return ev


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vid", required=True)
    ap.add_argument("--models", default=str(ROOT / "models"))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    meta_p = CASE / f"{a.vid}.meta.json"
    if not meta_p.exists():
        print(f"[infer] falta {meta_p}. Corre antes scripts/ingest_single_video.py")
        sys.exit(1)
    meta = json.loads(meta_p.read_text(encoding="utf-8"))
    feats = np.load(ROOT / meta["features_npz"], allow_pickle=False)
    validate_raw_features(feats, path=str(ROOT / meta["features_npz"]))
    words = json.loads((ROOT / meta["words_json"]).read_text(encoding="utf-8"))

    mdir = Path(a.models)
    missing = [m for m in MODES if not (mdir / f"ckpt_{m}.pt").exists()]
    if missing:
        print(f"[infer] faltan checkpoints: {', '.join(missing)}\n"
              f"        Corre antes: python scripts/train_checkpoints.py")
        sys.exit(1)

    # ----------------------------------------------------- ventanas (sin etiqueta)
    frames = feats["ac_frames"]
    starts = window_starts(float(meta["duration_s"]))
    f0c = speaker_f0_center(frames)
    pros = prosody_windows(frames, starts, f0c)
    ac = acoustic_windows(center_f0(frames, f0c), starts)
    ling, lpos, lmask = linguistic_windows(words, feats["word_feats"], starts)
    kin, kmask, hv = kinesic_windows(feats["k_times"], feats["k_vectors"], starts)
    d = dict(ac=ac, pros=pros, ling=ling, lpos=lpos, lmask=lmask, kin=kin, kmask=kmask,
             has_video=hv)
    print(f"[infer] {len(starts)} ventanas · con rostro seguido: {hv.mean():.0%}")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out_pred, labels, prov, provenance_by_mode = {}, None, {}, {}
    engines = {}
    for mode in MODES:
        engine = load_reporting_engine(mdir / f"ckpt_{mode}.pt", mode, dev, labels)
        engines[mode] = engine
        labels, prov = engine.labels, engine.provenance
        provenance_by_mode[mode] = prov
    for mode, engine in engines.items():
        out_pred[mode] = engine.predict(d)
        print(f"[infer] {mode}: listo")

    dang_col = ling.shape[-1] - 1        # última columna del ramal lingüístico
    rows = []
    for i, t0 in enumerate(starts):
        t1 = t0 + WINDOW_S
        mouth_peak, gaze = visual_metrics(kin[i], kmask[i], bool(hv[i]))
        rec = {
            "timestamp_start": round(float(t0), 3),
            "timestamp_end": round(float(t1), 3),
            "transcription": window_text(words, t0, t1),
            "features": {
                "f0_reset": round(float(pros[i, PIDX["f0_reset"]]), 4),
                "max_delta_mouth_press": mouth_peak,
                "is_dangling": bool(ling[i, :, dang_col].max() > 0.5),
            },
            "predictions": {m: {"class": labels[int(out_pred[m][i].argmax())],
                                "prob": round(float(out_pred[m][i].max()), 4)}
                            for m in MODES},
        }
        rec["features_extra"] = {
            "sil_dur": round(float(pros[i, PIDX["sil_dur"]]), 4),
            "f0_slope_pre": round(float(pros[i, PIDX["f0_slope_pre"]]), 4),
            "gaze_stability": gaze,
            "has_face": bool(hv[i]),
        }
        rows.append(rec)

    events = {m: merge_events(starts, out_pred[m].argmax(1), out_pred[m], labels) for m in MODES}
    disagree = int(sum(1 for i in range(len(starts))
                       if len({out_pred[m][i].argmax() for m in MODES}) > 1))

    art = {
        "video_id": a.vid, "url": meta.get("url"), "title": meta.get("title"),
        "duration_s": meta.get("duration_s"), "n_windows": len(starts),
        "window_s": WINDOW_S, "stride_s": STRIDE_S, "video_fps": VIDEO_FPS,
        "labels": labels,
        "face_tracking": {"coverage": meta.get("face_coverage"),
                          "dominance_margin": meta.get("dominance_margin")},
        "model_provenance": prov,
        "model_provenance_by_mode": provenance_by_mode,
        "window_disagreement_rate": round(disagree / max(len(starts), 1), 4),
        "events": events,
        "windows": rows,
    }
    out = Path(a.out) if a.out else ROOT / f"data/results_{a.vid}.json"
    out.write_text(json.dumps(art, indent=1, ensure_ascii=False), encoding="utf-8")

    print(f"\n[infer] artefacto -> {out}  ({out.stat().st_size / 1e6:.1f} MB)")
    print(f"[infer] ventanas en que los tres modos NO coinciden: {disagree} "
          f"({disagree / max(len(starts), 1):.1%})")
    print("\nEVENTOS POR MODO (ventanas contiguas fundidas)")
    allc = sorted({e["class"] for m in MODES for e in events[m]})
    w = max([len(c) for c in allc] + [10])
    print(f"  {'clase'.ljust(w)} {'audio_only':>11} {'audio_text':>11} {'trimodal':>11}")
    for c in allc:
        print(f"  {c.ljust(w)} " + "".join(
            f"{sum(1 for e in events[m] if e['class'] == c):>11}" + " " * 1 for m in MODES))
    print(f"  {'TOTAL'.ljust(w)} " + "".join(f"{len(events[m]):>11} " for m in MODES))
    if prov.get("warning"):
        print(f"\n  AVISO DE PROCEDENCIA: {prov['warning']}")


if __name__ == "__main__":
    main()
