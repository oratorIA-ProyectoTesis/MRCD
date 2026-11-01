#!/usr/bin/env python3
"""Ensambla data/windows_<source>.npz a partir del manifest (etiquetas auto o humanas)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mrcd.constants import (FEATURE_SCHEMA_VERSION,  # noqa: E402
                            KINESIC_SPEAKER_NORM_MIN_FRAMES, LABELS)
from mrcd.dataset import build_recording_windows, speaker_kinesic_norm  # noqa: E402


def speaker_kinesic_stats(data: dict) -> dict[str, dict]:
    """Media y desviacion por canal de cada hablante, sobre sus frames reales."""
    out = {}
    for spk in np.unique(data["group"]):
        rows = data["group"] == spk
        kin, mask = data["kin"][rows], data["kmask"][rows].astype(bool)
        if mask.sum() < KINESIC_SPEAKER_NORM_MIN_FRAMES:
            continue
        flat = kin[mask]
        mu, sd = flat.mean(0), flat.std(0)
        if not np.all(np.isfinite(mu)) or np.any(sd < 1e-6):
            continue
        out[str(spk)] = {
            "mu": mu.tolist(),
            "sd": sd.tolist(),
            "n_frames": int(mask.sum()),
        }
    return out


def apply_stats(data: dict, stats: dict[str, dict]) -> tuple[dict, list[str]]:
    """Aplica estadisticas YA calculadas. -> (data, hablantes sin estadisticas)

    ◆ Por que existe esta funcion y no se recalcula por npz: el conjunto gold es un
    subconjunto muestreado, asi que sus estadisticas por hablante NO coinciden con
    las del conjunto de entrenamiento. Recalcular en cada npz dejaria al mismo
    hablante con dos transformaciones distintas a ambos lados del corte, y el modelo
    veria escalas incompatibles entre lo que aprendio y lo que se le evalua.
    """
    skipped = []
    for spk in np.unique(data["group"]):
        st = stats.get(str(spk))
        if st is None:
            skipped.append(str(spk))
            continue
        rows = data["group"] == spk
        mu = np.asarray(st["mu"], np.float32)
        sd = np.asarray(st["sd"], np.float32)
        data["kin"][rows] = ((data["kin"][rows] - mu) / sd).astype(np.float32)
    return data, skipped


def apply_speaker_kinesic_norm(data: dict) -> tuple[dict, list[str]]:
    """Requisito 8 del goal: z-score cinesico por hablante sobre todas sus ventanas.

    Se hace aqui y no en `build_recording_windows` porque la calibracion necesita
    TODO el material del hablante: una grabacion suelta son pocos segundos y sus
    estadisticas no representan la cara en reposo de esa persona.

    NO se toca la rama linguistica: sus rasgos binarios dispersos tienen desviacion
    casi nula en algunos hablantes y el z-score los convierte en una firma de
    identidad (medido con identity_probe.py: 3.6x -> 11.3x el azar, empeora).
    """
    skipped = []
    for spk in np.unique(data["group"]):
        rows = data["group"] == spk
        kin, ok = speaker_kinesic_norm(data["kin"][rows], data["kmask"][rows])
        if ok:
            data["kin"][rows] = kin
        else:
            skipped.append(str(spk))
    return data, skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(ROOT / "data/dataset_manifest.json"))
    ap.add_argument("--labels", choices=["auto", "human", "gold"], default="auto")
    ap.add_argument(
        "--av-offset-ms",
        type=float,
        default=0.0,
        help="desfase A/V calibrado (video retrasado > 0)",
    )
    ap.add_argument(
        "--kinesic-norm",
        choices=["speaker", "raw"],
        default="speaker",
        help="'speaker' (por defecto) aplica z-score cinesico por hablante. "
        "'raw' lo desactiva y deja el vector 12-D crudo: solo para reproducir "
        "resultados anteriores o para el experimento de fuga del pre-registro.",
    )
    ap.add_argument("--norm-stats", default=None,
                    help="JSON con las estadisticas de normalizacion por hablante. Si existe se "
                         "REUTILIZA (imprescindible para que train y test compartan transformacion); "
                         "si no, se calcula y se escribe. Por defecto, junto al npz de salida.")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    man = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    parts = [
        build_recording_windows(r, ROOT, args.labels, args.av_offset_ms)
        for r in man["recordings"]
    ]
    parts = [p for p in parts if len(p["y"])]
    if not parts:
        sys.exit("Sin ventanas.")
    # alinear ancho de rasgos lingüísticos (por si unas grabaciones tienen RoBERTa y otras no)
    F = min(p["ling"].shape[2] for p in parts)
    data = {
        k: np.concatenate([p[k][..., :F] if k == "ling" else p[k] for p in parts])
        for k in parts[0]
    }

    skipped: list[str] = []
    if args.kinesic_norm == "speaker":
        stats_p = Path(
            args.norm_stats
            or (Path(args.out).parent if args.out else ROOT / "data")
            / "kinesic_norm_stats.json"
        )
        if stats_p.exists():
            stats = json.loads(stats_p.read_text(encoding="utf-8"))["speakers"]
            print(
                f"  estadisticas de normalizacion reutilizadas de {stats_p.name} "
                f"({len(stats)} hablantes)"
            )
        else:
            stats = speaker_kinesic_stats(data)
            stats_p.parent.mkdir(parents=True, exist_ok=True)
            stats_p.write_text(
                json.dumps(
                    {
                        "schema_version": FEATURE_SCHEMA_VERSION,
                        "source": Path(args.manifest).name,
                        "built_from_labels": args.labels,
                        "speakers": stats,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            print(
                f"  estadisticas de normalizacion escritas en {stats_p.name} "
                f"({len(stats)} hablantes)"
            )
        data, skipped = apply_stats(data, stats)

    out = Path(args.out or ROOT / f"data/windows_{args.labels}.npz")
    src = {"auto": "auto", "human": "human", "gold": "gold_llm"}[args.labels]
    np.savez_compressed(
        out,
        label_source=np.array(src),
        labels=np.array(LABELS),
        schema_version=np.array(FEATURE_SCHEMA_VERSION),
        kinesic_norm=np.array(args.kinesic_norm),
        **data,
    )
    counts = {l: int((data["y"] == i).sum()) for i, l in enumerate(LABELS)}
    print(
        f"{out}: {len(data['y'])} ventanas | esquema v{FEATURE_SCHEMA_VERSION} | "
        f"norm cinesica: {args.kinesic_norm} | con video: {data['has_video'].mean():.0%}"
    )
    if skipped:
        print(
            f"  AVISO: sin normalizar por falta de frames reales: {', '.join(skipped)}"
        )
    print(f"  por clase: {counts}")


if __name__ == "__main__":
    main()
