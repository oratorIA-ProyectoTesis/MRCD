#!/usr/bin/env python3
"""Paquete de anotación humana priorizado en clases raras -> data/annotation_batch_01/.

- 40–60 min de segmentos; prioriza densidad de candidatos (auto + baja confianza) de
  block, prolongation, repetition y revision.
- Estratificado: ningún orador supera el 15 % del tiempo total objetivo.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.constants import TAXONOMY  # noqa: E402

RARE = {"block": 3.0, "prolongation": 2.0, "repetition": 2.0, "revision": 2.0}

README = """# Lote de anotación {name}

Material para anotación humana en ELAN (https://archive.mpi.nl/tla/elan). Uso interno de investigación:
no redistribuir audio ni video.

## Cómo anotar
1. Abre cada `.eaf` en ELAN. El audio y el video están en la misma carpeta; si ELAN no los encuentra, usa
   *Edit → Linked Files* y apunta a los archivos de esta carpeta.
2. Tiers de apoyo (NO editar): `asr_words`, `vad_silence`, `auto_candidates`, `auto_evidence`, `auto_low_conf`.
3. Anota en **`human_disfluency`**, que usa el vocabulario controlado de 7 clases. Marca el intervalo exacto
   del evento. Los candidatos automáticos son solo sugerencias: acepta, corrige o ignora.
4. Clases:
   - `filler_word`: muletilla léxica o vocálica ("eh", "este", "o sea" sin función léxica).
   - `prolongation`: alargamiento anómalo de un sonido ("mmmuy", "ssssí"); no el alargamiento natural de fin de frase.
   - `repetition`: repetición de sonido, sílaba o palabra ("pe-pe-pero", "la la").
   - `block`: interrupción con tensión (silencio o sonido atascado), típicamente dentro de un constituyente.
   - `revision`: autocorrección o falso inicio ("fuimos a la… o sea, al cine").
   - `rhetorical_pause`: pausa deliberada en frontera de frase, con rostro relajado.
   - `neutral_pause`: silencio estructural sin tensión ni efecto retórico.
5. Anotación a ciegas: los dos anotadores trabajan por separado. Luego se calcula el acuerdo (κ).

## Después
`python scripts/build_windows.py --labels human` → `python scripts/run_ablation.py --data data/windows_human.npz`
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=50.0)
    ap.add_argument("--min-minutes", type=float, default=40.0)
    ap.add_argument("--max-minutes", type=float, default=60.0)
    ap.add_argument("--speaker-cap", type=float, default=0.15)
    ap.add_argument("--name", default="annotation_batch_01")
    a = ap.parse_args()

    man = json.loads((ROOT / "data/dataset_manifest.json").read_text(encoding="utf-8"))
    scored = []
    for r in man["recordings"]:
        ev = json.loads((ROOT / r["events_json"]).read_text(encoding="utf-8"))
        score = sum(RARE.get(e["category"], 0.2) * (1.0 if e.get("source") == "auto" else 0.5) for e in ev)
        scored.append((score / max(r["duration_s"] / 60, 0.5), r, ev))
    scored.sort(key=lambda x: -x[0])
    cap_s = a.speaker_cap * a.minutes * 60
    used, per_spk, chosen = 0.0, {}, []
    for dens, r, ev in scored:
        if used >= a.max_minutes * 60 or used >= a.minutes * 60:
            break
        spk = r["speaker_id"]
        if per_spk.get(spk, 0) + r["duration_s"] > cap_s:
            continue
        if used + r["duration_s"] > a.max_minutes * 60:
            continue
        chosen.append((dens, r, ev)); used += r["duration_s"]; per_spk[spk] = per_spk.get(spk, 0) + r["duration_s"]

    out = ROOT / "data" / a.name
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for dens, r, ev in chosen:
        for key in ("eaf", "wav", "mp4"):
            src = ROOT / r[key] if r.get(key) else None
            if src and src.exists():
                shutil.copy2(src, out / src.name)
        rows.append(dict(recording_id=r["recording_id"], speaker_id=r["speaker_id"], duration_s=r["duration_s"],
                         rare_density_per_min=round(dens, 2), url=r.get("url", ""), offset_s=r.get("offset_s", 0),
                         **{f"auto_{c}": sum(e["category"] == c and e.get("source") == "auto" for e in ev) for c in TAXONOMY}))
    if rows:
        with (out / "batch_manifest.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    (out / "README.md").write_text(README.format(name=a.name), encoding="utf-8")
    note = "" if used >= a.min_minutes * 60 else (f" — AVISO: solo {used/60:.1f} min disponibles con el tope de {a.speaker_cap:.0%} "
                                                   "por orador; hacen falta más oradores")
    print(f"[annotation] {len(rows)} segmentos, {used/60:.1f} min, {len(per_spk)} oradores -> {out}{note}")


if __name__ == "__main__":
    main()
