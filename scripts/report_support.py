#!/usr/bin/env python3
"""Reporte honesto de soporte por clase -> results/class_support.md (y .json)."""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.constants import LABELS, SMOKE_HEADER, TAXONOMY  # noqa: E402

WHY_ZERO = {  # hipótesis a verificar, no conclusiones
    "block": "Requiere pausa tras palabra funcional + tensión perioral o de entrecejo; es típica de habla tartamuda y rara en oradores entrenados. También depende de una buena detección facial durante el silencio.",
    "prolongation": "Exige ≥ 350 ms sostenidos con ΔF0 < 5 Hz (umbral estricto) fuera de cierre de frase; los oradores ensayados alargan poco y pYIN puede no seguir voces con mucha variación.",
    "repetition": "Depende de que Whisper transcriba literalmente; el ASR tiende a borrar repeticiones.",
    "revision": "Requiere marcador explícito tras cláusula incompleta; muchas autocorrecciones no usan marcador.",
    "rhetorical_pause": "Requiere 0.8–2.5 s en frontera con rostro en reposo; si no hay video o el rostro no es frontal durante la pausa, cae en neutral_pause.",
    "filler_word": "Whisper puede omitir muletillas; revisar la tasa de inserciones/omisiones en el manifest.",
    "neutral_pause": "Pocos silencios ≥ 600 ms detectados; revisar el VAD.",
}


def main():
    man = json.loads((ROOT / "data/dataset_manifest.json").read_text(encoding="utf-8"))
    recs = man["recordings"]
    ev_tot, low_tot = Counter(), Counter()
    per_spk = defaultdict(Counter)
    examples = defaultdict(list)
    for r in recs:
        events = json.loads((ROOT / r["events_json"]).read_text(encoding="utf-8"))
        for e in events:
            if e.get("source") == "auto_low_conf":
                low_tot[e["category"]] += 1; continue
            ev_tot[e["category"]] += 1
            per_spk[r["speaker_id"]][e["category"]] += 1
            examples[e["category"]].append((e["confidence"], r, e))
    win = {}
    wpath = ROOT / "data/windows_auto.npz"
    if wpath.exists():
        y = np.load(wpath)["y"]
        win = {l: int((y == i).sum()) for i, l in enumerate(LABELS)}

    L = [f"# Soporte por clase — {SMOKE_HEADER}", "",
         f"Grabaciones: {len(recs)} · Oradores: {len(per_spk)} · Duración total: {sum(r['duration_s'] for r in recs)/60:.1f} min", "",
         "| Clase | Eventos (conf ≥ 0.5) | Ventanas (primaria) | Candidatos baja conf. (solo revisión) | Oradores con ≥ 1 |",
         "|---|---|---|---|---|"]
    for c in TAXONOMY:
        L.append(f"| {c} | {ev_tot[c]} | {win.get(c, '—')} | {low_tot[c]} | {sum(1 for s in per_spk.values() if s[c])} |")
    L.append(f"| fluent | — | {win.get('fluent', '—')} | — | — |")
    L += ["", "## Clases con soporte 0 (hipótesis)", ""]
    zeros = [c for c in TAXONOMY if ev_tot[c] == 0]
    L += [f"- **{c}**: {WHY_ZERO[c]}" for c in zeros] or ["- Ninguna."]
    L += ["", "## Por orador (eventos)", "", "| Orador | " + " | ".join(TAXONOMY) + " |", "|---|" + "---|" * len(TAXONOMY)]
    for s, cnt in sorted(per_spk.items()):
        L.append(f"| {s} | " + " | ".join(str(cnt[c]) for c in TAXONOMY) + " |")
    L += ["", "## Ejemplos para verificación auditiva (3 por clase, mayor confianza)", ""]
    for c in TAXONOMY:
        ex = sorted(examples[c], key=lambda x: -x[0])[:3]
        if not ex:
            continue
        L.append(f"**{c}**")
        for conf, r, e in ex:
            abs_s = r.get("offset_s", 0) + e["start_ms"] / 1000
            url = r.get("url", "")
            link = f"{url}&t={int(abs_s)}s" if "youtube.com" in url else url
            L.append(f"- `{r['recording_id']}` {e['start_ms']/1000:.2f}–{e['end_ms']/1000:.2f} s (video original ≈ {abs_s:.1f} s) "
                     f"· conf {conf:.2f} · «{e.get('label','')}» · {link}")
        L.append("")
    out = ROOT / "results"; out.mkdir(exist_ok=True)
    (out / "class_support.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    (out / "class_support.json").write_text(json.dumps(dict(events=ev_tot, low_conf=low_tot, windows=win,
                                                            zero_classes=zeros), indent=2), encoding="utf-8")
    print("\n".join(L[:14]))


if __name__ == "__main__":
    main()
