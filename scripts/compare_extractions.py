#!/usr/bin/env python3
"""Compara dos extracciones del mismo corpus grabacion por grabacion.

Sirve para medir el efecto de un cambio de parametro (p. ej. `--asr-chunk-s`)
sobre la supervision debil, sin tener que confiar en la impresion de mirar dos
corridas por separado.

Lo que compara son ETIQUETAS HEURISTICAS, no exactitud: un aumento de eventos
significa que la cascada encontro mas candidatos, no que esos candidatos sean
correctos. La unica forma de saber si mejoro es el acuerdo con el juez ciego y,
en ultima instancia, con anotacion humana.

Uso:
    python scripts/compare_extractions.py --a data/dataset_manifest.json \
        --b data/v4/dataset_manifest.json --label-a "archivo entero" --label-b "trozos 8s"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mrcd.constants import TAXONOMY  # noqa: E402


def load(path: Path) -> dict[str, dict]:
    recs = json.loads(path.read_text(encoding="utf-8"))["recordings"]
    return {r["recording_id"]: r for r in recs}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default=str(ROOT / "data/dataset_manifest.json"))
    ap.add_argument("--b", required=True)
    ap.add_argument("--label-a", default="A")
    ap.add_argument("--label-b", default="B")
    ap.add_argument("--out", default=str(ROOT / "results/extraction_comparison.md"))
    args = ap.parse_args()

    A, B = load(Path(args.a)), load(Path(args.b))
    shared = sorted(set(A) & set(B))
    if not shared:
        sys.exit("Las dos extracciones no comparten ninguna grabacion.")

    tot_a = {c: 0 for c in TAXONOMY}
    tot_b = {c: 0 for c in TAXONOMY}
    words_a = words_b = 0
    dur = 0.0
    spk_with: dict[str, set[str]] = {c: set() for c in TAXONOMY}
    for rid in shared:
        ra, rb = A[rid], B[rid]
        words_a += ra.get("n_words", 0)
        words_b += rb.get("n_words", 0)
        dur += rb.get("duration_s", 0.0)
        for c in TAXONOMY:
            tot_a[c] += ra.get("event_counts", {}).get(c, 0)
            n = rb.get("event_counts", {}).get(c, 0)
            tot_b[c] += n
            if n:
                spk_with[c].add(rb.get("speaker_id", rid))

    chunk_b = next(
        (r.get("asr_chunk_s") for r in B.values() if r.get("asr_chunk_s")), "n/a"
    )
    L = [
        "# Comparacion de extracciones",
        "",
        f"- **{args.label_a}**: `{Path(args.a).name}`",
        f"- **{args.label_b}**: `{Path(args.b).name}` (asr_chunk_s = {chunk_b})",
        f"- Grabaciones comparadas: {len(shared)} · {dur / 60:.1f} min",
        f"- Palabras transcritas: {words_a} -> {words_b} "
        f"({(words_b - words_a) / max(words_a, 1) * 100:+.1f} %)",
        "",
        "> Estas son ETIQUETAS HEURISTICAS, no exactitud. Mas eventos significa que la",
        "> cascada encontro mas candidatos, no que acierte mas. Confirmarlo exige el",
        "> acuerdo con el juez ciego y con anotacion humana.",
        "",
        f"| Clase | {args.label_a} | {args.label_b} | cambio | oradores con >=1 |",
        "|---|---|---|---|---|",
    ]
    for c in TAXONOMY:
        a, b = tot_a[c], tot_b[c]
        d = b - a
        pct = f"{d / a * 100:+.0f} %" if a else ("nuevo" if b else "—")
        L.append(f"| {c} | {a} | {b} | **{d:+d}** ({pct}) | {len(spk_with[c])} |")
    L += [
        "",
        f"| **total** | {sum(tot_a.values())} | {sum(tot_b.values())} | "
        f"**{sum(tot_b.values()) - sum(tot_a.values()):+d}** | |",
        "",
    ]

    # Interpretacion de los dos efectos esperados, para que no se lean al reves.
    dr = tot_b["repetition"] - tot_a["repetition"]
    df = tot_b["filler_word"] - tot_a["filler_word"]
    dp = tot_b["prolongation"] - tot_a["prolongation"]
    L += [
        "## Como leer los dos efectos",
        "",
        f"**Repeticiones y muletillas ({dr:+d} y {df:+d}).** Recuperacion de recall en la",
        "capa ASR: Whisper borra repeticiones y muletillas cuando transcribe mucho",
        "contexto de una vez, y lo que no transcribe no puede etiquetarse porque",
        "`detect_repetitions` y `detect_fillers` operan sobre las palabras del ASR.",
        "",
        f"**Prolongaciones ({dp:+d}).** Si baja, es esperable y probablemente una mejora",
        "de PRECISION, no una perdida. `detect_prolongations` (auto_labeler.py:244)",
        "excluye toda palabra que sea muletilla o solape con una. Antes, un «eeeh» que",
        "el ASR borraba dejaba su alargamiento acustico atribuido a la palabra de",
        "contenido vecina como prolongacion. Ahora esa muletilla se transcribe, reclama",
        "el intervalo, y deja de contarse como prolongacion de otra cosa.",
        "",
        "**Prediccion falsable:** si esa lectura es correcta, el acuerdo del juez ciego",
        "con la clase `prolongation` deberia SUBIR respecto al 12/95 actual",
        "(results/judge_agreement.md), porque desaparecen falsos positivos que eran",
        "muletillas mal atribuidas. Si no sube, la lectura es incorrecta y el cambio",
        "solo movio ruido.",
        "",
    ]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L[:30]))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
