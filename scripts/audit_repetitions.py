#!/usr/bin/env python3
"""Cuantas repeticiones nuevas son artefactos de costura del troceado del ASR.

Contexto. Transcribir en trozos de 8 s con 2 s de solape subio los eventos de
`repetition` de 92 a 228, pero el acuerdo del juez con esa clase CAYO del 85 % al
67 % (results/v4/judge_agreement.md). La hipotesis es que parte de las
repeticiones nuevas no las dijo el hablante: son la misma palabra transcrita a
ambos lados de una costura, con tiempos ligeramente distintos, que la
deduplicacion por region no llega a eliminar.

Geometria de las costuras. Con `chunk_s=8` y `overlap_s=2` el paso es 6 s. El
trozo k cubre [6k, 6k+8] y aporta las palabras cuyo centro cae en [6k+2, 6k+8]
(el trozo 0 aporta [0, 8]). Los limites de region quedan por tanto en
8, 14, 20, 26, ... es decir 8 + 6k.

Una repeticion es SOSPECHOSA si un limite cae entre el final del primer token y
el inicio del segundo: eso es justo lo que produce una costura. Es un indicio, no
una prueba: un hablante tambien puede repetir una palabra exactamente ahi.

Contraste independiente. Una repeticion que YA existia en v3 (transcripcion del
archivo entero, sin costuras) no puede ser artefacto del troceado. La fraccion de
repeticiones nuevas que ademas cruzan costura es la estimacion buscada.

Uso:
    python scripts/audit_repetitions.py --a data/features --b data/v4/features
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load_events(feat_dir: Path, category: str = "repetition") -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for p in sorted(feat_dir.glob("*.events.json")):
        rid = p.name[: -len(".events.json")]
        try:
            evs = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        out[rid] = [e for e in evs if e.get("category") == category]
    return out


def seam_times(duration_s: float, chunk_s: float, overlap_s: float) -> list[float]:
    """Limites de region: 8, 14, 20, ... para chunk=8, overlap=2."""
    step = chunk_s - overlap_s
    seams, k = [], 0
    while True:
        t = chunk_s + step * k
        if t >= duration_s:
            break
        seams.append(t)
        k += 1
    return seams


def crosses_seam(ev: dict, seams: list[float], tol: float = 0.0) -> float | None:
    """Devuelve el limite que cae DENTRO del evento, o None."""
    for t in seams:
        if ev["start"] - tol <= t <= ev["end"] + tol:
            return t
    return None


def overlaps(a: dict, b: dict, tol: float = 0.35) -> bool:
    """Mismo evento en las dos extracciones (los tiempos no son identicos)."""
    return not (a["end"] + tol < b["start"] or b["end"] + tol < a["start"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--a", default=str(ROOT / "data/features"), help="extraccion de referencia (v3)"
    )
    ap.add_argument(
        "--b", default=str(ROOT / "data/v4/features"), help="extraccion troceada (v4)"
    )
    ap.add_argument("--manifest-b", default=str(ROOT / "data/v4/dataset_manifest.json"))
    ap.add_argument("--chunk-s", type=float, default=8.0)
    ap.add_argument("--overlap-s", type=float, default=2.0)
    ap.add_argument("--out", default=str(ROOT / "results/repetition_audit.md"))
    a = ap.parse_args()

    A, B = load_events(Path(a.a)), load_events(Path(a.b))
    man = {
        r["recording_id"]: r
        for r in json.loads(Path(a.manifest_b).read_text(encoding="utf-8"))[
            "recordings"
        ]
    }

    n_a = sum(len(v) for v in A.values())
    n_b = sum(len(v) for v in B.values())

    nuevos, viejos, cruzan_nuevos, cruzan_viejos = 0, 0, 0, 0
    ejemplos: list[tuple] = []
    for rid, evs in B.items():
        dur = man.get(rid, {}).get("duration_s", 0.0)
        seams = seam_times(dur, a.chunk_s, a.overlap_s)
        prev = A.get(rid, [])
        for e in evs:
            es_nuevo = not any(overlaps(e, o) for o in prev)
            seam = crosses_seam(e, seams)
            if es_nuevo:
                nuevos += 1
                if seam is not None:
                    cruzan_nuevos += 1
                    if len(ejemplos) < 15:
                        ejemplos.append((rid, e, seam))
            else:
                viejos += 1
                if seam is not None:
                    cruzan_viejos += 1

    pct = lambda x, n: f"{x / n * 100:.1f} %" if n else "—"
    L = [
        "# Auditoria de repeticiones: cuantas son costura del troceado",
        "",
        f"- Referencia (archivo entero): `{Path(a.a)}` -> **{n_a}** eventos",
        f"- Troceada ({a.chunk_s:g} s / solape {a.overlap_s:g} s): `{Path(a.b)}` -> **{n_b}** eventos",
        f"- Limites de region en 8, 14, 20, ... s (paso {a.chunk_s - a.overlap_s:g} s)",
        "",
        "> Cruzar un limite es un INDICIO, no una prueba: un hablante tambien puede",
        "> repetir una palabra justo ahi. Lo que lo vuelve informativo es comparar la",
        "> tasa entre eventos nuevos y eventos que ya existian sin costuras.",
        "",
        "| grupo | eventos | cruzan costura | tasa |",
        "|---|---|---|---|",
        f"| ya estaban en la referencia | {viejos} | {cruzan_viejos} | {pct(cruzan_viejos, viejos)} |",
        f"| **nuevos del troceado** | {nuevos} | **{cruzan_nuevos}** | **{pct(cruzan_nuevos, nuevos)}** |",
        "",
    ]

    base = cruzan_viejos / viejos if viejos else 0.0
    exceso = cruzan_nuevos - round(base * nuevos)
    L += [
        "## Lectura",
        "",
        f"Los eventos que ya existian sin trocear cruzan un limite el {pct(cruzan_viejos, viejos)} "
        "de las veces. Esa es la tasa base esperable por azar, porque los limites caen en "
        "posiciones arbitrarias del habla.",
        "",
        f"Si los eventos nuevos fueran repeticiones genuinas recuperadas, deberian cruzar "
        f"limites a esa misma tasa. Cruzan al {pct(cruzan_nuevos, nuevos)}.",
        "",
    ]
    if nuevos and viejos:
        if cruzan_nuevos / nuevos > base * 1.5:
            L += [
                f"**Exceso estimado: ~{max(exceso, 0)} de {nuevos} eventos nuevos "
                f"son atribuibles a costuras**, no al hablante. El troceado esta "
                "generando repeticiones falsas, coherente con la caida del acuerdo del "
                "juez en esta clase (85 % -> 67 %).",
                "",
            ]
        else:
            L += [
                "La tasa de los nuevos NO excede la base. Las repeticiones nuevas no se "
                "explican por costuras, y la caida del acuerdo del juez tiene otra causa "
                "que hay que buscar.",
                "",
            ]

    if ejemplos:
        L += [
            "## Ejemplos de eventos nuevos que cruzan un limite",
            "",
            "| grabacion | t | limite | tokens |",
            "|---|---|---|---|",
        ]
        for rid, e, seam in ejemplos:
            toks = " ".join(e.get("evidence", {}).get("tokens", []))[:40]
            L.append(
                f"| {rid} | {e['start']:.2f}–{e['end']:.2f} | {seam:.0f} s | `{toks}` |"
            )
        L.append("")

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
