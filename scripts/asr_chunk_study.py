#!/usr/bin/env python3
"""Cuanta disfluencia sobrevive a la transcripcion, segun el tamano de trozo.

Hallazgo que motiva este script: el mismo audio transcrito en trozos cortos
solapados conserva «la, la, la primera», y transcrito de una sola pasada se
convierte en «La primera». La repeticion no la pierde el clasificador: la borra
el ASR antes de que el clasificador la vea.

La causa es el objetivo con el que se entrena Whisper: producir texto legible.
Cuanto mas contexto ve de una vez, mas «corrige» hacia prosa fluida. Con trozos
cortos tiene menos contexto con el que racionalizar y transcribe mas literal.

Esto convierte el tamano de trozo en un HIPERPARAMETRO del pipeline, no en un
detalle de implementacion: gobierna cuanta senal linguistica llega al modelo y,
en el entrenamiento, que etiquetas produce la cascada heuristica.

Metrica: marcadores de disfluencia observables EN EL TEXTO.
  - repeticiones: palabras adyacentes con la misma forma normalizada
  - muletillas:   tokens del lexico (`FILLER_LEXICON`)
  - terminos de edicion: `EDIT_TERMS`
  - alargamientos: vocal repetida 3+ veces ("procesaaaa")

No mide exactitud de transcripcion. Mide cuanta EVIDENCIA de disfluencia
sobrevive, que es lo que el ramal linguistico puede usar.

Uso:
    python scripts/asr_chunk_study.py --wav grabaciones/sesion.wav
    python scripts/asr_chunk_study.py --wav ... --chunks 5 10 15 30 0
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mrcd.constants import EDIT_TERMS, SAMPLE_RATE  # noqa: E402
from mrcd.extractors.linguistic import Word, lexicon_for  # noqa: E402

ELONGATION = re.compile(r"([aeiou])\1{2,}", re.IGNORECASE)


def read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as w:
        if w.getframerate() != SAMPLE_RATE:
            raise SystemExit(
                f"{path.name} esta a {w.getframerate()} Hz; se esperan {SAMPLE_RATE}"
            )
        pcm = np.frombuffer(w.readframes(w.getnframes()), np.int16)
    return pcm.astype(np.float32) / 32768.0


def transcribe_chunked(
    asr, audio: np.ndarray, chunk_s: float, overlap_s: float = 2.0
) -> list[Word]:
    """Transcribe en trozos de `chunk_s` con solape, quedandose con la region NUEVA.

    `chunk_s <= 0` transcribe el archivo entero de una pasada (el comportamiento
    actual de `extract_and_label.py`).

    ◆ La deduplicacion es por REGION, no por coincidencia de texto ni por rejilla
    de tiempo, y la razon es el objeto de estudio mismo: una repeticion legitima
    del hablante («la la») es indistinguible de un duplicado de solape si se
    compara texto, asi que deduplicar por texto borraria justo el fenomeno que se
    quiere medir. Deduplicar por rejilla de tiempo tampoco sirve: la misma palabra
    recibe timestamps ligeramente distintos en cada trozo y se cuela igual. Con
    rejilla de 50 ms esta medicion daba 29 «repeticiones» a 5 s que en realidad
    eran «Buenos Buenos dias dias», «mi mi nombre», «es es distinguir distinguir».

    La regla: el solape existe solo para dar contexto al ASR al inicio del trozo;
    esas palabras ya las aporto el trozo anterior, asi que se descartan. Cada
    palabra proviene de exactamente un trozo, por construccion.
    """
    if chunk_s <= 0:
        return asr.transcribe(audio)

    step = int((chunk_s - overlap_s) * SAMPLE_RATE)
    size = int(chunk_s * SAMPLE_RATE)
    words: list[Word] = []
    for k, start in enumerate(range(0, len(audio), step)):
        seg = audio[start : start + size]
        if len(seg) < SAMPLE_RATE:  # < 1 s: no aporta
            break
        offset = start / SAMPLE_RATE
        # Region nueva: todo el trozo si es el primero; a partir del solape si no.
        # Se decide por el CENTRO de la palabra, para que una palabra a caballo de
        # la frontera caiga en un solo lado.
        new_from = 0.0 if k == 0 else overlap_s
        for w in asr.transcribe(seg):
            if (w.start + w.end) / 2 < new_from:
                continue
            words.append(
                Word(
                    w.text,
                    w.norm,
                    w.start + offset,
                    w.end + offset,
                    w.prob,
                    w.punct_after,
                )
            )
    words.sort(key=lambda w: w.start)
    return words


def markers(words: list[Word], variety: str = "es-PE") -> dict:
    """Marcadores de disfluencia visibles en el texto."""
    lex = lexicon_for(variety)
    reps = sum(1 for a, b in zip(words, words[1:]) if a.norm and a.norm == b.norm)
    fillers = sum(1 for w in words if w.norm in lex)
    edits = sum(1 for w in words if w.norm in EDIT_TERMS)
    elong = sum(1 for w in words if ELONGATION.search(w.text))
    return {
        "palabras": len(words),
        "repeticiones": reps,
        "muletillas": fillers,
        "edicion": edits,
        "alargamientos": elong,
        "total_marcadores": reps + fillers + edits + elong,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", required=True)
    ap.add_argument("--whisper", default="small")
    ap.add_argument(
        "--chunks",
        type=float,
        nargs="+",
        default=[5, 8, 13, 30, 0],
        help="tamanos de trozo en s; 0 = archivo entero",
    )
    ap.add_argument("--overlap", type=float, default=2.0)
    ap.add_argument("--out", default=str(ROOT / "results/asr_chunk_study.md"))
    a = ap.parse_args()

    from mrcd.extractors.linguistic import VerbatimASR

    audio = read_wav(Path(a.wav))
    dur = len(audio) / SAMPLE_RATE
    asr = VerbatimASR(a.whisper)
    print(f"audio: {dur:.1f}s | whisper: {a.whisper} | solape: {a.overlap}s\n")

    rows = []
    for c in a.chunks:
        words = transcribe_chunked(asr, audio, c, a.overlap)
        m = markers(words)
        label = "entero" if c <= 0 else f"{c:g}s"
        rows.append((label, m, " ".join(w.text for w in words)))
        print(
            f"  trozo {label:>7s}: {m['palabras']:4d} palabras | "
            f"rep={m['repeticiones']:2d} mul={m['muletillas']:2d} "
            f"edi={m['edicion']:2d} alarg={m['alargamientos']:2d} "
            f"| marcadores={m['total_marcadores']:2d}"
        )

    L = [
        "# Disfluencias que sobreviven al ASR, por tamano de trozo",
        "",
        f"- Audio: `{Path(a.wav).name}` ({dur:.1f} s)",
        f"- Whisper: `{a.whisper}` (VerbatimASR) | solape {a.overlap} s",
        "",
        "> Whisper se entrena para producir texto legible. Cuanto mas contexto ve de",
        "> una vez, mas corrige hacia prosa fluida y mas disfluencias borra. El tamano",
        "> de trozo no es un detalle: gobierna cuanta evidencia linguistica llega al",
        "> clasificador, y en entrenamiento, que etiquetas produce la cascada.",
        "",
        "| trozo | palabras | repeticiones | muletillas | edicion | alargamientos | marcadores |",
        "|---|---|---|---|---|---|---|",
    ]
    for label, m, _ in rows:
        L.append(
            f"| {label} | {m['palabras']} | {m['repeticiones']} | {m['muletillas']} "
            f"| {m['edicion']} | {m['alargamientos']} | **{m['total_marcadores']}** |"
        )

    best = max(rows, key=lambda r: r[1]["total_marcadores"])
    whole = next((r for r in rows if r[0] == "entero"), None)
    L += [""]
    if whole and best[0] != "entero":
        gain = best[1]["total_marcadores"] - whole[1]["total_marcadores"]
        L += [
            f"Transcribir en trozos de **{best[0]}** conserva **{gain} marcadores mas** que "
            f"transcribir el archivo entero ({best[1]['total_marcadores']} vs "
            f"{whole[1]['total_marcadores']}).",
            "",
        ]
    L += ["## Transcripciones", ""]
    for label, _, text in rows:
        L += [f"**{label}**", "", f"> {text}", ""]

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    (out.with_suffix(".json")).write_text(
        json.dumps(
            {
                "audio": Path(a.wav).name,
                "whisper": a.whisper,
                "resultados": [{"trozo": l, **m} for l, m, _ in rows],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
