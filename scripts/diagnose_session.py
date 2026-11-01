#!/usr/bin/env python3
"""Diagnostico de una sesion grabada: separa fallos del ASR de fallos del clasificador.

Cuando una disfluencia no se detecta, la UI en vivo no dice en que capa se perdio.
Hay dos causas con arreglos distintos:

  1. El ASR la borro (Whisper normaliza el habla: «la la la» -> «la»), asi que el
     ramal linguistico nunca la vio.
  2. El ASR la transcribio bien y el clasificador aun asi fallo.

Este script separa ambas sobre el WAV de la sesion:

  --transcribe   reprocesa el audio con otro tamano de Whisper y compara
                 que disfluencias sobreviven en cada uno.
  --counterfactual  TEST DE MEMORIZACION LEXICA. Reevalua cada ventana con la
                 acustica intacta y el texto neutralizado. Si la clase predicha
                 no cambia, la decision venia de la senal acustica. Si colapsa,
                 el modelo estaba leyendo palabras, no escuchando.

El contrafactual importa por una razon concreta: la cascada heuristica que genero
las etiquetas de entrenamiento usa el lexico para decidir (`FILLER_LEXICON`,
`EDIT_TERMS`). Un modelo entrenado sobre esas etiquetas puede aprender a reconocer
las PALABRAS que la regla buscaba en vez del fenomeno acustico, y en ese caso su
confianza alta no es evidencia de nada.

Uso:
    python scripts/diagnose_session.py --wav grabaciones/sesion.wav \
        --session grabaciones/session.json --counterfactual
    python scripts/diagnose_session.py --wav ... --transcribe large-v3
"""

from __future__ import annotations

try:  # must run before numpy / torch / ctranslate2 / mrcd are imported (see scripts/_pin_omp.py)
    import _pin_omp  # noqa: F401  (script run as a file)
except ImportError:  # imported as scripts.<module>
    from scripts import _pin_omp  # noqa: F401

import argparse
import json
import sys
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from mrcd.constants import CONTEXT_S, SAMPLE_RATE, WINDOW_S  # noqa: E402
from mrcd.extractors.linguistic import Word  # noqa: E402
from mrcd.live.window_builder import build_window, window_text  # noqa: E402
from scripts.infer_video import load_ckpt, predict  # noqa: E402

# Texto de relleno para el contrafactual: palabras de contenido, longitud media,
# sin muletillas, sin terminos de edicion y sin repeticion. Si la clase sobrevive
# a esto, no dependia del lexico.
NEUTRAL_TOKENS = [
    "camino",
    "ventana",
    "libro",
    "puerta",
    "tiempo",
    "ciudad",
    "planta",
    "mesa",
    "nube",
    "piedra",
    "campo",
    "puente",
]


def read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as w:
        if w.getframerate() != SAMPLE_RATE:
            raise SystemExit(
                f"{path.name} esta a {w.getframerate()} Hz; se esperan {SAMPLE_RATE}"
            )
        pcm = np.frombuffer(w.readframes(w.getnframes()), np.int16)
    return pcm.astype(np.float32) / 32768.0


def transcribe(audio: np.ndarray, model_size: str) -> list[Word]:
    """Transcribe el audio completo reutilizando el ASR del propio pipeline.

    Se usa `VerbatimASR` y no una llamada nueva a faster-whisper para que el
    diagnostico mida EL MISMO ASR que corre en produccion: ya trae
    `condition_on_previous_text=False`, `suppress_tokens=[]` y `vad_filter=False`,
    que son justamente los ajustes que evitan que Whisper limpie las disfluencias.
    Reimplementarlo aqui arriesgaria diagnosticar una configuracion distinta de la
    que realmente se usa.
    """
    from mrcd.extractors.linguistic import VerbatimASR

    return VerbatimASR(model_size).transcribe(audio)


def neutralize(words: list[Word]) -> list[Word]:
    """Sustituye cada palabra por una neutra, CONSERVANDO sus tiempos.

    Los tiempos se conservan a proposito: los rasgos de duracion, huecos y
    alargamiento se calculan a partir de ellos, asi que la unica variable que
    cambia es la IDENTIDAD LEXICA. Cualquier cambio en la prediccion es atribuible
    al lexico y a nada mas.
    """
    out = []
    for i, w in enumerate(words):
        tok = NEUTRAL_TOKENS[i % len(NEUTRAL_TOKENS)]
        # `punct_after` se conserva: la puntuacion marca frontera de clausula y es
        # un rasgo prosodico-sintactico, no identidad lexica. Neutralizarla tambien
        # cambiaria dos variables a la vez y el test dejaria de ser limpio.
        out.append(Word(tok, tok, w.start, w.end, w.prob, w.punct_after))
    return out


def evaluate(
    audio: np.ndarray,
    words: list[Word],
    starts: list[float],
    net,
    norm,
    labels,
    dev: str,
    ling_dim: int,
    include_pros: bool,
    context: bool,
) -> list[dict]:
    rows = []
    for t0 in starts:
        tensors = build_window(
            audio,
            words,
            t0,
            word_offset_s=0.0,
            context=context,
            include_pros=include_pros,
            ling_dim=ling_dim,
        )
        p = predict(net, norm, tensors, dev)[0]
        idx = int(p.argmax())
        rows.append(
            {
                "t": t0,
                "clase": labels[idx],
                "conf": float(p[idx]),
                "probs": {l: float(v) for l, v in zip(labels, p)},
                "texto": window_text(words, t0),
            }
        )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", required=True)
    ap.add_argument("--session", default=None, help="session.json de la misma sesion")
    ap.add_argument("--ckpt", default=str(ROOT / "models/ckpt_audio_text.pt"))
    ap.add_argument("--whisper", default="small")
    ap.add_argument(
        "--transcribe",
        default=None,
        help="segundo tamano de Whisper con el que comparar (p. ej. large-v3)",
    )
    ap.add_argument(
        "--counterfactual",
        action="store_true",
        help="reevalua con la acustica intacta y el texto neutralizado",
    )
    ap.add_argument("--out", default=str(ROOT / "results/session_diagnosis.md"))
    a = ap.parse_args()

    audio = read_wav(Path(a.wav))
    dur = len(audio) / SAMPLE_RATE
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net, norm, labels, prov = load_ckpt(Path(a.ckpt), dev)
    ling_dim = net.ling.proj[0].in_features
    include_pros = net.pros is not None
    context = ling_dim > 10

    print(f"audio: {dur:.1f}s | checkpoint: {Path(a.ckpt).name} | modo: {net.mode}")
    print(f"transcribiendo con whisper '{a.whisper}'...")
    words = transcribe(audio, a.whisper)
    print(f"  {len(words)} palabras")

    # Mismos inicios de ventana que uso la sesion en vivo, si esta disponible.
    if a.session:
        sess = json.loads(Path(a.session).read_text(encoding="utf-8"))
        starts = [r["timestamp"] for r in sess["ventanas"]]
        live = {round(r["timestamp"], 1): r for r in sess["ventanas"]}
    else:
        starts = list(np.arange(CONTEXT_S, dur - WINDOW_S, 2.5))
        live = {}

    base = evaluate(
        audio, words, starts, net, norm, labels, dev, ling_dim, include_pros, context
    )

    L = [
        "# Diagnostico de sesion",
        "",
        f"- Audio: `{Path(a.wav).name}` ({dur:.1f} s)",
        f"- Checkpoint: `{Path(a.ckpt).name}` (modo {net.mode})",
        f"- Whisper: `{a.whisper}` (VerbatimASR del pipeline)",
        "",
    ]
    if prov.get("warning"):
        L += [f"> **{prov['warning']}**", ""]

    # ---------------------------------------------------- comparacion de ASR
    if a.transcribe:
        print(f"transcribiendo con whisper '{a.transcribe}'...")
        words_b = transcribe(audio, a.transcribe)
        alt = evaluate(
            audio,
            words_b,
            starts,
            net,
            norm,
            labels,
            dev,
            ling_dim,
            include_pros,
            context,
        )
        L += [
            f"## ASR: `{a.whisper}` vs `{a.transcribe}`",
            "",
            "Misma acustica, misma arquitectura: lo unico que cambia es el texto.",
            "Un cambio de clase aqui es un fallo atribuible al ASR, no al clasificador.",
            "",
            "| t | texto "
            + a.whisper
            + " | texto "
            + a.transcribe
            + " | clase A | clase B |",
            "|---|---|---|---|---|",
        ]
        flips = 0
        for r, s in zip(base, alt):
            flip = r["clase"] != s["clase"]
            flips += flip
            mark = " **<-**" if flip else ""
            L.append(
                f"| {r['t']:.1f} | {r['texto'][:40] or '—'} | {s['texto'][:40] or '—'} "
                f"| {r['clase']} | {s['clase']}{mark} |"
            )
        L += [
            "",
            f"Ventanas en que la clase cambia solo por el ASR: **{flips}/{len(base)}**",
            "",
        ]
        print(f"  clase cambia por ASR en {flips}/{len(base)} ventanas")

    # ------------------------------------------- contrafactual lexico
    if a.counterfactual:
        print("contrafactual: neutralizando lexico, conservando tiempos...")
        cf = evaluate(
            audio,
            neutralize(words),
            starts,
            net,
            norm,
            labels,
            dev,
            ling_dim,
            include_pros,
            context,
        )
        L += [
            "## Test de memorizacion lexica",
            "",
            "Cada ventana reevaluada con la **acustica intacta** y el texto sustituido por",
            "palabras neutras que conservan los tiempos originales. La unica variable que",
            "cambia es la identidad lexica.",
            "",
            "- Si la clase **se mantiene**, la decision venia de la senal acustica/temporal.",
            "- Si **colapsa**, el modelo estaba reconociendo palabras, no escuchando.",
            "",
            "Importa porque la cascada heuristica que genero las etiquetas de entrenamiento",
            "decide por lexico (`FILLER_LEXICON`, `EDIT_TERMS`): el modelo pudo aprender la",
            "regla en vez del fenomeno.",
            "",
            "| t | texto original | clase real | conf | clase sin lexico | conf | sobrevive |",
            "|---|---|---|---|---|---|---|",
        ]
        survived = 0
        per_class: dict[str, list[int]] = {}
        for r, s in zip(base, cf):
            ok = r["clase"] == s["clase"]
            survived += ok
            per_class.setdefault(r["clase"], []).append(int(ok))
            L.append(
                f"| {r['t']:.1f} | {r['texto'][:38] or '—'} | {r['clase']} | "
                f"{r['conf'] * 100:.0f}% | {s['clase']} | {s['conf'] * 100:.0f}% | "
                f"{'si' if ok else '**NO**'} |"
            )
        L += [
            "",
            f"Clases que sobreviven sin lexico: **{survived}/{len(base)}** "
            f"({survived / max(len(base), 1) * 100:.0f}%)",
            "",
            "### Dependencia lexica por clase",
            "",
            "| clase | ventanas | sobreviven | lectura |",
            "|---|---|---|---|",
        ]
        for cls, oks in sorted(per_class.items(), key=lambda kv: -len(kv[1])):
            rate = sum(oks) / len(oks)
            read = (
                "acustica" if rate >= 0.8 else "mixta" if rate >= 0.4 else "**lexica**"
            )
            L.append(
                f"| {cls} | {len(oks)} | {sum(oks)}/{len(oks)} ({rate * 100:.0f}%) | {read} |"
            )
        L += [
            "",
            "Una clase con dependencia **lexica** no esta midiendo el fenomeno acustico",
            "que su definicion describe. Es el patron esperado si el modelo reaprendio la",
            "cascada heuristica en vez de generalizar.",
            "",
        ]
        print(f"  sobreviven sin lexico: {survived}/{len(base)}")

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
