#!/usr/bin/env python3
"""Adjudicación ciega de las ventanas donde los modos discrepan.

Convierte «el trimodal corrigió al bimodal» de afirmación en medición.

El juez recibe exactamente lo mismo que en `llm_judge.py`: fotogramas,
transcripción del entorno y medidas acústicas objetivas. **Nunca** recibe qué
predijo cada modo, ni cuál es el modo bajo examen, ni la fuerza de evidencia que
puso la ventana en la cola. Si lo recibiera, su veredicto estaría contaminado por
lo mismo que debe arbitrar.

Es resumible: cada veredicto se guarda en JSONL y no se vuelve a pagar.

Uso:
    python scripts/adjudicate.py --vid Ka_okSSytes --model gpt-4o
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import scripts.llm_judge as J  # noqa: E402

CASE = ROOT / "data/case"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vid", required=True)
    ap.add_argument("--model", default="gpt-4o")
    ap.add_argument("--backend", choices=("api", "ollama"), default="api")
    ap.add_argument("--base-url", default="http://localhost:11434/v1")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()

    qpath = CASE / f"{a.vid}.adjudication_queue.json"
    if not qpath.exists():
        print(f"[adj] falta {qpath}. Corre antes scripts/qa_report.py --vid {a.vid}")
        sys.exit(1)
    queue = json.loads(qpath.read_text(encoding="utf-8"))
    if a.limit:
        queue = queue[:a.limit]

    meta = json.loads((CASE / f"{a.vid}.meta.json").read_text(encoding="utf-8"))
    words = json.loads((ROOT / meta["words_json"]).read_text(encoding="utf-8"))
    z = np.load(ROOT / meta["features_npz"], allow_pickle=False)
    J.validate_raw_features(z, path=str(ROOT / meta["features_npz"]))
    mp4 = ROOT / meta["mp4"]

    output_name = "adjudication_ollama.jsonl" if a.backend == "ollama" else "adjudication.jsonl"
    out = CASE / f"{a.vid}.{output_name}"
    prev = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines() if l.strip()] \
        if out.exists() else []
    # Un registro sin clase utilizable NO cuenta como hecho: una versión anterior
    # leía la clave equivocada del JSON del juez y guardaba label=null, de modo
    # que el veredicto se perdía aunque la llamada hubiese salido bien.
    done = {r["window_index"] for r in prev if r.get("label")}
    broken = {r["window_index"] for r in prev if not r.get("label")} - done
    todo = [c for c in queue if c["window_index"] not in done]
    print(f"[adj] {len(queue)} en cola · {len(done)} ya juzgadas · {len(todo)} pendientes "
          f"· modelo {a.model}")
    if broken:
        print(f"[adj] {len(broken)} registros previos no tienen clase utilizable y se vuelven "
              "a juzgar (los antiguos quedan en el archivo, pero se ignoran por window_index).")
    if not todo:
        print("[adj] nada que hacer.")
        return

    if a.backend == "ollama":
        key, call, provider = "ollama", J.call_ollama, "ollama"
    else:
        provider, key = J.pick_provider(a.model)
        call = J.call_openai if provider == "openai" else J.call_anthropic
    err_log = CASE / f"{a.vid}.adjudication_{a.backend}_errors.jsonl"

    t0, ok, err = time.time(), 0, 0
    kinds: Counter = Counter()
    notes: Counter = Counter()
    with out.open("a", encoding="utf-8") as fh:
        for k, c in enumerate(todo, 1):
            s, e = float(c["start"]), float(c["end"])
            try:
                imgs = [] if a.backend == "ollama" else J.frames_at(mp4, [s, (s + e) / 2, e])
                meas = J.acoustic_summary(z, s, e)
                prev, seg, nxt = J.context_text(words, s, e)
                prompt = J.build_prompt(prev, seg, nxt, meas)
                # blindaje: el prompt no debe contener nada del artefacto de predicción
                low = prompt.lower()
                assert "trimodal" not in low and "audio_text" not in low and "audio_only" not in low, \
                    "fuga: el prompt menciona un modo del modelo"
                if a.backend == "ollama":
                    def local_call(model, text, images, api_key):
                        return J.call_ollama(model, text, images, api_key, a.base_url)
                    v, n_used, note = J.ask_judge(local_call, a.model, prompt, imgs, key, J.CLASSES)
                else:
                    v, n_used, note = J.ask_judge(call, a.model, prompt, imgs, key, J.CLASSES)
                notes[note] += 1
                rec = {"window_index": c["window_index"], "start": s, "end": e,
                       "label": v["classification"],
                       "confidence": v.get("confidence"),
                       "rationale": v.get("rationale"),
                       "model": a.model, "backend": a.backend, "blind": True,
                       "n_frames": n_used, "note": note}
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n"); fh.flush()
                ok += 1
            except Exception as ex:
                err += 1
                kinds[type(ex).__name__] += 1
                J.log_failure(err_log, window_index=c["window_index"], start=s, end=e,
                              kind=type(ex).__name__, reason=str(ex)[:300],
                              raw=getattr(ex, "raw", "")[:500])
                if err <= 3:
                    print(f"  error en ventana {c['window_index']}: {type(ex).__name__}: {str(ex)[:160]}")
            if k % 10 == 0:
                print(f"   {k}/{len(todo)} · {(time.time() - t0) / k:.1f} s/ítem · errores {err}",
                      flush=True)

    print(f"[adj] listo: {ok} nuevas, {err} errores -> {out}")
    if kinds:
        print(f"[adj] errores por tipo: {dict(kinds)}")
        print(f"[adj] respuesta cruda de cada fallo en {err_log.relative_to(ROOT)}")
    if notes:
        print(f"[adj] escalado: {dict(notes)}")
        if notes.get("sin_fotogramas"):
            print(f"[adj] AVISO: {notes['sin_fotogramas']} juicios salieron SIN fotogramas porque "
                  "el modelo rechazó las imágenes. Van marcados con n_frames=0. Arbitrar el aporte "
                  "del vídeo con juicios que no vieron vídeo no vale: el informe los separa.")
    if a.backend == "ollama":
        print_kappa(CASE / f"{a.vid}.adjudication.jsonl", out)
    print(f"[adj] ahora: python scripts/qa_report.py --vid {a.vid}")


def print_kappa(gpt_path: Path, ollama_path: Path) -> None:
    """Print Cohen's kappa over one usable label per shared window."""
    if not gpt_path.exists() or not ollama_path.exists():
        return
    def usable(path):
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        return {r["window_index"]: r["label"] for r in rows if r.get("label")}
    gpt, ollama = usable(gpt_path), usable(ollama_path)
    pairs = [(gpt[i], ollama[i]) for i in sorted(gpt.keys() & ollama.keys())]
    if not pairs:
        print("[adj] kappa GPT-4o vs Ollama: sin ventanas comparables")
        return
    classes = J.CLASSES
    index = {label: i for i, label in enumerate(classes)}
    matrix = [[0] * len(classes) for _ in classes]
    for left, right in pairs:
        if left in index and right in index:
            matrix[index[left]][index[right]] += 1
    n = sum(map(sum, matrix))
    observed = sum(matrix[i][i] for i in range(len(classes))) / n
    row_totals = [sum(row) for row in matrix]
    col_totals = [sum(matrix[row][col] for row in range(len(classes))) for col in range(len(classes))]
    expected = sum(r * c for r, c in zip(row_totals, col_totals)) / (n * n)
    kappa = (observed - expected) / (1 - expected) if expected < 1 else 1.0
    print(f"[adj] kappa GPT-4o vs Ollama: n={n}, acuerdo={observed:.1%}, kappa={kappa:.3f}")


if __name__ == "__main__":
    main()
