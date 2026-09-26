#!/usr/bin/env python3
"""Blind local visual adjudication for the case-study disagreement queue.

Unlike the general adjudicator, this path is deliberately Ollama-only and refuses
to run unless ``ollama show <model>`` declares the ``vision`` capability.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import scripts.llm_judge as J  # noqa: E402

CASE = ROOT / "data/case"
MIN_VISUAL_COVERAGE = 0.90


def ollama_executable() -> str:
    """Resolve Ollama even when a newly installed Windows app is not yet in PATH."""
    return (shutil.which("ollama")
            or str(Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Ollama" / "ollama.exe"))


def require_vision_model(model: str) -> None:
    """Abort before adjudication when Ollama does not advertise visual input."""
    try:
        result = subprocess.run([ollama_executable(), "show", model], capture_output=True,
                                text=True, encoding="utf-8", errors="replace", check=False)
    except FileNotFoundError as exc:
        raise RuntimeError("Ollama no está instalado o no está disponible en PATH. "
                           "Instálalo e inicia el servicio antes de ejecutar el segundo juez local.") from exc
    if result.returncode:
        detail = (result.stderr or result.stdout).strip()
        raise RuntimeError(f"No se pudo inspeccionar el modelo local {model!r}: {detail}")
    capabilities = set()
    in_capabilities = False
    for line in result.stdout.splitlines():
        fields = line.strip().split()
        if not fields:
            in_capabilities = False
        elif fields[0].lower() == "capabilities":
            capabilities.update(field.lower() for field in fields[1:])
            in_capabilities = True
        elif in_capabilities and line[:1].isspace():
            capabilities.update(field.lower() for field in fields)
        else:
            in_capabilities = False
    if "vision" not in capabilities:
        shown = ", ".join(sorted(capabilities)) or "ninguna"
        raise RuntimeError(f"El modelo {model!r} no declara la capability 'vision' "
                           f"en `ollama show` (capabilities detectadas: {shown}). "
                           "La adjudicación se cancela para no producir juicios sin evidencia visual.")


def visual_coverage(rows: list[dict], expected: int | None = None) -> float:
    denominator = expected if expected is not None else len(rows)
    return sum(row.get("n_frames", 0) > 0 for row in rows) / max(denominator, 1)


def require_visual_coverage(rows: list[dict], expected: int) -> None:
    coverage = visual_coverage(rows, expected)
    seen = sum(row.get("n_frames", 0) > 0 for row in rows)
    print(f"[local-judge] cobertura visual: {seen}/{expected} ({coverage:.1%}) con n_frames > 0")
    if coverage < MIN_VISUAL_COVERAGE:
        raise RuntimeError(f"cobertura visual insuficiente: {seen}/{expected} ({coverage:.1%}); "
                           f"se exige al menos {MIN_VISUAL_COVERAGE:.0%} antes de calcular kappa.")


def load_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    latest: dict[int, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if row.get("label"):
                latest[row["window_index"]] = row
    return list(latest.values())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vid", required=True)
    parser.add_argument("--model", default="qwen2.5vl:7b")
    parser.add_argument("--base-url", default="http://localhost:11434/v1")
    args = parser.parse_args()

    require_vision_model(args.model)
    qpath = CASE / f"{args.vid}.adjudication_queue.json"
    if not qpath.exists():
        raise SystemExit(f"[local-judge] falta {qpath}; corre primero scripts/qa_report.py --vid {args.vid}")
    queue = json.loads(qpath.read_text(encoding="utf-8"))
    meta = json.loads((CASE / f"{args.vid}.meta.json").read_text(encoding="utf-8"))
    words = json.loads((ROOT / meta["words_json"]).read_text(encoding="utf-8"))
    z = np.load(ROOT / meta["features_npz"], allow_pickle=False)
    J.validate_raw_features(z, path=str(ROOT / meta["features_npz"]))
    mp4 = ROOT / meta["mp4"]
    out = CASE / f"{args.vid}.adjudication_local.jsonl"
    done = {row["window_index"] for row in load_rows(out)}
    todo = [item for item in queue if item["window_index"] not in done]
    print(f"[local-judge] {len(queue)} en cola · {len(done)} ya juzgadas · {len(todo)} pendientes · modelo {args.model}")

    errors = CASE / f"{args.vid}.adjudication_local_errors.jsonl"
    failures: Counter = Counter()
    started = time.time()
    with out.open("a", encoding="utf-8") as handle:
        for count, item in enumerate(todo, 1):
            start, end = float(item["start"]), float(item["end"])
            try:
                middle = (start + end) / 2
                images = J.frames_at(mp4, [start, middle, max(end - 0.05, middle)])
                if not images:
                    raise RuntimeError("No se pudieron extraer fotogramas de la ventana")
                previous, segment, following = J.context_text(words, start, end)
                prompt = J.build_prompt(previous, segment, following, J.acoustic_summary(z, start, end))
                verdict, n_frames, note = J.ask_judge(
                    lambda model, text, imgs, key: J.call_ollama(model, text, imgs, key, args.base_url),
                    args.model, prompt, images, "ollama", J.CLASSES)
                handle.write(json.dumps({"window_index": item["window_index"], "start": start, "end": end,
                                         "label": verdict["classification"],
                                         "confidence": verdict.get("confidence"),
                                         "rationale": verdict.get("rationale"), "model": args.model,
                                         "backend": "ollama", "blind": True, "n_frames": n_frames,
                                         "note": note}, ensure_ascii=False) + "\n")
                handle.flush()
            except Exception as exc:
                failures[type(exc).__name__] += 1
                J.log_failure(errors, window_index=item["window_index"], start=start, end=end,
                              kind=type(exc).__name__, reason=str(exc)[:300], raw=getattr(exc, "raw", "")[:500])
                print(f"[local-judge] error en ventana {item['window_index']}: {type(exc).__name__}: {str(exc)[:160]}")
            if count % 10 == 0:
                print(f"[local-judge] {count}/{len(todo)} · {(time.time() - started) / count:.1f} s/ítem", flush=True)

    rows = load_rows(out)
    print(f"[local-judge] listo: {len(rows)}/{len(queue)} respuestas utilizables -> {out}")
    if failures:
        print(f"[local-judge] errores por tipo: {dict(failures)}")
    require_visual_coverage(rows, expected=len(queue))
    print(f"[local-judge] adjudicación válida para comparar: python scripts/qa_report.py --vid {args.vid}")


if __name__ == "__main__":
    main()
