#!/usr/bin/env python3
"""Analyze the frozen audio_text v1 predictions on adjudicated case windows."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.constants import EDIT_TERMS, FILLER_LEXICON  # noqa: E402
from scripts.qa_report import dedupe_judgments, load_jsonl, yt_link  # noqa: E402

LEXICAL = {"filler_word", "repetition", "revision"}
DISFLUENCY = LEXICAL | {"prolongation", "block"}
PAUSES = {"neutral_pause", "rhetorical_pause"}
FLUENT = "fluent"


def md_table(headers: list[str], rows: list[list[object]]) -> list[str]:
    return ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers),
            *["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]]


def binary_confusion(rows: list[dict], positive: set[str]) -> dict[str, int]:
    counts = {"tp": 0, "fn": 0, "fp": 0, "tn": 0}
    for row in rows:
        actual = row["judge"] in positive
        predicted = row["prediction"] in positive
        key = "tp" if actual and predicted else "fn" if actual else "fp" if predicted else "tn"
        counts[key] += 1
    return counts


def safe_ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    parser = argparse.ArgumentParser()
    parser.add_argument("--vid", default="Ka_okSSytes")
    parser.add_argument("--results", default=None)
    parser.add_argument("--out", default=str(ROOT / "results"))
    args = parser.parse_args()

    results_path = Path(args.results) if args.results else ROOT / f"data/results_{args.vid}.json"
    baseline = json.loads(results_path.read_text(encoding="utf-8"))
    judgments, unusable = dedupe_judgments(load_jsonl(
        ROOT / f"data/case/{args.vid}.adjudication.jsonl"))
    windows = baseline["windows"]
    rows = []
    for judgment in judgments:
        index = judgment["window_index"]
        prediction = windows[index]["predictions"]["audio_text"]
        rows.append({
            "index": index,
            "start": windows[index]["timestamp_start"],
            "text": windows[index].get("transcription", ""),
            "judge": judgment["label"],
            "prediction": prediction["class"],
            "probability": prediction["prob"],
        })

    # Strict requested family: lexical/repetition disfluencies versus fluent.
    lexical_rows = [row for row in rows if row["judge"] in LEXICAL | {FLUENT}]
    lexical = binary_confusion(lexical_rows, LEXICAL)

    # Product-level signal: any supported disfluency versus fluent; pause ground
    # truth is excluded, while off-family pause predictions count as negative.
    product_rows = [row for row in rows if row["judge"] in DISFLUENCY | {FLUENT}]
    product = binary_confusion(product_rows, DISFLUENCY)

    pause_rows = [row for row in rows if row["judge"] in PAUSES]
    pause_counts = Counter((row["judge"], row["prediction"]
                            if row["prediction"] in PAUSES else "other")
                           for row in pause_rows)

    revision_as_filler = [row for row in rows
                          if row["judge"] == "filler_word" and row["prediction"] == "revision"]
    revision_predictions = [row for row in rows if row["prediction"] == "revision"]
    overlap = sorted(set(FILLER_LEXICON["es"]) & set(EDIT_TERMS))
    manifest = json.loads((ROOT / "data/dataset_manifest.json").read_text(encoding="utf-8"))
    weak_revision_events = []
    for recording in manifest["recordings"]:
        events = json.loads((ROOT / recording["events_json"]).read_text(encoding="utf-8"))
        weak_revision_events.extend(event for event in events
                                    if event["category"] == "revision" and event.get("source") == "auto")
    weak_terms = Counter(event.get("evidence", {}).get("edit_term", "<fragment>")
                         for event in weak_revision_events)

    result = {
        "video_id": args.vid,
        "source": str(results_path.resolve()),
        "n_adjudicated": len(rows),
        "n_unusable": unusable,
        "lexical_vs_fluent": {"n": len(lexical_rows), **lexical},
        "product_disfluency_vs_fluent": {"n": len(product_rows), **product},
        "pauses": {"n": len(pause_rows), "matrix": {
            actual: {predicted: pause_counts[(actual, predicted)]
                     for predicted in ("neutral_pause", "rhetorical_pause", "other")}
            for actual in ("neutral_pause", "rhetorical_pause")}},
        "revision_vs_filler": {
            "filler_judgments": sum(row["judge"] == "filler_word" for row in rows),
            "filler_predicted_as_revision": len(revision_as_filler),
            "all_revision_predictions": len(revision_predictions),
            "lexicon_overlap": overlap,
            "weak_label_events": len(weak_revision_events),
            "weak_label_edit_terms": dict(weak_terms),
            "cases": revision_as_filler,
        },
    }

    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    (output / "baseline_v1_sensitivity.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    tp, fn, fp, tn = product["tp"], product["fn"], product["fp"], product["tn"]
    ltp, lfn, lfp, ltn = lexical["tp"], lexical["fn"], lexical["fp"], lexical["tn"]
    lines = ["# Sensibilidad funcional de `audio_text` v1", "",
             "> Baseline congelado: se reutilizan las predicciones existentes; no hubo "
             "reentrenamiento ni modificación de probabilidades.", "",
             "## Matriz binaria de producto: disfluencia vs. fluente", "",
             *md_table(["juez \\ modelo", "disfluencia", "fluente/no disfluencia"], [
                 ["disfluencia", tp, fn], ["fluente", fp, tn],
             ]), "",
             f"- Sensibilidad: **{safe_ratio(tp, tp + fn):.1%}** ({tp}/{tp + fn}).",
             f"- Precisión positiva: **{safe_ratio(tp, tp + fp):.1%}** ({tp}/{tp + fp}).",
             f"- Especificidad: **{safe_ratio(tn, tn + fp):.1%}** ({tn}/{tn + fp}).",
             f"- Exactitud: **{safe_ratio(tp + tn, len(product_rows)):.1%}** ({tp + tn}/{len(product_rows)}).", "",
             "La clase positiva agrupa `filler_word`, `repetition`, `revision`, "
             "`prolongation` y `block`. Se excluyen las 16 ventanas cuyo juez indicó pausa.", "",
             "## Familia solicitada: léxica/repetición vs. fluente", "",
             *md_table(["juez \\ modelo", "filler/repetition/revision", "fluente/otra"], [
                 ["filler/repetition/revision", ltp, lfn], ["fluente", lfp, ltn],
             ]), "",
             f"- Sensibilidad: **{safe_ratio(ltp, ltp + lfn):.1%}** ({ltp}/{ltp + lfn}).",
             f"- Exactitud: **{safe_ratio(ltp + ltn, len(lexical_rows)):.1%}** "
             f"({ltp + ltn}/{len(lexical_rows)}).", "",
             "## Pausas", "",
             *md_table(["juez \\ modelo", "neutral_pause", "rhetorical_pause", "fuera de familia"], [
                 [actual, pause_counts[(actual, "neutral_pause")],
                  pause_counts[(actual, "rhetorical_pause")], pause_counts[(actual, "other")]]
                 for actual in ("neutral_pause", "rhetorical_pause")
             ]), "",
             f"- Acierto fino de pausa: **{sum(row['judge'] == row['prediction'] for row in pause_rows)}/{len(pause_rows)} "
             f"({safe_ratio(sum(row['judge'] == row['prediction'] for row in pause_rows), len(pause_rows)):.1%})**.", "",
             "## Diagnóstico `revision` vs. `filler_word`", "",
             f"- `filler_word` juzgados como `revision`: **{len(revision_as_filler)}/"
             f"{sum(row['judge'] == 'filler_word' for row in rows)}**.",
             f"- Todas las predicciones `revision`: **{len(revision_predictions)}**; ninguna "
             "coincidió con una revisión adjudicada.",
             f"- Solapamiento literal de léxicos: `{', '.join(overlap) or 'ninguno'}`.",
             f"- En el corpus, `o sea` originó **{weak_terms['o sea']}/{len(weak_revision_events)}** "
             f"eventos débiles de `revision` ({safe_ratio(weak_terms['o sea'], len(weak_revision_events)):.1%}).",
             "- `o sea` activa simultáneamente `is_filler` e `is_edit_term`. Además, "
             "la heurística etiqueta revisión cuando encuentra un término de edición tras "
             "hasta tres palabras sin puntuación fuerte; no verifica semánticamente que exista reparación.",
             "- `bueno` no pertenece a `EDIT_TERMS`. Sus errores como `revision` son una "
             "correlación aprendida de la supervisión débil/contexto, no una regla directa.", "",
             "- Este baseline no consumía embeddings RoBERTa: su entrada lingüística era "
             "el vector manual de 10 rasgos más `zone` y `dangling` (12 dimensiones).", "",
             "### Casos", ""]
    url = baseline.get("url")
    lines += md_table(["t", "texto", "p(revision)"], [
        [yt_link(url, row["start"]), row["text"] or "—", f"{row['probability']:.4f}"]
        for row in revision_as_filler
    ])
    (output / "baseline_v1_sensitivity.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n".join(lines[:28]))
    print(f"[baseline] reporte -> {output / 'baseline_v1_sensitivity.md'}")


if __name__ == "__main__":
    main()
