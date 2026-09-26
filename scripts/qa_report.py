#!/usr/bin/env python3
"""Informe de control de calidad del estudio de caso.

Sobre el apartado de «casos de éxito del trimodal»
--------------------------------------------------
Afirmar que el modo trimodal CORRIGIÓ al bimodal exige saber cuál de los dos
acertó. En un vídeo nuevo no hay etiquetas de verdad, así que escoger las
discrepancias favorables al trimodal y llamarlas éxitos sería seleccionar a
dedo: con el mismo criterio se podría montar la lista contraria.

Este informe hace dos cosas en su lugar:

1. Mide la **tasa de discrepancia** entre modos, que no necesita verdad.
2. Prepara una **cola de adjudicación ciega**: las ventanas donde los modos
   discrepan se mandan al juez multimodal sin decirle qué predijo cada uno. Una
   vez adjudicadas, el informe cuenta cuántas veces gana cada modo. Eso convierte
   la afirmación en una medición que puede salir en contra.

Uso:
    python scripts/qa_report.py --vid Ka_okSSytes
    # (opcional) adjudicar:  python scripts/adjudicate.py --vid Ka_okSSytes
    # y volver a correr este script para incorporar los resultados
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from math import comb
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.constants import BACKGROUND, SMOKE_HEADER, TAXONOMY  # noqa: E402
from core.models.fusion import MODES  # noqa: E402

CASE = ROOT / "data/case"
PAUSE_CLASSES = {"rhetorical_pause", "neutral_pause"}
CLASSES = list(TAXONOMY) + [BACKGROUND]


def load_jsonl(p: Path) -> list[dict]:
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def dedupe_judgments(rows: list[dict]) -> tuple[list[dict], int]:
    """Un veredicto por ventana: el más reciente que traiga clase utilizable.

    El archivo puede contener varias líneas para la misma ventana cuando se
    vuelve a juzgar. También puede arrastrar registros sin clase, escritos por
    una versión que leía la clave equivocada del JSON del juez. Esos no son
    veredictos: contarlos en el denominador rebajaría todos los porcentajes.
    """
    best: dict[int, dict] = {}
    for r in rows:
        i = r.get("window_index")
        if i is None:
            continue
        if r.get("label") or i not in best:
            best[i] = r
    usable = [r for r in best.values() if r.get("label")]
    return usable, len(best) - len(usable)


def md_table(header: list[str], rows: list[list]) -> list[str]:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return out


def mcnemar(rows: list[dict], W: list[dict], a: str, b: str) -> tuple[int, int, float]:
    """Prueba de McNemar sobre los pares discordantes de dos modos.

    Comparar porcentajes de acierto sin más ignora que ambos modos se evalúan
    sobre las MISMAS ventanas: son medidas pareadas. McNemar mira sólo aquellas
    en que uno acierta y el otro falla, que es donde está la información, y
    contrasta ese reparto contra una moneda. Devuelve (a_gana, b_gana, p).
    """
    ab = ba = 0
    for r in rows:
        g, i = r.get("label"), r.get("window_index")
        if g is None or i is None or i >= len(W):
            continue
        ra = W[i]["predictions"][a]["class"] == g
        rb = W[i]["predictions"][b]["class"] == g
        if ra and not rb:
            ab += 1
        elif rb and not ra:
            ba += 1
    n = ab + ba
    if n == 0:
        return ab, ba, 1.0
    k = min(ab, ba)
    p = 2 * sum(comb(n, x) for x in range(k + 1)) / 2 ** n
    return ab, ba, min(p, 1.0)


def _cohen_kappa_details(rows_a: list[dict], rows_b: list[dict]) -> tuple[int, float, float, list[list[int]]]:
    by_b = {r.get("window_index"): r for r in rows_b
            if r.get("label") in CLASSES and r.get("n_frames", 0) > 0}
    pairs = [(r["label"], by_b[r["window_index"]]["label"]) for r in rows_a
             if r.get("window_index") in by_b and r.get("label") in CLASSES
             and r.get("n_frames", 0) > 0]
    matrix = [[0 for _ in CLASSES] for _ in CLASSES]
    indexes = {label: i for i, label in enumerate(CLASSES)}
    for label_a, label_b in pairs:
        matrix[indexes[label_a]][indexes[label_b]] += 1
    n = len(pairs)
    if not n:
        return 0, 0.0, 0.0, matrix
    observed = sum(matrix[i][i] for i in range(len(CLASSES))) / n
    row_totals = [sum(row) for row in matrix]
    col_totals = [sum(matrix[row][col] for row in range(len(CLASSES))) for col in range(len(CLASSES))]
    expected = sum(r * c for r, c in zip(row_totals, col_totals)) / (n * n)
    kappa = (observed - expected) / (1 - expected) if expected < 1 else 1.0
    return n, observed, kappa, matrix


def cohen_kappa(rows_a: list[dict], rows_b: list[dict]) -> float:
    """Cohen's kappa over paired, usable video judgments."""
    return _cohen_kappa_details(rows_a, rows_b)[2]


def ts(sec: float) -> str:
    return f"{int(sec) // 60:02d}:{int(sec) % 60:02d}"


def yt_link(url: str | None, sec: float) -> str:
    if not url:
        return ts(sec)
    sep = "&" if "?" in url else "?"
    return f"[{ts(sec)}]({url}{sep}t={int(sec)}s)"


def build_queue(W: list[dict], url: str | None, limit: int) -> list[dict]:
    """Ventanas donde trimodal y audio_text discrepan, priorizando las que
    tienen evidencia multimodal fuerte (tensión labial o reinicio tonal)."""
    cand = []
    for i, w in enumerate(W):
        a, t = w["predictions"]["audio_text"], w["predictions"]["trimodal"]
        if a["class"] == t["class"]:
            continue
        f = w["features"]
        strength = abs(f.get("f0_reset", 0.0)) + 4.0 * (f.get("max_delta_mouth_press") or 0.0)
        cand.append({"window_index": i, "start": w["timestamp_start"], "end": w["timestamp_end"],
                     "transcription": w["transcription"], "features": f,
                     "link": yt_link(url, w["timestamp_start"]), "evidence_strength": round(strength, 4),
                     # se guardan aparte: NO deben ir al juez
                     "_blind_hidden": {m: w["predictions"][m] for m in MODES}})
    cand.sort(key=lambda c: -c["evidence_strength"])
    return cand[:limit]


def audio_text_agreements(rows: list[dict], windows: list[dict]) -> list[dict]:
    """Juicios donde audio_text coincide con el árbitro, preservando evidencia.

    Es un desglose descriptivo del benchmark adjudicado; no selecciona por
    confianza ni por la predicción de otro modo.
    """
    matched = []
    for row in rows:
        index, label = row.get("window_index"), row.get("label")
        if index is None or label is None or not (0 <= index < len(windows)):
            continue
        window = windows[index]
        if window["predictions"]["audio_text"]["class"] == label:
            matched.append({"window_index": index, "label": label, "window": window})
    return sorted(matched, key=lambda x: x["window_index"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vid", required=True)
    ap.add_argument("--results", default=None)
    ap.add_argument("--queue-size", type=int, default=60)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    rp = Path(a.results) if a.results else ROOT / f"data/results_{a.vid}.json"
    if not rp.exists():
        print(f"[qa] falta {rp}. Corre antes scripts/infer_video.py --vid {a.vid}")
        sys.exit(1)
    R = json.loads(rp.read_text(encoding="utf-8"))
    W, EV, labels, url = R["windows"], R["events"], R["labels"], R.get("url")
    n = len(W)

    L = [f"# Estudio de caso — {R.get('title') or a.vid}", ""]
    prov = R.get("model_provenance") or {}
    if prov.get("warning"):
        L += [f"> **{prov['warning']}**", ">",
              "> Los checkpoints se entrenaron con etiquetas heurísticas (supervisión débil).",
              "> Las predicciones de abajo sirven para inspeccionar el comportamiento del",
              "> pipeline. No son evidencia de exactitud: eso exige el conjunto de prueba",
              "> con etiquetas independientes y su validación humana.", ""]
    L += [f"- Vídeo: {url or '(sin URL)'}",
          f"- Duración: {R.get('duration_s', 0) / 60:.1f} min · {n} ventanas "
          f"({R['window_s']} s, paso {R['stride_s']} s)",
          f"- Etiquetas de entrenamiento: `{prov.get('label_source', '?')}` "
          f"({prov.get('n_windows', '?')} ventanas)"]
    ft = R.get("face_tracking") or {}
    if ft.get("coverage") is not None:
        L.append(f"- Rostro seguido presente en {ft['coverage']:.0%} de los frames · "
                 f"margen de dominancia {ft.get('dominance_margin', 0):.2f}")
        if (ft.get("dominance_margin") or 0) < 0.25:
            L.append("- **Aviso:** margen de dominancia bajo. El montaje reparte tiempo entre "
                     "dos personas; los rasgos cinésicos de este vídeo pueden estar mezclando "
                     "rostros pese al seguimiento.")
    L.append("")

    # ------------------------------------------------- (a) conteo y distribución
    L += ["## a) Eventos detectados", "",
          "Ventanas contiguas de la misma clase fundidas en un evento. Con paso de "
          f"{R['stride_s']} s y ventana de {R['window_s']} s cada instante cae en unas seis "
          "ventanas, así que contar ventanas multiplicaría los eventos por seis.", ""]
    classes = [c for c in TAXONOMY]
    rows = []
    for c in classes:
        rows.append([c] + [sum(1 for e in EV[m] if e["class"] == c) for m in MODES])
    rows.append(["**total**"] + [f"**{len(EV[m])}**" for m in MODES])
    L += md_table(["clase"] + list(MODES), rows) + [""]

    dur_min = (R.get("duration_s") or 1) / 60
    L += [f"Tasa por minuto (trimodal): **{len(EV['trimodal']) / dur_min:.1f}** eventos/min.", ""]

    # ----------------------------------------- (c) discrepancia y falsos positivos
    L += ["## c) Discrepancia entre modos", ""]
    pair = {}
    for x, y in (("audio_only", "audio_text"), ("audio_text", "trimodal"), ("audio_only", "trimodal")):
        diff = sum(1 for w in W if w["predictions"][x]["class"] != w["predictions"][y]["class"])
        pair[(x, y)] = diff
    L += md_table(["par de modos", "ventanas discrepantes", "% del total"],
                  [[f"{x} vs {y}", v, f"{v / n:.1%}"] for (x, y), v in pair.items()]) + [""]
    L += [f"Ventanas donde los tres modos coinciden: "
          f"**{1 - R['window_disagreement_rate']:.1%}**.", ""]

    # falsos positivos plausibles en pausas naturales, sin necesidad de verdad
    L += ["### Pausas naturales clasificadas como disfluencia", "",
          "Una pausa con silencio largo y reinicio tonal claro es, prosódicamente, una "
          "frontera de discurso. Si el modelo la marca como disfluencia de suspensión "
          "(bloqueo, prolongación), es un falso positivo probable. No es prueba de error "
          "—el rasgo no es infalible— pero sí la cola que conviene auditar.", ""]
    susp = {"block", "prolongation", "repetition", "revision", "filler_word"}
    rows = []
    for m in MODES:
        flags = [w for w in W
                 if w["predictions"][m]["class"] in susp
                 and w["features"]["f0_reset"] > 2.0
                 and (w.get("features_extra", {}).get("sil_dur") or 0) > 0.8]
        rows.append([m, len(flags), f"{len(flags) / n:.1%}"])
    L += md_table(["modo", "ventanas sospechosas", "% del total"], rows) + [""]

    # distribución de f0_reset por clase predicha (descriptivo, sin verdad)
    L += ["### Reinicio tonal por clase predicha (trimodal)", "",
          "Comprobación de coherencia interna: las clases de pausa deberían mostrar un "
          "reinicio tonal mayor que las de suspensión. Si no se separan, el modelo no "
          "está usando el rasgo.", ""]
    byc: dict[str, list[float]] = {}
    for w in W:
        byc.setdefault(w["predictions"]["trimodal"]["class"], []).append(w["features"]["f0_reset"])
    rows = [[c, len(v), f"{np.median(v):+.2f}", f"{np.percentile(v, 25):+.2f} … {np.percentile(v, 75):+.2f}"]
            for c, v in sorted(byc.items(), key=lambda kv: -len(kv[1]))]
    L += md_table(["clase predicha", "ventanas", "f0_reset mediano", "rango intercuartílico"], rows) + [""]

    # ------------------------------------------------ (b) adjudicación, no cherry-picking
    L += ["## b) ¿Corrige el trimodal al bimodal?", ""]
    adj, n_unusable = dedupe_judgments(load_jsonl(CASE / f"{a.vid}.adjudication.jsonl"))
    queue = build_queue(W, url, a.queue_size)
    qpath = CASE / f"{a.vid}.adjudication_queue.json"
    qpath.write_text(json.dumps(queue, indent=1, ensure_ascii=False), encoding="utf-8")

    if not adj:
        L += ["**Sin adjudicar todavía: este apartado no tiene respuesta.**", "",
              "Para decir que un modo corrigió al otro hay que saber cuál acertó, y este "
              "vídeo no tiene etiquetas de verdad. Escoger las discrepancias favorables a "
              "un modo y llamarlas éxitos no sería un resultado, sería una selección.", "",
              f"Se preparó la cola de adjudicación con las **{len(queue)}** ventanas donde "
              "`audio_text` y `trimodal` discrepan, ordenadas por fuerza de evidencia "
              "(reinicio tonal y pico de tensión labial):", "",
              f"    {qpath.relative_to(ROOT)}", "",
              "El juez recibe fotogramas, transcripción y medidas acústicas — nunca lo que "
              "predijo cada modo. Después:", "",
              "```",
              f"python scripts/adjudicate.py --vid {a.vid} --model gpt-4o",
              f"python scripts/qa_report.py --vid {a.vid}",
              "```", ""]
        # muestra sin veredicto: casos a inspeccionar, no "éxitos"
        L += ["### Discrepancias con mayor evidencia multimodal (pendientes de juicio)", "",
              "Se listan para inspección manual. La columna de predicciones muestra AMBAS "
              "sin declarar ganador.", ""]
        rows = []
        for c in queue[:12]:
            h = c["_blind_hidden"]
            rows.append([c["link"],
                         (c["transcription"][:44] + "…") if len(c["transcription"]) > 45 else (c["transcription"] or "—"),
                         f"{c['features']['f0_reset']:+.2f}",
                         (f"{c['features']['max_delta_mouth_press']:.3f}"
                          if c['features']['max_delta_mouth_press'] is not None else "n/d"),
                         h["audio_text"]["class"], h["trimodal"]["class"]])
        L += md_table(["t", "transcripción", "f0_reset", "Δ tensión labial",
                       "audio_text", "trimodal"], rows) + [""]
    else:
        # Un juicio emitido SIN fotogramas no puede arbitrar si el vídeo aporta:
        # el árbitro tampoco vio vídeo. Se separan y sólo los multimodales cuentan.
        seen = [r for r in adj if r.get("n_frames", 3) > 0]
        blind_av = [r for r in adj if r.get("n_frames", 3) == 0]

        def tally(rs):
            by, ties = {m: 0 for m in MODES}, 0
            for r in rs:
                gold, i = r.get("label"), r.get("window_index")
                if gold is None or i is None or i >= n:
                    continue
                winners = [m for m in MODES if W[i]["predictions"][m]["class"] == gold]
                if len(winners) == len(MODES) or not winners:
                    ties += 1
                for m in winners:
                    by[m] += 1
            return by, ties

        by, ties = tally(seen)
        tot = len(seen)
        L += [f"Adjudicadas **{len(adj)}** ventanas discrepantes por el juez a ciegas.", ""]
        if n_unusable:
            L += [f"> Se descartaron **{n_unusable}** registros sin clase utilizable (quedaron así "
                  "por una versión anterior que leía la clave equivocada del JSON del juez). "
                  "No entran en el denominador. Vuelve a correr `adjudicate.py` para rejuzgarlos.", ""]
        if blind_av:
            L += [f"> De ellas, **{len(blind_av)}** se resolvieron SIN fotogramas porque el modelo "
                  "rechazó las imágenes. Quedan **excluidas** del recuento: un árbitro que no vio "
                  "vídeo no puede decidir si el vídeo aporta. Aparecen aparte más abajo.", ""]
        if not tot:
            L += ["**No queda ninguna adjudicación multimodal utilizable.** Todas se resolvieron "
                  "sin fotogramas, así que este apartado sigue sin respuesta. Revisa "
                  f"`data/case/{a.vid}.adjudication_errors.jsonl` para ver qué devolvió el modelo.", ""]
        else:
            L += [f"Recuento sobre las **{tot}** que sí vieron fotogramas:", "",
                  *md_table(["modo", "aciertos", "% de las adjudicadas"],
                            [[m, by[m], f"{by[m] / max(tot, 1):.1%}"] for m in MODES]), "",
                  f"Ventanas donde ningún modo acertó o acertaron todos: {ties}.", ""]
            if tot < 20:
                L += [f"> Con {tot} ventanas, cualquier diferencia entre modos cabe dentro del ruido. "
                      "Sirve para inspeccionar casos, no para concluir.", ""]
        if blind_av:
            byb, _ = tally(blind_av)
            L += ["Sólo como referencia, el recuento de las que no vieron fotogramas "
                  "(mide el aporte del texto, no el del vídeo):", "",
                  *md_table(["modo", "aciertos", "% "],
                            [[m, byb[m], f"{byb[m] / len(blind_av):.1%}"] for m in MODES]), ""]
        adj = seen or adj
        matches = audio_text_agreements(adj, W)
        L += ["### Coincidencias de `audio_text` con el juez", "",
              "Desglose completo de las ventanas adjudicadas en que el modo bimodal "
              "coincidió con el juez. Es una línea base descriptiva, no evidencia de "
              "generalización fuera de esta cola.", ""]
        if not matches:
            L += ["No hay coincidencias utilizables de `audio_text` en la adjudicación actual.", ""]
        else:
            rows = []
            for item in matches:
                w = item["window"]
                pred = w["predictions"]["audio_text"]
                rows.append([
                    yt_link(url, w["timestamp_start"]),
                    item["label"],
                    (w["transcription"][:36] + "…") if len(w["transcription"]) > 37 else (w["transcription"] or "—"),
                    f"{pred['prob']:.3f}",
                    f"{w['features'].get('f0_reset', 0.0):+.2f}",
                    "sí" if w["features"].get("is_dangling") else "no",
                ])
            L += [f"**{len(matches)}** coincidencias de `audio_text`:", "",
                  *md_table(["t", "juez", "transcripción", "p(audio_text)", "f0_reset", "colgante"], rows), ""]
        if tot:
            L += ["**Contraste pareado (McNemar).** Los tres modos se evalúan sobre las mismas "
                  "ventanas, así que comparar porcentajes sueltos no basta. McNemar mira sólo "
                  "los pares en que un modo acierta y el otro falla.", ""]
            pairs = [("trimodal", "audio_text"), ("trimodal", "audio_only"),
                     ("audio_only", "audio_text")]
            rows_mc, sig = [], []
            for x, y in pairs:
                ab, ba, p = mcnemar(seen, W, x, y)
                rows_mc.append([f"{x} vs {y}", ab, ba, f"{p:.3f}",
                                "**sí**" if p < 0.05 else "no"])
                if p < 0.05:
                    sig.append((x, y, ab, ba, p))
            L += md_table([f"comparación", "a favor del 1º", "a favor del 2º", "p", "¿significativa?"],
                          rows_mc) + [""]
            if not sig:
                L += ["Ninguna diferencia entre modos es estadísticamente significativa en esta "
                      "muestra. La diferencia aparente de aciertos cabe dentro del azar.", ""]
            else:
                for x, y, ab, ba, p in sig:
                    L += [f"`{x}` supera a `{y}` de forma significativa ({ab} contra {ba}, "
                          f"p = {p:.3f}).", ""]
            L += ["> **Sesgo de la cola, a tener en cuenta al leer lo anterior.** Estas ventanas "
                  "no son una muestra aleatoria de las discrepancias: se ordenaron por fuerza de "
                  "evidencia (`|f0_reset| + 4 × tensión labial`) y se cortaron por arriba. Es "
                  "decir, están escogidas entre las que más favorecen al modo trimodal. Un "
                  "resultado favorable al trimodal aquí NO se generaliza; uno desfavorable pesa "
                  "más de lo que sugiere su tamaño.", ""]
        # casos concretos, ahora sí con veredicto independiente
        wins = [r for r in adj
                if r.get("window_index") is not None and r["window_index"] < n
                and W[r["window_index"]]["predictions"]["trimodal"]["class"] == r.get("label")
                and W[r["window_index"]]["predictions"]["audio_text"]["class"] != r.get("label")]
        if wins:
            L += ["### Casos donde el trimodal acertó y el bimodal no (veredicto del juez)", ""]
            rows = []
            for r in wins[:10]:
                w = W[r["window_index"]]
                rows.append([yt_link(url, w["timestamp_start"]),
                             (w["transcription"][:40] + "…") if len(w["transcription"]) > 41 else (w["transcription"] or "—"),
                             f"{w['features']['f0_reset']:+.2f}",
                             (f"{w['features']['max_delta_mouth_press']:.3f}"
                              if w['features']['max_delta_mouth_press'] is not None else "n/d"),
                             w["predictions"]["audio_text"]["class"], r.get("label")])
            L += md_table(["t", "transcripción", "f0_reset", "Δ tensión labial",
                           "bimodal dijo", "juez dijo"], rows) + [""]

        judge_sources = [("Gemini", CASE / f"{a.vid}.adjudication_gemini.jsonl"),
                         ("Ollama local", CASE / f"{a.vid}.adjudication_local.jsonl")]
        for judge_name, judge_path in judge_sources:
            if not judge_path.exists():
                continue
            second_judge, _ = dedupe_judgments(load_jsonl(judge_path))
            n_kappa, raw_agreement, kappa, matrix = _cohen_kappa_details(adj, second_judge)
            L += [f"## Acuerdo entre jueces (GPT-4o vs {judge_name})", "",
                  "Se comparan solo ventanas con clasificación válida y `n_frames > 0` en ambos jueces. "
                  "Los juicios sin fotogramas quedan fuera del conjunto comparable.", "",
                  f"- Ventanas comparadas: **{n_kappa}**",
                  f"- Acuerdo bruto: **{raw_agreement:.1%}**",
                  f"- Kappa de Cohen: **{kappa:.3f}**", "",
                  f"Matriz de confusión (filas GPT-4o, columnas {judge_name}):", "",
                  *md_table([f"GPT-4o \\ {judge_name}"] + CLASSES,
                            [[CLASSES[i]] + matrix[i] for i in range(len(CLASSES))]), ""]
            if kappa < 0.4:
                L += ["> **Advertencia crítica:** el acuerdo entre jueces es bajo; este gold set "
                      "no es confiable como ground truth.", ""]

    L += ["## Qué falta para que esto sea citable", "",
          "1. Adjudicar la cola con el juez multimodal (apartado b).",
          "2. Validar al juez contra anotación humana (`data/judge/human_check.csv`). "
          "Sin el kappa juez-humano, el juez es un árbitro sin credenciales.",
          "3. Reentrenar los checkpoints con etiquetas limpias: los actuales salen de "
          "heurísticas cuya tasa de error en `prolongation` ronda el 85 % según el propio juez.", ""]

    out = Path(a.out) if a.out else ROOT / "case_study_interview.md"
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"[qa] informe -> {out}")
    print(f"[qa] cola de adjudicación -> {qpath}  ({len(queue)} ventanas)")
    print(f"[qa] eventos trimodal: {len(EV['trimodal'])} · "
          f"coincidencia entre los tres modos: {1 - R['window_disagreement_rate']:.1%}")


if __name__ == "__main__":
    main()
