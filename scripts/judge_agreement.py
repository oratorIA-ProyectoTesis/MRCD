#!/usr/bin/env python3
"""Fiabilidad del juez multimodal: acuerdo con las heurísticas, entre modelos y con humanos.

- Heurística vs juez: mide ortogonalidad (un acuerdo altísimo indicaría que el juez solo repite las reglas).
- Juez A vs juez B (dos modelos): fiabilidad entre evaluadores automáticos.
- Juez vs humano: el único acuerdo que valida el test ante un jurado. Requiere
  data/judge/human_check.csv (se genera una plantilla con --make-template).

Salida: results/judge_agreement.md / .json
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core.constants import BACKGROUND, RANDOM_SEED, TAXONOMY  # noqa: E402

CLASSES = list(TAXONOMY) + [BACKGROUND]
JUDGE_DIR = ROOT / "data/judge"


def load_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def kappa(a: list[str], b: list[str]) -> float:
    """Kappa de Cohen sobre etiquetas categóricas."""
    if not a:
        return float("nan")
    cats = sorted(set(a) | set(b))
    idx = {c: i for i, c in enumerate(cats)}
    M = np.zeros((len(cats), len(cats)))
    for x, y in zip(a, b):
        M[idx[x], idx[y]] += 1
    n = M.sum()
    po = np.trace(M) / n
    pe = (M.sum(0) * M.sum(1)).sum() / n ** 2
    return float((po - pe) / (1 - pe)) if pe < 1 else float("nan")


def confusion_md(a, b, title_a="heurística", title_b="juez") -> list[str]:
    cats = [c for c in CLASSES if c in set(a) | set(b)]
    M = Counter(zip(a, b))
    L = [f"| {title_a} \\ {title_b} | " + " | ".join(cats) + " |", "|---|" + "---|" * len(cats)]
    for x in cats:
        L.append(f"| {x} | " + " | ".join(str(M.get((x, y), 0)) for y in cats) + " |")
    return L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judgments", default=str(JUDGE_DIR / "judgments.jsonl"))
    ap.add_argument("--judgments-b", default=None, help="segundo modelo, para fiabilidad entre jueces")
    ap.add_argument("--human", default=str(JUDGE_DIR / "human_check.csv"))
    ap.add_argument("--make-template", type=int, default=0, help="N ítems estratificados para revisión humana")
    ap.add_argument("--seed", type=int, default=RANDOM_SEED)
    a = ap.parse_args()

    jp = Path(a.judgments)
    if not jp.exists() or not (J := load_jsonl(jp)):
        print(f"[agreement] todavía no hay juicios en {jp}. Corre primero scripts/llm_judge.py")
        return

    if a.make_template:
        rng = random.Random(a.seed)
        by = defaultdict(list)
        for j in J:
            by[j["judge"]].append(j)
        per = max(1, a.make_template // max(len(by), 1))
        sel = [j for c, lst in by.items() for j in rng.sample(lst, min(per, len(lst)))]
        rng.shuffle(sel)
        out = JUDGE_DIR / "human_check.csv"
        with out.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["recording_id", "start_s", "end_s", "video", "etiqueta_humana(" + "|".join(CLASSES) + ")", "notas"])
            for j in sel:
                w.writerow([j["rid"], round(j["start"], 2), round(j["end"], 2),
                            f"data/segments/{j['rid']}.mp4", "", ""])
        print(f"[plantilla] {len(sel)} ítems -> {out}. Ábrelo, mira cada intervalo en el video y llena la columna "
              "'etiqueta_humana'. Sin esta validación, el test del juez no es defendible.")
        return

    heur = [j["heuristic"] for j in J]
    judge = [j["judge"] for j in J]
    res = dict(n=len(J), model=J[0].get("model"), kappa_heuristic_vs_judge=round(kappa(heur, judge), 3),
               judge_distribution=dict(Counter(judge)), heuristic_distribution=dict(Counter(heur)))
    L = ["# Fiabilidad del juez multimodal", "", f"- Juicios: {len(J)} · modelo: {J[0].get('model')}",
         f"- Kappa heurística vs juez: **{res['kappa_heuristic_vs_judge']}** "
         "(bajo/moderado = el juez aporta una opinión independiente; muy alto = solo repite las reglas)", ""]
    L += ["## Matriz heurística vs juez", ""] + confusion_md(heur, judge) + [""]

    # ¿qué hace el juez con las ventanas que las heurísticas no marcaron?
    neg = [j for j in J if j["origin"] == "negative"]
    if neg:
        found = Counter(j["judge"] for j in neg if j["judge"] != BACKGROUND)
        res["negatives"] = dict(n=len(neg), events_found=dict(found))
        L += [f"## Muestreo negativo ({len(neg)} ventanas sin candidato heurístico)", "",
              f"- El juez encontró fenómenos en {sum(found.values())} de ellas: {dict(found)}",
              "- Esto estima lo que las heurísticas NO ven (su recall).", ""]

    if a.judgments_b and Path(a.judgments_b).exists():
        B = {(j["rid"], round(j["start"], 2)): j for j in load_jsonl(Path(a.judgments_b))}
        pairs = [(j["judge"], B[(j["rid"], round(j["start"], 2))]["judge"]) for j in J
                 if (j["rid"], round(j["start"], 2)) in B]
        if pairs:
            k = kappa([x for x, _ in pairs], [y for _, y in pairs])
            res["kappa_judge_a_vs_b"] = round(k, 3)
            L += [f"## Fiabilidad entre jueces (n={len(pairs)})", "", f"- Kappa juez A vs juez B: **{k:.3f}**", ""]

    hp = Path(a.human)
    if hp.exists():
        H = {}
        for r in csv.DictReader(hp.open(encoding="utf-8")):
            lab = (r.get("etiqueta_humana(" + "|".join(CLASSES) + ")") or r.get("etiqueta_humana") or "").strip()
            if lab in CLASSES:
                H[(r["recording_id"], round(float(r["start_s"]), 2))] = lab
        pairs = [(H[(j["rid"], round(j["start"], 2))], j["judge"]) for j in J
                 if (j["rid"], round(j["start"], 2)) in H]
        if pairs:
            k = kappa([x for x, _ in pairs], [y for _, y in pairs])
            acc = float(np.mean([x == y for x, y in pairs]))
            res["human_validation"] = dict(n=len(pairs), kappa=round(k, 3), accuracy=round(acc, 3))
            L += [f"## Validación humana (n={len(pairs)})", "",
                  f"- Kappa juez vs humano: **{k:.3f}** · coincidencia exacta: {acc:.1%}",
                  "- Landis & Koch (1977): 0.61–0.80 sustancial; 0.41–0.60 moderado.", ""]
            L += ["### Matriz humano vs juez", ""] + confusion_md([x for x, _ in pairs], [y for _, y in pairs],
                                                                  "humano", "juez") + [""]
        else:
            L += ["## Validación humana", "", "- El CSV existe pero no tiene etiquetas válidas todavía.", ""]
    else:
        L += ["## Validación humana", "",
              "- **PENDIENTE**: sin ella no se puede afirmar que las etiquetas del juez aproximen la verdad.",
              "- Genera la plantilla: `python scripts/judge_agreement.py --make-template 60`", ""]

    out = ROOT / "results"; out.mkdir(exist_ok=True)
    (out / "judge_agreement.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    (out / "judge_agreement.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n".join(L[:12]))


if __name__ == "__main__":
    main()
