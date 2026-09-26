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
from core.dataset import JUDGE_INTERVAL_VERSION, gold_windows, seconds_to_ms, window_starts  # noqa: E402

CLASSES = list(TAXONOMY) + [BACKGROUND]
JUDGE_DIR = ROOT / "data/judge"
HUMAN_IDENTITY = ("recording_id", "start_ms", "end_ms", "provider", "model", "input_fingerprint")
HUMAN_LABEL = "etiqueta_humana(" + "|".join(CLASSES) + ")"


def load_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def select_experiment(rows: list[dict], provider: str | None = None,
                      model: str | None = None) -> tuple[list[dict], tuple[str, str]]:
    """Keep one identified experiment; GOLD later selects the current fingerprint."""
    if (provider is None) != (model is None):
        raise ValueError("pass --provider and --model together for a judge experiment")
    identified = [row for row in rows if row.get("provider") and row.get("model")
                  and row.get("input_fingerprint")
                  and row.get("interval_identity_version") == JUDGE_INTERVAL_VERSION]
    experiments = sorted({(row["provider"], row["model"]) for row in identified})
    if provider is None:
        if len(experiments) != 1:
            raise ValueError(f"select --provider and --model; available versioned experiments: {experiments}. "
                             "Legacy rows without fingerprints or ms-v1 interval identity are not comparable; "
                             "rerun llm_judge.py")
        provider, model = experiments[0]
    elif (provider, model) not in experiments:
        raise ValueError(f"no versioned judgments for {provider}/{model}; available: {experiments}. "
                         "Rerun llm_judge.py for this experiment")
    return [row for row in identified if (row["provider"], row["model"]) == (provider, model)], (provider, model)


def select_current_gold(rows: list[dict], manifest: dict, root: Path,
                        provider: str, model: str) -> list[dict]:
    """Restrict agreement to the exact GOLD population exported by the latest run."""
    cache = {(j["rid"], seconds_to_ms(j["start"]), seconds_to_ms(j["end"]),
              j["input_fingerprint"]): j for j in rows}
    current = []
    for rec in manifest["recordings"]:
        path = root / rec["events_json"].replace(".events.json", ".gold.json")
        if not path.exists():
            raise ValueError(f"missing current GOLD file {path}; rerun llm_judge.py")
        events = json.loads(path.read_text(encoding="utf-8"))
        try:
            gold_windows(events, window_starts(rec["duration_s"]))
        except ValueError as exc:
            raise ValueError(f"{path}: {exc}") from exc
        for event in events:
            if event.get("provider") != provider or event.get("model") != model:
                raise ValueError(f"{path}: GOLD belongs to a different judge experiment; rerun llm_judge.py")
            fingerprint = event.get("input_fingerprint")
            if not fingerprint:
                raise ValueError(f"{path}: GOLD has no input fingerprint; rerun llm_judge.py")
            if event.get("interval_identity_version") != JUDGE_INTERVAL_VERSION:
                raise ValueError(f"{path}: GOLD uses obsolete interval identity; rerun llm_judge.py")
            key = (rec["recording_id"], int(event["start_ms"]), int(event["end_ms"]), fingerprint)
            judgment = cache.get(key)
            if judgment is None or judgment["judge"] != event["category"]:
                raise ValueError(f"{path}: GOLD does not match the selected cache fingerprint; rerun llm_judge.py")
            if event.get("n_frames") is None or judgment.get("n_frames") != event["n_frames"]:
                raise ValueError(f"{path}: GOLD visual provenance does not match the cached judgment")
            current.append(judgment)
    if not current:
        raise ValueError("current GOLD has no successful judgments; rerun llm_judge.py before agreement")
    return current


def matched_judge_pairs(primary: list[dict], secondary: list[dict]) -> list[tuple[str, str]]:
    """Compare only identical intervals and input fingerprints across judges."""
    def key(row):
        return (row["rid"], seconds_to_ms(row["start"]),
                seconds_to_ms(row["end"]), row["input_fingerprint"])

    by_input = {key(row): row for row in secondary}
    return [(row["judge"], by_input[key(row)]["judge"]) for row in primary if key(row) in by_input]


def human_identity(judgment: dict) -> tuple[str, str, str, str, str, str]:
    """Exact experiment, media-input and interval identity for a human annotation."""
    return (str(judgment["rid"]), str(seconds_to_ms(judgment["start"])),
            str(seconds_to_ms(judgment["end"])), str(judgment["provider"]),
            str(judgment["model"]), str(judgment["input_fingerprint"]))


def write_human_template(path: Path, judgments: list[dict], size: int, seed: int) -> bool:
    """Create a lineage-bound template once; never overwrite human annotations."""
    rng = random.Random(seed)
    by = defaultdict(list)
    for judgment in judgments:
        by[judgment["judge"]].append(judgment)
    per = max(1, size // max(len(by), 1))
    selected = [j for group in by.values() for j in rng.sample(group, min(per, len(group)))]
    rng.shuffle(selected)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow([*HUMAN_IDENTITY, "start_s", "end_s", "video", HUMAN_LABEL, "notas"])
            for judgment in selected:
                writer.writerow([*human_identity(judgment), judgment["start"], judgment["end"],
                                 f"data/segments/{judgment['rid']}.mp4", "", ""])
    except FileExistsError:
        return False
    return True


def load_human_pairs(path: Path, judgments: list[dict]) -> list[tuple[str, str]]:
    """Reject legacy or stale labels instead of silently joining approximate intervals."""
    current = {human_identity(j): j for j in judgments}
    pairs = []
    seen = set()
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        if not set(HUMAN_IDENTITY).issubset(reader.fieldnames or []):
            raise ValueError(f"{path}: legacy human CSV lacks exact GOLD identity. Archive/rename it, "
                             "regenerate with --make-template, and migrate labels after checking each video")
        for line, row in enumerate(reader, 2):
            key = tuple(row[field] for field in HUMAN_IDENTITY)
            if key not in current:
                raise ValueError(f"{path}:{line}: human row is not in current GOLD experiment; "
                                 "archive the CSV and regenerate the template before revalidating")
            if key in seen:
                raise ValueError(f"{path}:{line}: duplicate human annotation identity")
            seen.add(key)
            label = (row.get(HUMAN_LABEL) or row.get("etiqueta_humana") or "").strip()
            if label and label not in CLASSES:
                raise ValueError(f"{path}:{line}: invalid human label {label!r}")
            if label:
                pairs.append((label, current[key]["judge"]))
    return pairs


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
    ap.add_argument("--gold-manifest", default=str(ROOT / "data/dataset_manifest.json"),
                    help="manifest for the current GOLD population; required for lineage validation")
    ap.add_argument("--provider", default=None, help="proveedor del experimento principal")
    ap.add_argument("--model", default=None, help="modelo del experimento principal")
    ap.add_argument("--judgments-b", default=None, help="segundo modelo, para fiabilidad entre jueces")
    ap.add_argument("--provider-b", default=None, help="proveedor del segundo experimento")
    ap.add_argument("--model-b", default=None, help="modelo del segundo experimento")
    ap.add_argument("--human", default=str(JUDGE_DIR / "human_check.csv"))
    ap.add_argument("--make-template", type=int, default=0, help="N ítems estratificados para revisión humana")
    ap.add_argument("--seed", type=int, default=RANDOM_SEED)
    a = ap.parse_args()

    jp = Path(a.judgments)
    if not jp.exists() or not (J := load_jsonl(jp)):
        print(f"[agreement] todavía no hay juicios en {jp}. Corre primero scripts/llm_judge.py")
        return
    try:
        J, (provider, model) = select_experiment(J, a.provider, a.model)
        manifest = json.loads(Path(a.gold_manifest).read_text(encoding="utf-8"))
        J = select_current_gold(J, manifest, ROOT, provider, model)
    except (ValueError, OSError, KeyError) as exc:
        ap.error(str(exc))

    if a.make_template:
        out = Path(a.human)
        created = write_human_template(out, J, a.make_template, a.seed)
        if created:
            print(f"[plantilla] creada -> {out}. Revisa el video y completa '{HUMAN_LABEL}'.")
        else:
            print(f"[plantilla] conservada sin cambios -> {out}. Para otra muestra, archiva/renombra "
                  "el CSV existente y vuelve a ejecutar --make-template.")
        return

    heur = [j["heuristic"] for j in J]
    judge = [j["judge"] for j in J]
    n_visual = sum(j.get("n_frames", 0) > 0 for j in J)
    visual_status = "audio_text_only" if n_visual == 0 else "mixed" if n_visual < len(J) else "visual"
    res = dict(n=len(J), provider=provider, model=model,
               visual_evidence={"status": visual_status,
                                "n_with_frames": n_visual, "n_without_frames": len(J) - n_visual,
                                "fallback_rate": round((len(J) - n_visual) / len(J), 4)},
               kappa_heuristic_vs_judge=round(kappa(heur, judge), 3),
               judge_distribution=dict(Counter(judge)), heuristic_distribution=dict(Counter(heur)))
    L = ["# Fiabilidad del juez GOLD", "", f"- Juicios: {len(J)} · proveedor: {provider} · modelo: {model}",
         f"- Evidencia visual usada: {n_visual}/{len(J)} · fallback sin fotogramas: {(len(J)-n_visual)/len(J):.1%}",
         f"- Kappa heurística vs juez: **{res['kappa_heuristic_vs_judge']}** "
         "(bajo/moderado = el juez aporta una opinión independiente; muy alto = solo repite las reglas)", ""]
    if n_visual == 0:
        L += ["> Todos los juicios se resolvieron sin fotogramas: este informe NO valida un juez multimodal.", ""]
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
        try:
            other, (provider_b, model_b) = select_experiment(
                load_jsonl(Path(a.judgments_b)), a.provider_b, a.model_b)
        except ValueError as exc:
            ap.error(f"second judge: {exc}")
        pairs = matched_judge_pairs(J, other)
        if pairs:
            k = kappa([x for x, _ in pairs], [y for _, y in pairs])
            res["kappa_judge_a_vs_b"] = round(k, 3)
            res["judge_b"] = {"provider": provider_b, "model": model_b, "n_pairs": len(pairs)}
            L += [f"## Fiabilidad entre jueces (n={len(pairs)})", "", f"- Kappa juez A vs juez B: **{k:.3f}**", ""]

    hp = Path(a.human)
    if hp.exists():
        try:
            pairs = load_human_pairs(hp, J)
        except ValueError as exc:
            ap.error(str(exc))
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
