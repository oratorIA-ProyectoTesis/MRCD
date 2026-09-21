#!/usr/bin/env python3
"""Benchmark de ablación: audio_only vs audio_text vs trimodal.

- Validación cruzada agrupada por hablante (GroupKFold): ningún orador aparece en train y test.
- F1 por clase y macro-F1 con IC 95 % por bootstrap de CLÚSTER (hablante); si hay < 5
  hablantes, bootstrap por BLOQUES temporales de 6 s (ventanas solapadas NO son independientes).
- Diferencias pareadas (trimodal − audio_only, audio_text − audio_only) con su IC 95 %.

Por defecto se niega a correr con etiquetas 'auto' (circularidad). Usa --allow-auto-labels
solo para verificar que el pipeline corre de punta a punta.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402

from core.constants import SMOKE_HEADER  # noqa: E402
from core.models.fusion import MODES, CrossModalFusion  # noqa: E402

KEYS = ("ac", "pros", "ling", "lpos", "lmask", "kin", "kmask")
NUM_KEYS = ("ac", "pros", "ling", "kin")   # tensores continuos a estandarizar


def to_batch(d, idx, dev):
    b = {k: torch.as_tensor(d[k][idx]).to(dev) for k in KEYS if k in d}
    for k in NUM_KEYS:
        if k in b:
            b[k] = b[k].float()
    b["lmask"], b["kmask"] = b["lmask"].bool(), b["kmask"].bool()
    return b


def standardize(d, tr):
    """z-score con estadísticas SOLO del conjunto de entrenamiento."""
    out = dict(d)
    for k in NUM_KEYS:
        if k not in d:
            continue
        x = d[k][tr].reshape(-1, d[k].shape[-1])
        mu, sd = x.mean(0), x.std(0) + 1e-6
        out[k] = ((d[k] - mu) / sd).astype(np.float32)
    return out


def train_eval(d, tr, te, mode, n_cls, args, seed, test_d=None):
    """Entrena sobre `d[tr]`; evalúa sobre `test_d[te]` si se da, o `d[te]`."""
    torch.manual_seed(seed); np.random.seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ds = standardize(d, tr)
    if test_d is not None:
        mu_sd = {k: (d[k][tr].reshape(-1, d[k].shape[-1]).mean(0), d[k][tr].reshape(-1, d[k].shape[-1]).std(0) + 1e-6)
                 for k in NUM_KEYS if k in d and k in test_d}
        ts = dict(test_d)
        for k, (mu, sd) in mu_sd.items():
            ts[k] = ((test_d[k] - mu) / sd).astype(np.float32)
    else:
        ts = ds
    net = CrossModalFusion(n_cls, ac_dim=ds["ac"].shape[-1], ling_dim=ds["ling"].shape[-1], d=args.d,
                           mode=mode, modality_dropout=args.modality_dropout,
                           pros_dim=ds["pros"].shape[-1] if "pros" in ds else 0).to(dev)
    counts = np.bincount(d["y"][tr], minlength=n_cls).astype(float)
    w = torch.tensor(np.where(counts > 0, counts.sum() / (n_cls * np.maximum(counts, 1)), 0.0), dtype=torch.float32, device=dev)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-3)
    y = torch.as_tensor(d["y"]).to(dev)
    for ep in range(args.epochs):
        net.train()
        perm = np.random.permutation(tr)
        for s in range(0, len(perm), args.bs):
            idx = perm[s:s + args.bs]
            loss = F.cross_entropy(net(to_batch(ds, idx, dev)), y[idx], weight=w)
            opt.zero_grad(); loss.backward(); opt.step()
    net.eval(); preds = []
    with torch.no_grad():
        for s in range(0, len(te), 512):
            preds.append(net(to_batch(ts, te[s:s + 512], dev)).argmax(1).cpu().numpy())
    return np.concatenate(preds) if preds else np.zeros(0, np.int64)


def f1s(y, p, n_cls):
    per = f1_score(y, p, labels=range(n_cls), average=None, zero_division=0)
    present = [c for c in range(1, n_cls) if (y == c).any()]            # macro sobre clases de disfluencia presentes
    macro = float(np.mean(per[present])) if present else float("nan")
    return per, macro


def bootstrap(y, preds, units, n_cls, B, rng):
    uniq = np.unique(units)
    groups = {u: np.where(units == u)[0] for u in uniq}
    stats = {m: {"per": [], "macro": []} for m in preds}
    diffs = {k: [] for k in ("trimodal-audio_only", "audio_text-audio_only", "trimodal-audio_text")}
    for _ in range(B):
        idx = np.concatenate([groups[u] for u in rng.choice(uniq, len(uniq), replace=True)])
        mac = {}
        for m, p in preds.items():
            per, mac[m] = f1s(y[idx], p[idx], n_cls)
            stats[m]["per"].append(per); stats[m]["macro"].append(mac[m])
        for k in diffs:
            a, b = k.split("-")
            if a in mac and b in mac:
                diffs[k].append(mac[a] - mac[b])
    ci = lambda v: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]
    out = {m: {"macro_ci": ci(s["macro"]), "per_ci": [ci(np.asarray(s["per"])[:, c]) for c in range(n_cls)]} for m, s in stats.items()}
    return out, {k: ci(v) for k, v in diffs.items() if v}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data/windows_gold.npz"))
    ap.add_argument("--test-data", default=None,
                    help="npz de TEST con etiquetas independientes (p. ej. data/windows_gold.npz). "
                         "Con esto, --data se usa solo para ENTRENAR (supervisión débil).")
    ap.add_argument("--allow-auto-labels", action="store_true")
    ap.add_argument("--allow-same-label-source", action="store_true",
                    help="permite entrenar y evaluar con el mismo origen de etiquetas "
                         "(techo supervisado). Por defecto se rechaza por circular.")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--modality-dropout", type=float, default=0.5)
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--out", default=str(ROOT / "results"))
    args = ap.parse_args()

    d = dict(np.load(args.data, allow_pickle=False))
    src = str(d.get("label_source", "unknown"))
    test = dict(np.load(args.test_data, allow_pickle=False)) if args.test_data else None
    test_src = str(test.get("label_source", "unknown")) if test else src

    # --- guardia contra la circularidad, antes de gastar horas de cómputo.
    # El diseño de supervisión débil exige que el origen de las etiquetas de
    # ENTRENAMIENTO sea distinto del de las de TEST. Si coinciden, el resultado
    # mide cuánto se ajusta el modelo a un etiquetador, no cuánto acierta, y
    # tiene muy buena pinta justo por eso.
    if args.test_data:
        if Path(args.test_data).resolve() == Path(args.data).resolve():
            sys.exit(f"--data y --test-data apuntan al MISMO archivo ({args.data}).\n"
                     "Entrenarías y evaluarías sobre las mismas etiquetas. Para supervisión\n"
                     "débil: --data data/windows_auto.npz --test-data data/windows_gold.npz")
        if src == test_src and not args.allow_same_label_source:
            sys.exit(f"Entrenamiento y test comparten origen de etiquetas ('{src}').\n"
                     "Eso es circular: el test hereda los sesgos del mismo etiquetador que\n"
                     "produjo el entrenamiento, y el macro-F1 sale inflado.\n"
                     "Para supervisión débil: --data data/windows_auto.npz "
                     "--test-data data/windows_gold.npz\n"
                     "Si de verdad quieres el techo supervisado (mismo origen, hablantes\n"
                     "separados), pásalo explícito con --allow-same-label-source.")
    if test_src not in ("human", "gold_llm") and not args.allow_auto_labels:
        sys.exit(f"Etiquetas de test '{test_src}': evaluar sobre las mismas heurísticas que etiquetan es circular. "
                 "Usa --test-data data/windows_gold.npz (juez multimodal) o etiquetas humanas; "
                 "--allow-auto-labels solo para prueba de humo.")
    labels = [str(x) for x in d["labels"]]; n_cls = len(labels)
    y_train_all, groups = d["y"], d["group"]
    y = test["y"] if test else d["y"]
    tgroups = test["group"] if test else groups
    uniq = np.unique(groups)
    if test is not None:
        k = min(args.folds, len(uniq))
        folds = list(GroupKFold(n_splits=k).split(np.zeros(len(uniq)), None, uniq))
        splits = []
        for tr_s, te_s in folds:
            tr_spk, te_spk = set(uniq[tr_s]), set(uniq[te_s])
            splits.append((np.where(np.isin(groups, list(tr_spk)))[0], np.where(np.isin(tgroups, list(te_spk)))[0]))
        split_note = (f"Entrenamiento con etiquetas '{src}' (supervisión débil) y evaluación con etiquetas "
                      f"'{test_src}' independientes; GroupKFold por hablante (k={k}, hablantes={len(uniq)})")
    elif len(uniq) >= 2:
        k = min(args.folds, len(uniq))
        splits = list(GroupKFold(n_splits=k).split(y, y, groups))
        split_note = f"GroupKFold por hablante (k={k}, hablantes={len(uniq)})"
    else:
        blocks = (d["start_ms"] // 60000)                    # bloques de 60 s dentro de la única grabación
        ub = np.unique(blocks); k = min(args.folds, len(ub))
        splits = list(GroupKFold(n_splits=k).split(y, y, blocks))
        split_note = f"ADVERTENCIA: 1 hablante -> GroupKFold por bloques de 60 s (k={k}); NO es independiente del hablante"
    print(split_note)

    preds = {}
    t0 = time.time()
    for mode in MODES:
        p = np.full(len(y), -1)
        for fi, (tr, te) in enumerate(splits):
            p[te] = train_eval(d, tr, te, mode, n_cls, args, args.seed + fi, test_d=test)
        preds[mode] = p
        print(f"  {mode:11s} macro-F1={f1s(y, p, n_cls)[1]:.3f}  ({time.time() - t0:.0f}s)")

    rng = np.random.default_rng(args.seed)
    src_arr = test if test else d
    if len(np.unique(tgroups)) >= 5:
        units, unit_note = tgroups, "bootstrap de clúster por hablante"
    else:
        units, unit_note = src_arr["rec"].astype(str) + ":" + (src_arr["start_ms"] // 6000).astype(str), "bootstrap por bloques de 6 s (pocos hablantes)"
    cis, diff_ci = bootstrap(y, preds, units, n_cls, args.bootstrap, rng)

    header = (SMOKE_HEADER if test_src not in ("human", "gold_llm") else
              ("Etiquetas humanas" if test_src == "human" else
               "Test con etiquetas de un juez multimodal (VLM); su acuerdo con anotación humana debe reportarse aparte"))
    res = dict(header=header, label_source=src, test_label_source=test_src, n_windows=int(len(y)), n_speakers=int(len(uniq)), split=split_note, ci_method=unit_note,
               class_support={l: int((y == i).sum()) for i, l in enumerate(labels)},
               train_class_support={l: int((y_train_all == i).sum()) for i, l in enumerate(labels)}, modes={}, paired_macro_f1_diff_ci95=diff_ci,
               hyperparams=vars(args))
    for m, p in preds.items():
        per, mac = f1s(y, p, n_cls)
        res["modes"][m] = dict(macro_f1=mac, macro_ci95=cis[m]["macro_ci"],
                               per_class={l: dict(f1=float(per[i]), ci95=cis[m]["per_ci"][i]) for i, l in enumerate(labels)})
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / "ablation_results.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")

    fmt = lambda f, c: f"{f:.3f} [{c[0]:.3f}, {c[1]:.3f}]"
    lines = []
    if test_src not in ("human", "gold_llm"):
        lines += [f"# {SMOKE_HEADER}", ""]
    elif test_src == "gold_llm":
        lines += ["# Ablación con test independiente (juez multimodal)", "",
                  "> El conjunto de prueba fue etiquetado por un VLM a ciegas, sin acceso a las heurísticas. "
                  "El acuerdo juez–humano se reporta en `results/judge_agreement.md` y condiciona la lectura de esta tabla.", ""]
    lines += [f"## Ablación multimodal — entrenamiento: **{src}** · test: **{test_src}**", ""]
    lines += [f"- {split_note}", f"- IC 95 %: {unit_note} (B={args.bootstrap})", f"- Ventanas: {len(y)}", "",
              "| Clase | Soporte | " + " | ".join(MODES) + " |", "|---|---|" + "---|" * len(MODES)]
    for i, l in enumerate(labels):
        lines.append(f"| {l} | {int((y == i).sum())} | " + " | ".join(
            fmt(res['modes'][m]['per_class'][l]['f1'], res['modes'][m]['per_class'][l]['ci95']) for m in MODES) + " |")
    lines.append("| **macro-F1 (disfluencias)** | | " + " | ".join(
        fmt(res['modes'][m]['macro_f1'], res['modes'][m]['macro_ci95']) for m in MODES) + " |")
    lines += ["", "**Diferencias pareadas de macro-F1 (IC 95 %)**", ""]
    lines += [f"- {k}: [{v[0]:+.3f}, {v[1]:+.3f}]" for k, v in diff_ci.items()]
    (out / "ablation_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
