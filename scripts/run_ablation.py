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

from mrcd.constants import (
    FEATURE_SCHEMA_VERSION,  # noqa: E402
    MIN_SUPPORT_FOR_METRIC,
    SMOKE_HEADER,
)
from mrcd.models.fusion import MODES, CrossModalFusion  # noqa: E402

KEYS = ("ac", "pros", "ling", "lpos", "lmask", "kin", "kmask")
NUM_KEYS = ("ac", "pros", "ling", "kin")  # tensores continuos a estandarizar


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
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ds = standardize(d, tr)
    if test_d is not None:
        mu_sd = {
            k: (
                d[k][tr].reshape(-1, d[k].shape[-1]).mean(0),
                d[k][tr].reshape(-1, d[k].shape[-1]).std(0) + 1e-6,
            )
            for k in NUM_KEYS
            if k in d and k in test_d
        }
        ts = dict(test_d)
        for k, (mu, sd) in mu_sd.items():
            ts[k] = ((test_d[k] - mu) / sd).astype(np.float32)
    else:
        ts = ds
    net = CrossModalFusion(
        n_cls,
        ac_dim=ds["ac"].shape[-1],
        ling_dim=ds["ling"].shape[-1],
        d=args.d,
        mode=mode,
        modality_dropout=args.modality_dropout,
        pros_dim=ds["pros"].shape[-1] if "pros" in ds else 0,
    ).to(dev)
    counts = np.bincount(d["y"][tr], minlength=n_cls).astype(float)
    w = torch.tensor(
        np.where(counts > 0, counts.sum() / (n_cls * np.maximum(counts, 1)), 0.0),
        dtype=torch.float32,
        device=dev,
    )
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-3)
    y = torch.as_tensor(d["y"]).to(dev)
    for ep in range(args.epochs):
        net.train()
        perm = np.random.permutation(tr)
        for s in range(0, len(perm), args.bs):
            idx = perm[s : s + args.bs]
            loss = F.cross_entropy(net(to_batch(ds, idx, dev)), y[idx], weight=w)
            opt.zero_grad()
            loss.backward()
            opt.step()
    net.eval()
    preds = []
    with torch.no_grad():
        for s in range(0, len(te), 512):
            preds.append(
                net(to_batch(ts, te[s : s + 512], dev)).argmax(1).cpu().numpy()
            )
    return np.concatenate(preds) if preds else np.zeros(0, np.int64)


def measurable_classes(y, n_cls, min_support=MIN_SUPPORT_FOR_METRIC):
    """Clases de disfluencia con soporte suficiente para que su F1 signifique algo.

    Requisito 10 del goal. Un F1 sobre n=2 no es una medicion: cualquier remuestreo
    bootstrap lo manda a 0 o a 1 y el IC resultante ([0.000, 0.000] en la tabla
    actual para `block`) parece preciso cuando en realidad no hay dato. Esas clases
    se reportan como «sin datos suficientes» y quedan FUERA del macro-F1.
    """
    return [c for c in range(1, n_cls) if int((y == c).sum()) >= min_support]


def f1s(y, p, n_cls, classes=None):
    per = f1_score(y, p, labels=range(n_cls), average=None, zero_division=0)
    cls = (
        measurable_classes(y, n_cls)
        if classes is None
        else [c for c in classes if (y == c).any()]
    )
    macro = float(np.mean(per[cls])) if cls else float("nan")
    return per, macro


def bootstrap(y, preds, units, n_cls, B, rng, classes):
    """`classes` se fija desde el test COMPLETO y no se recalcula por remuestra: si
    cada iteracion usara su propio conjunto de clases medibles, el macro-F1 seria el
    promedio de metricas definidas sobre soportes distintos y su IC no significaria
    nada."""
    uniq = np.unique(units)
    groups = {u: np.where(units == u)[0] for u in uniq}
    stats = {m: {"per": [], "macro": []} for m in preds}
    diffs = {
        k: []
        for k in (
            "trimodal-audio_only",
            "audio_text-audio_only",
            "trimodal-audio_text",
            "trimodal-kinesic_only",
            "kinesic_only-audio_only",
        )
    }
    for _ in range(B):
        idx = np.concatenate(
            [groups[u] for u in rng.choice(uniq, len(uniq), replace=True)]
        )
        mac = {}
        for m, p in preds.items():
            per, mac[m] = f1s(y[idx], p[idx], n_cls, classes)
            stats[m]["per"].append(per)
            stats[m]["macro"].append(mac[m])
        for k in diffs:
            a, b = k.split("-")
            if a in mac and b in mac:
                diffs[k].append(mac[a] - mac[b])
    ci = lambda v: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]
    out = {
        m: {
            "macro_ci": ci(s["macro"]),
            "per_ci": [ci(np.asarray(s["per"])[:, c]) for c in range(n_cls)],
        }
        for m, s in stats.items()
    }
    return out, {k: ci(v) for k, v in diffs.items() if v}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data/windows_gold.npz"))
    ap.add_argument(
        "--test-data",
        default=None,
        help="npz de TEST con etiquetas independientes (p. ej. data/windows_gold.npz). "
        "Con esto, --data se usa solo para ENTRENAR (supervisión débil).",
    )
    ap.add_argument("--allow-auto-labels", action="store_true")
    ap.add_argument(
        "--allow-same-label-source",
        action="store_true",
        help="permite entrenar y evaluar con el mismo origen de etiquetas "
        "(techo supervisado). Por defecto se rechaza por circular.",
    )
    ap.add_argument(
        "--partition",
        choices=("speaker", "recording", "window"),
        default="speaker",
        help="speaker (por defecto, sin fuga) | recording: mismo hablante en train y "
        "test, distinto segmento | window: aleatoria, ventanas solapadas a ambos "
        "lados. Las dos ultimas producen metricas INFLADAS y no se exportan como "
        "oficiales.",
    )
    ap.add_argument(
        "--leaky-ok",
        action="store_true",
        help="reconoce explicitamente que se pide una particion con fuga. Sin esta "
        "bandera, --partition recording|window aborta.",
    )
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

    # --- requisito 7: guardia de fuga, antes de gastar computo.
    # Medido con scripts/identity_probe.py sobre este mismo corpus: una sonda lineal
    # identifica cual de 17 charlas es una ventana de 3 s con 75 % de exactitud
    # balanceada usando SOLO la cara (12.8x el azar); la rama acustica llega a 10.9x.
    # Dejar material de la misma charla a ambos lados del corte le entrega esa señal
    # al clasificador y el F1 resultante mide reconocimiento de charla, no de
    # disfluencia. No es un problema del video: es del protocolo.
    leaky = args.partition != "speaker"
    if leaky and not args.leaky_ok:
        sys.exit(
            f"--partition {args.partition} deja material de la misma charla en entrenamiento\n"
            "y prueba. El macro-F1 resultante esta INFLADO y no es una metrica del sistema.\n"
            "  speaker   : folds disjuntos por hablante (por defecto, sin fuga)\n"
            "  recording : mismo hablante, distinto segmento -> mide fuga intra-charla\n"
            "  window    : aleatoria -> fuga intra-charla + ventanas solapadas (83 %)\n"
            "Si es deliberado (experimento de fuga del pre-registro), añade --leaky-ok.\n"
            "El resultado se escribira como ablation_table.LEAKY.md y NO como oficial."
        )

    # --- requisito 11: el esquema de rasgos debe coincidir entre datos y codigo.
    for name, arr in (("--data", d), ("--test-data", test)):
        if arr is None:
            continue
        v = int(arr["schema_version"]) if "schema_version" in arr else None
        if v is None:
            print(
                f"AVISO: {name} no declara schema_version (npz anterior a la v"
                f"{FEATURE_SCHEMA_VERSION}). Reconstruyelo con scripts/build_windows.py."
            )
        elif v != FEATURE_SCHEMA_VERSION:
            sys.exit(
                f"{name} tiene esquema de rasgos v{v} y el codigo espera "
                f"v{FEATURE_SCHEMA_VERSION}.\nReconstruye el npz con "
                "scripts/build_windows.py en vez de mezclar versiones en silencio."
            )
    kin_norm = str(d.get("kinesic_norm", "desconocida"))
    if test is not None and str(test.get("kinesic_norm", "desconocida")) != kin_norm:
        sys.exit(
            "Train y test usan normalizacion cinesica distinta "
            f"('{kin_norm}' vs '{test.get('kinesic_norm')}'). El modelo veria "
            "escalas incompatibles entre entrenamiento y evaluacion."
        )

    # --- guardia contra la circularidad, antes de gastar horas de cómputo.
    # El diseño de supervisión débil exige que el origen de las etiquetas de
    # ENTRENAMIENTO sea distinto del de las de TEST. Si coinciden, el resultado
    # mide cuánto se ajusta el modelo a un etiquetador, no cuánto acierta, y
    # tiene muy buena pinta justo por eso.
    if args.test_data:
        if Path(args.test_data).resolve() == Path(args.data).resolve():
            sys.exit(
                f"--data y --test-data apuntan al MISMO archivo ({args.data}).\n"
                "Entrenarías y evaluarías sobre las mismas etiquetas. Para supervisión\n"
                "débil: --data data/windows_auto.npz --test-data data/windows_gold.npz"
            )
        if src == test_src and not args.allow_same_label_source:
            sys.exit(
                f"Entrenamiento y test comparten origen de etiquetas ('{src}').\n"
                "Eso es circular: el test hereda los sesgos del mismo etiquetador que\n"
                "produjo el entrenamiento, y el macro-F1 sale inflado.\n"
                "Para supervisión débil: --data data/windows_auto.npz "
                "--test-data data/windows_gold.npz\n"
                "Si de verdad quieres el techo supervisado (mismo origen, hablantes\n"
                "separados), pásalo explícito con --allow-same-label-source."
            )
    if test_src not in ("human", "gold_llm") and not args.allow_auto_labels:
        sys.exit(
            f"Etiquetas de test '{test_src}': evaluar sobre las mismas heurísticas que etiquetan es circular. "
            "Usa --test-data data/windows_gold.npz (juez multimodal) o etiquetas humanas; "
            "--allow-auto-labels solo para prueba de humo."
        )
    labels = [str(x) for x in d["labels"]]
    n_cls = len(labels)
    y_train_all, groups = d["y"], d["group"]
    y = test["y"] if test else d["y"]
    tgroups = test["group"] if test else groups
    uniq = np.unique(groups)
    if test is not None:
        # Unidad de agrupamiento segun --partition. `speaker` es la unica sin fuga;
        # `recording` y `window` existen para MEDIR la fuga, no para reportar el
        # sistema (ver la guardia de arriba y PREREGISTRO.md).
        if args.partition == "speaker":
            gunit, tunit = groups, tgroups
        elif args.partition == "recording":
            gunit, tunit = d["rec"], (test["rec"] if test else d["rec"])
        else:  # window: cada ventana es su propio grupo
            gunit = np.arange(len(groups)).astype(str)
            tunit = np.arange(len(tgroups)).astype(str)
        uunit = np.unique(gunit)
        k = min(args.folds, len(uunit))
        folds = list(GroupKFold(n_splits=k).split(np.zeros(len(uunit)), None, uunit))
        splits = []
        for tr_s, te_s in folds:
            tr_u, te_u = set(uunit[tr_s]), set(uunit[te_s])
            tr_idx = np.where(np.isin(gunit, list(tr_u)))[0]
            te_idx = np.where(np.isin(tunit, list(te_u)))[0]
            if args.partition == "window":
                # Sin correspondencia entre unidades de train y test: se reparte el
                # test al azar, que es justamente el protocolo que produce la fuga.
                rs = np.random.default_rng(args.seed + len(splits))
                perm = rs.permutation(len(tgroups))
                te_idx = np.array_split(perm, k)[len(splits)]
                tr_idx = np.arange(len(groups))
            splits.append((tr_idx, te_idx))
        tag = {
            "speaker": "por hablante (sin fuga)",
            "recording": "por segmento — MISMO hablante a ambos lados (fuga intra-charla)",
            "window": "aleatoria por ventana — fuga intra-charla + solapamiento 83 % (FUGA MAXIMA)",
        }
        split_note = (
            f"Entrenamiento con etiquetas '{src}' (supervisión débil) y evaluación con etiquetas "
            f"'{test_src}' independientes; GroupKFold {tag[args.partition]} "
            f"(k={k}, unidades={len(uunit)}, hablantes={len(uniq)})"
        )
    elif len(uniq) >= 2:
        k = min(args.folds, len(uniq))
        splits = list(GroupKFold(n_splits=k).split(y, y, groups))
        split_note = f"GroupKFold por hablante (k={k}, hablantes={len(uniq)})"
    else:
        blocks = d["start_ms"] // 60000  # bloques de 60 s dentro de la única grabación
        ub = np.unique(blocks)
        k = min(args.folds, len(ub))
        splits = list(GroupKFold(n_splits=k).split(y, y, blocks))
        split_note = f"ADVERTENCIA: 1 hablante -> GroupKFold por bloques de 60 s (k={k}); NO es independiente del hablante"
    print(split_note)

    preds = {}
    t0 = time.time()
    for mode in MODES:
        p = np.full(len(y), -1)
        for fi, (tr, te) in enumerate(splits):
            p[te] = train_eval(
                d, tr, te, mode, n_cls, args, args.seed + fi, test_d=test
            )
        preds[mode] = p
        print(
            f"  {mode:11s} macro-F1={f1s(y, p, n_cls)[1]:.3f}  ({time.time() - t0:.0f}s)"
        )

    measurable = measurable_classes(y, n_cls)
    unmeasurable = [labels[c] for c in range(1, n_cls) if c not in measurable]
    if unmeasurable:
        print(f"  clases SIN datos suficientes (soporte < {MIN_SUPPORT_FOR_METRIC}), "
              f"excluidas del macro-F1: {', '.join(unmeasurable)}")

    rng = np.random.default_rng(args.seed)
    src_arr = test if test else d
    if len(np.unique(tgroups)) >= 5:
        units, unit_note = tgroups, "bootstrap de clúster por hablante"
    else:
        units, unit_note = (
            src_arr["rec"].astype(str)
            + ":"
            + (src_arr["start_ms"] // 6000).astype(str),
            "bootstrap por bloques de 6 s (pocos hablantes)",
        )
    cis, diff_ci = bootstrap(y, preds, units, n_cls, args.bootstrap, rng, measurable)

    header = (
        SMOKE_HEADER
        if test_src not in ("human", "gold_llm")
        else (
            "Etiquetas humanas"
            if test_src == "human"
            else "Test con etiquetas de un juez multimodal (VLM); su acuerdo con anotación humana debe reportarse aparte"
        )
    )
    res = dict(
        header=header,
        label_source=src,
        test_label_source=test_src,
        n_windows=int(len(y)),
        n_speakers=int(len(uniq)),
        split=split_note,
        ci_method=unit_note,
        class_support={l: int((y == i).sum()) for i, l in enumerate(labels)},
        train_class_support={
            l: int((y_train_all == i).sum()) for i, l in enumerate(labels)
        },
        modes={},
        paired_macro_f1_diff_ci95=diff_ci,
        hyperparams=vars(args),
    )
    for m, p in preds.items():
        per, mac = f1s(y, p, n_cls)
        res["modes"][m] = dict(
            macro_f1=mac,
            macro_ci95=cis[m]["macro_ci"],
            per_class={
                l: dict(f1=float(per[i]), ci95=cis[m]["per_ci"][i])
                for i, l in enumerate(labels)
            },
        )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    json_name = "ablation_results.LEAKY.json" if leaky else "ablation_results.json"
    (out / json_name).write_text(
        json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    fmt = lambda f, c: f"{f:.3f} [{c[0]:.3f}, {c[1]:.3f}]"
    lines = []
    if test_src not in ("human", "gold_llm"):
        lines += [f"# {SMOKE_HEADER}", ""]
    elif test_src == "gold_llm":
        lines += [
            "# Ablación con test independiente (juez multimodal)",
            "",
            "> El conjunto de prueba fue etiquetado por un VLM a ciegas, sin acceso a las heurísticas. "
            "El acuerdo juez–humano se reporta en `results/judge_agreement.md` y condiciona la lectura de esta tabla.",
            "",
        ]
    lines += [
        f"## Ablación multimodal — entrenamiento: **{src}** · test: **{test_src}**",
        "",
    ]
    lines += [
        f"- {split_note}",
        f"- IC 95 %: {unit_note} (B={args.bootstrap})",
        f"- Ventanas: {len(y)}",
        "",
        "| Clase | Soporte | " + " | ".join(MODES) + " |",
        "|---|---|" + "---|" * len(MODES),
    ]
    for i, l in enumerate(labels):
        sup = int((y == i).sum())
        if i != 0 and i not in measurable:
            cells = " | ".join(["sin datos suficientes"] * len(MODES))
        else:
            cells = " | ".join(
                fmt(res["modes"][m]["per_class"][l]["f1"],
                    res["modes"][m]["per_class"][l]["ci95"]) for m in MODES)
        lines.append(f"| {l} | {sup} | {cells} |")
    lines.append(
        "| **macro-F1 (disfluencias)** | | "
        + " | ".join(
            fmt(res["modes"][m]["macro_f1"], res["modes"][m]["macro_ci95"])
            for m in MODES
        )
        + " |"
    )
    lines += ["", "**Diferencias pareadas de macro-F1 (IC 95 %)**", ""]
    lines += [f"- {k}: [{v[0]:+.3f}, {v[1]:+.3f}]" for k, v in diff_ci.items()]
    table_name = "ablation_table.LEAKY.md" if leaky else "ablation_table.md"
    (out / table_name).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
