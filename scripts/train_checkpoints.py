#!/usr/bin/env python3
"""Entrena los tres modos y GUARDA los checkpoints para inferencia.

`run_ablation.py` entrena dentro del bucle de validación cruzada y descarta los
modelos: sirve para medir, no para desplegar. Este script entrena una vez sobre
todo el conjunto y persiste cada modo en `models/ckpt_{modo}.pt`.

Cada checkpoint lleva dentro las estadísticas de normalización (media y
desviación por columna) usadas en entrenamiento. Sin eso la inferencia sobre un
vídeo nuevo normalizaría con estadísticas distintas y las predicciones no
significarían nada.

También lleva un bloque de procedencia con el origen de las etiquetas. Un
checkpoint entrenado con etiquetas heurísticas produce predicciones útiles para
inspeccionar el pipeline, no evidencia validada: el bloque lo deja escrito para
que ningún informe posterior lo presente como otra cosa.

Uso:
    python scripts/train_checkpoints.py --data data/windows_auto.npz
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.constants import LABELS, SMOKE_HEADER  # noqa: E402
from core.models.fusion import MODES, CrossModalFusion, count_params  # noqa: E402

NUM_KEYS = ("ac", "pros", "ling", "kin")
BATCH_KEYS = ("ac", "pros", "ling", "lpos", "lmask", "kin", "kmask")


def stats(d: dict) -> dict:
    """Media/desviación por columna de cada tensor continuo."""
    out = {}
    for k in NUM_KEYS:
        if k not in d:
            continue
        x = d[k].reshape(-1, d[k].shape[-1])
        out[k] = (x.mean(0).astype(np.float32), (x.std(0) + 1e-6).astype(np.float32))
    return out


def apply_stats(d: dict, st: dict) -> dict:
    out = dict(d)
    for k, (mu, sd) in st.items():
        if k in d:
            out[k] = ((d[k] - mu) / sd).astype(np.float32)
    return out


def to_batch(d: dict, idx, dev: str) -> dict:
    b = {k: torch.as_tensor(d[k][idx]).to(dev) for k in BATCH_KEYS if k in d}
    for k in NUM_KEYS:
        if k in b:
            b[k] = b[k].float()
    for k in ("lmask", "kmask"):
        if k in b:
            b[k] = b[k].bool()
    return b


def train_one(d: dict, mode: str, args, n_cls: int, dev: str):
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    st = stats(d)
    ds = apply_stats(d, st)
    net = CrossModalFusion(n_cls, ac_dim=ds["ac"].shape[-1], ling_dim=ds["ling"].shape[-1],
                           d=args.d, mode=mode, modality_dropout=args.modality_dropout,
                           pros_dim=ds["pros"].shape[-1] if "pros" in ds else 0).to(dev)
    counts = np.bincount(d["y"], minlength=n_cls).astype(float)
    w = torch.tensor(np.where(counts > 0, counts.sum() / (n_cls * np.maximum(counts, 1)), 0.0),
                     dtype=torch.float32, device=dev)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-3)
    y = torch.as_tensor(d["y"]).to(dev)
    n = len(d["y"])
    for ep in range(args.epochs):
        net.train()
        perm = np.random.permutation(n)
        tot = 0.0
        for s in range(0, n, args.bs):
            idx = perm[s:s + args.bs]
            loss = nn.functional.cross_entropy(net(to_batch(ds, idx, dev)), y[idx], weight=w)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss.detach()) * len(idx)
        print(f"  [{mode}] época {ep + 1}/{args.epochs}  pérdida={tot / n:.4f}", flush=True)
    return net, st, w.detach().cpu().tolist()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data/windows_auto.npz"))
    ap.add_argument("--out", default=str(ROOT / "models"))
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--modality-dropout", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    ap.add_argument("--checkpoint-name", default=None,
                    help="nombre del archivo cuando se entrena exactamente un modo")
    a = ap.parse_args()

    dp = Path(a.data)
    if not dp.exists():
        print(f"[train] no existe {dp}. Corre antes scripts/build_windows.py --labels auto")
        sys.exit(1)
    z = np.load(dp, allow_pickle=False)
    d = {k: z[k] for k in z.files if k not in ("labels", "label_source")}
    labels = [s.decode() if isinstance(s, bytes) else str(s) for s in z["labels"]] if "labels" in z.files else list(LABELS)
    src = str(z["label_source"]) if "label_source" in z.files else "auto"
    n_cls = len(labels)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(a.out).resolve(); out.mkdir(parents=True, exist_ok=True)

    print(f"[train] {len(d['y'])} ventanas · etiquetas '{src}' · dispositivo {dev}")
    if src not in ("human", "gold_llm"):
        print(f"[train] AVISO: {SMOKE_HEADER}")

    manifest = {"created": dt.datetime.now().isoformat(timespec="seconds"),
                "label_source": src, "n_windows": int(len(d["y"])), "labels": labels,
                "checkpoints": {}}
    if a.checkpoint_name and len(a.modes) != 1:
        ap.error("--checkpoint-name requiere exactamente un modo")
    for mode in a.modes:
        print(f"[train] modo {mode}")
        net, st, class_weights = train_one(d, mode, a, n_cls, dev)
        path = out / (a.checkpoint_name or f"ckpt_{mode}.pt")
        torch.save({
            "state_dict": net.state_dict(),
            "mode": mode,
            "labels": labels,
            "dims": {"ac": int(d["ac"].shape[-1]), "ling": int(d["ling"].shape[-1]),
                     "pros": int(d["pros"].shape[-1]) if "pros" in d else 0, "d": a.d},
            "norm": {k: (mu.tolist(), sd.tolist()) for k, (mu, sd) in st.items()},
            "provenance": {
                "label_source": src,
                "n_windows": int(len(d["y"])),
                "trained": manifest["created"],
                "warning": None if src in ("human", "gold_llm") else SMOKE_HEADER,
            },
            "config": {"epochs": a.epochs, "lr": a.lr, "bs": a.bs, "d": a.d,
                       "modality_dropout": a.modality_dropout, "seed": a.seed},
            "loss": {"name": "weighted_cross_entropy", "class_weights": class_weights},
        }, path)
        manifest["checkpoints"][mode] = {"path": str(path.relative_to(ROOT.resolve())),
                                         "params": count_params(net)}
        print(f"  -> {path}  ({count_params(net):,} parámetros)")

    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[train] listo. Manifiesto en {out / 'manifest.json'}")


if __name__ == "__main__":
    main()
