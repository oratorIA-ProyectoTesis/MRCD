"""Carga de checkpoints e inferencia por lotes.

Vive en la libreria y no en `scripts/` porque `DisfluencyEngine` lo necesita: si
el paquete importara de `scripts/`, instalarlo en otro proyecto arrastraria las
herramientas de linea de comandos y sus dependencias (argparse, rutas del repo,
yt-dlp). La dependencia va en el sentido correcto: los scripts usan la libreria.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from mrcd.constants import FEATURE_SCHEMA_VERSION
from mrcd.models.fusion import CrossModalFusion

BATCH_KEYS = ("ac", "pros", "ling", "lpos", "lmask", "kin", "kmask")
NUM_KEYS = ("ac", "pros", "ling", "kin")


def load_ckpt(
    path: Path,
    dev: str,
    expect_schema: int | None = FEATURE_SCHEMA_VERSION,
    allow_schema_mismatch: bool = False,
):
    """Requisito 11: un checkpoint incompatible debe FALLAR, no adaptarse en silencio.

    El artefacto de `models_stale/` tiene `ling=10, pros=0` frente al `ling=12,
    pros=9` del modelo contextual. Cargarlo «adaptandose» produce predicciones que
    parecen validas y no lo son, y asi se perdio el rastro del baseline v1
    (models/BASELINE.md). Mas vale un error que un numero equivocado.
    """
    ck = torch.load(path, map_location=dev, weights_only=False)
    got = ck.get("schema_version")
    if expect_schema is not None and not allow_schema_mismatch:
        if got is None:
            raise ValueError(
                f"{path.name} no declara schema_version (es anterior a la v{expect_schema}).\n"
                "No se puede saber con que definicion de rasgos se entreno. Reentrenalo con\n"
                "scripts/train_checkpoints.py, o pasa allow_schema_mismatch=True si asumes\n"
                "el riesgo de forma explicita."
            )
        if int(got) != int(expect_schema):
            raise ValueError(
                f"{path.name} tiene esquema de rasgos v{got} y el codigo espera "
                f"v{expect_schema}.\nLas dimensiones o el significado de los rasgos "
                "cambiaron: las predicciones no serian comparables."
            )
    net = CrossModalFusion(
        len(ck["labels"]),
        ac_dim=ck["dims"]["ac"],
        ling_dim=ck["dims"]["ling"],
        d=ck["dims"]["d"],
        mode=ck["mode"],
        pros_dim=ck["dims"].get("pros", 0),
    ).to(dev)
    net.load_state_dict(ck["state_dict"])
    net.eval()
    norm = {
        k: (np.asarray(mu, np.float32), np.asarray(sd, np.float32))
        for k, (mu, sd) in ck["norm"].items()
    }
    return net, norm, ck["labels"], ck.get("provenance", {})


@torch.no_grad()
def predict(net, norm, d: dict, dev: str, bs: int = 128):
    n = len(d["ac"])
    probs = []
    for s in range(0, n, bs):
        sl = slice(s, min(s + bs, n))
        b = {}
        for k in BATCH_KEYS:
            if k not in d:
                continue
            x = d[k][sl]
            if k in norm:
                mu, sd = norm[k]
                x = ((x - mu) / sd).astype(np.float32)
            t = torch.as_tensor(x).to(dev)
            b[k] = (
                t.bool()
                if k in ("lmask", "kmask")
                else (t.float() if k in NUM_KEYS else t)
            )
        probs.append(torch.softmax(net(b), -1).cpu().numpy())
    return np.concatenate(probs, 0)
