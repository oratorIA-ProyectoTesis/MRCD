#!/usr/bin/env python3
"""R1 — Sonda de identidad: ¿cuánta información de charla porta cada rama?

Entrena una sonda LINEAL que predice de cuál de las K charlas proviene la ventana,
usando una sola rama por vez (cinésica, prosódica, lingüística o acústica). Si una
rama identifica la charla muy por encima del azar, cualquier protocolo que deje
material de la misma charla en train y test le regala esa señal al clasificador de
disfluencias.

Por qué LINEAL: es la convención de `linear probing`. Mide información linealmente
accesible en los rasgos, no la capacidad de la sonda. Una sonda profunda mediría lo
segundo y confundiría el resultado.

Por qué por SEGMENTO: agrupar por segmento (`rec`) replica exactamente la partición
P2 del pre-registro. Entrenar y evaluar sobre ventanas sueltas daría casi 100 % por
el solapamiento del 83 % entre ventanas contiguas, que es duplicación temporal y no
identidad de charla.

Métrica: exactitud balanceada (macro-recall). El azar es 1/K con independencia del
desbalance de ventanas por charla; la exactitud cruda no lo es (la charla más larga
aporta el 16 % de las ventanas).

Uso:
    python scripts/identity_probe.py --data data/windows_auto.npz
    python scripts/identity_probe.py --data data/windows_auto.npz --normalize speaker
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import balanced_accuracy_score  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402


BRANCHES = ("kinesic", "prosodic", "linguistic", "acoustic")


# --------------------------------------------------------------- agregación
def _pool_seq(x: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """(N, T, C) -> (N, 2C) concatenando media y desviación sobre el eje temporal.

    Con `mask` (N, T) booleana, solo promedia los pasos válidos; las filas sin ningún
    paso válido quedan en cero (no se inventan valores).
    """
    if mask is None:
        return np.concatenate([x.mean(1), x.std(1)], 1)
    m = mask.astype(np.float64)[..., None]
    n = m.sum(1)
    safe = np.maximum(n, 1.0)
    mu = (x * m).sum(1) / safe
    var = (((x - mu[:, None, :]) ** 2) * m).sum(1) / safe
    out = np.concatenate([mu, np.sqrt(var)], 1)
    out[(n[:, 0] == 0)] = 0.0
    return out


def _invariant_seq(x: np.ndarray) -> np.ndarray:
    """(N, T, C) -> rasgos INVARIANTES a transformacion afin por canal.

    El z-score por hablante resta la media y divide por la desviacion de cada canal,
    asi que cualquier rasgo construido con esos dos momentos queda en cero por
    construccion. Estos dos no:

      - autocorrelacion de retardo 1 por canal (C rasgos): normalizada por la
        varianza, luego inmune a escala y desplazamiento.
      - correlacion de Pearson entre pares de canales (C*(C-1)/2): idem.

    Miden DINAMICA y ACOPLAMIENTO, no nivel. Si la charla sigue siendo identificable
    con esto despues de normalizar, el z-score no basta como remedio.
    """
    xc = x - x.mean(1, keepdims=True)
    sd = xc.std(1) + 1e-8  # (N, C)
    ac1 = (xc[:, :-1, :] * xc[:, 1:, :]).mean(1) / (sd**2)  # (N, C)
    z = xc / sd[:, None, :]
    C = x.shape[2]
    iu = np.triu_indices(C, k=1)
    corr = np.einsum("ntc,ntd->ncd", z, z) / x.shape[1]  # (N, C, C)
    return np.concatenate([ac1, corr[:, iu[0], iu[1]]], 1)


def branch_raw(d: dict, branch: str) -> tuple[np.ndarray, np.ndarray | None]:
    """Serie cruda y mascara de una rama. La agregacion ocurre dentro del fold."""
    seqs = {
        "kinesic": ("kin", "kmask"),
        "linguistic": ("ling", "lmask"),
        "acoustic": ("ac", None),
    }
    if branch == "prosodic":
        return d["pros"].astype(np.float64), None
    key, mkey = seqs[branch]
    mask = d[mkey].astype(bool) if mkey else None
    return d[key].astype(np.float64), mask


# ------------------------------------------------------- normalización R3
def speaker_zscore_seq(
    x: np.ndarray, speakers: np.ndarray, fit_rows: np.ndarray
) -> np.ndarray:
    """z-score por hablante y por canal sobre la SERIE (N, T, C), no sobre los rasgos.

    ◆ El orden importa y es la fuente de un error facil de cometer. Normalizar los
    rasgos YA agregados anula la varianza entre hablantes por construccion, para
    cualquier rasgo, y la sonda cae a 1/K sin que eso signifique nada. El pipeline
    real (`core/dataset.py`) normaliza la serie cruda ANTES de agregar, que es lo
    que se replica aqui.

    Estadisticas tomadas solo de `fit_rows` (el train del fold): evita filtrar el
    test y refleja el despliegue real, donde se calibra con material disponible del
    hablante y se aplica a material nuevo suyo.
    """
    out = x.copy()
    for s in np.unique(speakers):
        rows = speakers == s
        fit = rows & fit_rows
        if fit.sum() < 2:  # sin material de calibración: se deja crudo
            continue
        flat = x[fit].reshape(-1, x.shape[-1])
        mu = flat.mean(0)
        sd = flat.std(0) + 1e-8
        out[rows] = (x[rows] - mu) / sd
    return out


# ------------------------------------------------------------------ sonda
def run_probe(
    raw: np.ndarray,
    mask: np.ndarray | None,
    features: str,
    talk: np.ndarray,
    seg: np.ndarray,
    speakers: np.ndarray,
    normalize: str,
    folds: int,
    seed: int,
    max_iter: int,
) -> tuple[np.ndarray, np.ndarray]:
    """GroupKFold por segmento. Devuelve (y_true, y_pred) sobre todas las ventanas.

    Recibe la serie CRUDA y agrega dentro del fold, porque la normalizacion por
    hablante actua sobre la serie y sus estadisticas salen solo del train del fold.
    """
    pred = np.full(len(talk), -1, dtype=np.int64)
    k = min(folds, len(np.unique(seg)))
    for tr, te in GroupKFold(n_splits=k).split(raw, talk, seg):
        seq = raw
        if normalize == "speaker" and raw.ndim == 3:
            fit_rows = np.zeros(len(raw), bool)
            fit_rows[tr] = True
            seq = speaker_zscore_seq(raw, speakers, fit_rows)
        if raw.ndim == 3:
            Xf = (
                _invariant_seq(seq) if features == "invariant" else _pool_seq(seq, mask)
            )
        else:  # prosodia: ya viene agregada
            Xf = seq
        sc = StandardScaler().fit(Xf[tr])
        # multinomial es el comportamiento por defecto con lbfgs en sklearn >= 1.5;
        # el argumento explicito fue retirado de la API.
        clf = LogisticRegression(
            max_iter=max_iter,
            class_weight="balanced",
            random_state=seed,
        )
        clf.fit(sc.transform(Xf[tr]), talk[tr])
        pred[te] = clf.predict(sc.transform(Xf[te]))
    return talk, pred


def bootstrap_ci(
    y: np.ndarray, p: np.ndarray, units: np.ndarray, B: int, rng: np.random.Generator
) -> list[float]:
    """IC 95 % remuestreando CLÚSTERES (segmentos), no ventanas: las ventanas de un
    mismo segmento se solapan un 83 % y no son independientes."""
    uniq = np.unique(units)
    groups = {u: np.where(units == u)[0] for u in uniq}
    vals = []
    for _ in range(B):
        idx = np.concatenate(
            [groups[u] for u in rng.choice(uniq, len(uniq), replace=True)]
        )
        if len(np.unique(y[idx])) < 2:
            continue
        vals.append(balanced_accuracy_score(y[idx], p[idx]))
    if not vals:
        return [float("nan"), float("nan")]
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data/windows_auto.npz"))
    ap.add_argument("--normalize", choices=("raw", "speaker"), default="raw")
    ap.add_argument(
        "--features",
        choices=("moments", "invariant"),
        default="moments",
        help="moments = media+desv (R1). invariant = autocorr + corr cruzada (R4), "
        "inmunes al z-score por canal.",
    )
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--max-iter", type=int, default=2000)
    ap.add_argument("--out", default=str(ROOT / "results"))
    args = ap.parse_args()

    d = dict(np.load(args.data, allow_pickle=False))
    speakers = d["group"].astype(str)
    seg = d["rec"].astype(str)

    # Una charla con un solo segmento nunca puede aparecer en train y test a la vez:
    # la sonda jamás podría predecirla y arrastraría la exactitud balanceada hacia
    # abajo por un motivo ajeno a la fuga. Se excluye y se declara.
    n_seg = {s: len(np.unique(seg[speakers == s])) for s in np.unique(speakers)}
    dropped = sorted(s for s, n in n_seg.items() if n < 2)
    keep = ~np.isin(speakers, dropped)
    speakers, seg = speakers[keep], seg[keep]
    talks = np.unique(speakers)
    talk = np.searchsorted(talks, speakers)
    K = len(talks)
    chance = 1.0 / K

    print(
        f"charlas usables: {K} (excluidas por tener 1 solo segmento: {dropped or 'ninguna'})"
    )
    print(
        f"segmentos: {len(np.unique(seg))} | ventanas: {keep.sum()} | azar = {chance:.4f}"
    )
    print(f"normalizacion: {args.normalize}\n")

    rng = np.random.default_rng(args.seed)
    sub = {k: (v[keep] if getattr(v, 'ndim', 0) and len(v) == len(keep) else v)
           for k, v in d.items()}
    rows = {}
    branches = [
        b for b in BRANCHES if not (args.features == "invariant" and b == "prosodic")
    ]
    for br in branches:
        raw, mask = branch_raw(sub, br)
        y, p = run_probe(
            raw,
            mask,
            args.features, talk, seg, speakers, args.normalize, args.folds, args.seed, args.max_iter
        )
        bacc = float(balanced_accuracy_score(y, p))
        ci = bootstrap_ci(y, p, seg, args.bootstrap, rng)
        rows[br] = dict(
            balanced_accuracy=bacc,
            ci95=ci,
            ratio_vs_chance=bacc / chance,
            n_features=int(raw.shape[-1]),
        )
        print(
            f"  {br:11s} bacc={bacc:.4f} [{ci[0]:.4f}, {ci[1]:.4f}]  = {bacc / chance:5.1f}x azar"
        )

    res = dict(
        data=str(Path(args.data).name),
        normalize=args.normalize,
        n_talks=K,
        n_segments=int(len(np.unique(seg))),
        n_windows=int(keep.sum()),
        dropped_single_segment=dropped,
        chance=chance,
        folds=args.folds,
        bootstrap=args.bootstrap,
        seed=args.seed,
        branches=rows,
    )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = f"identity_probe_{args.features}_{args.normalize}"
    (out / f"{tag}.json").write_text(
        json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    lines = [
        f"# {'R4' if args.features == 'invariant' else 'R1'} — Sonda de identidad de charla ({args.features} / {args.normalize})",
        "",
        "> Sonda lineal por rama sobre GroupKFold por segmento (réplica de la partición P2).",
        "> Mide información de charla linealmente accesible en cada rama. Resultado",
        "> descriptivo pre-especificado: se reporta sin prueba de hipótesis ni corrección.",
        "",
        f"- Charlas: {K} · segmentos: {len(np.unique(seg))} · ventanas: {keep.sum()}",
        f"- Excluidas por tener un solo segmento: {', '.join(dropped) if dropped else 'ninguna'}",
        f"- Azar (exactitud balanceada): {chance:.4f}",
        f"- IC 95 %: bootstrap de clúster por segmento (B={args.bootstrap})",
        "",
        "| Rama | Canales | Exactitud balanceada | IC 95 % | × azar |",
        "|---|---|---|---|---|",
    ]
    for br in branches:
        r = rows[br]
        lines.append(
            f"| {br} | {r['n_features']} | {r['balanced_accuracy']:.4f} | "
            f"[{r['ci95'][0]:.4f}, {r['ci95'][1]:.4f}] | {r['ratio_vs_chance']:.1f}× |"
        )
    lines += [
        "",
        "Lectura: una rama muy por encima del azar identifica la charla por sí sola. "
        "Cualquier protocolo que deje material de la misma charla en entrenamiento y "
        "prueba le entrega esa señal al clasificador de disfluencias.",
    ]
    (out / f"{tag}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n-> {out / (tag + '.md')}")


if __name__ == "__main__":
    main()
