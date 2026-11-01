"""Supervised adaptation of the final layer with human turn-level labels (new module, 2026-10-30).

The distributed checkpoint was trained with weak (heuristic) labels. Here a small head is fitted on the
FROZEN pooled representation of the base network, using only existing human annotation, with
multiple-instance learning: the turn label is "the class is present somewhere in the turn", the turn score
is the maximum over its windows (the same aggregation used at inference), loss = BCE on that score.

Nothing here overwrites the base checkpoint: ``save_adapted`` writes a NEW file that stores the head, the
class list, the hash of the base checkpoint and the training configuration. The base network is loaded
from the original file.
"""

from __future__ import annotations

import copy
import hashlib
import io
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from mrcd.inference import BATCH_KEYS, NUM_KEYS


# ----------------------------------------------------------------------------- features
@torch.no_grad()
def predict_pooled(net, norm: dict, d: dict, dev: str = "cpu", bs: int = 128):
    """Like ``mrcd.inference.predict`` but also returns the pooled vector that feeds ``net.head``.

    Returns ``(probs (N,K), pooled (N,H))``. The head input is captured with a forward hook, so the base
    network code is not duplicated; ``softmax(net.head(pooled)) == probs`` holds exactly.
    """
    n = len(d["ac"])
    captured: list[torch.Tensor] = []
    hook = net.head.register_forward_hook(
        lambda m, inp, out: captured.append(inp[0].detach().cpu())
    )
    probs = []
    try:
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
    finally:
        hook.remove()
    return np.concatenate(probs, 0), torch.cat(captured, 0).numpy()


# ----------------------------------------------------------------------------- head
class Head(nn.Module):
    """Standardize -> linear (or 1-hidden-layer MLP) -> one logit per class (sigmoid at scoring time)."""

    def __init__(
        self,
        in_dim: int,
        n_classes: int,
        kind: str = "linear",
        hidden: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()
        if kind not in ("linear", "mlp"):
            raise ValueError("kind must be 'linear' or 'mlp'")
        self.kind, self.hidden = kind, hidden
        self.register_buffer("mu", torch.zeros(in_dim))
        self.register_buffer("sd", torch.ones(in_dim))
        self.net = (
            nn.Linear(in_dim, n_classes)
            if kind == "linear"
            else nn.Sequential(
                nn.Linear(in_dim, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, n_classes),
            )
        )

    def forward(self, x):
        return self.net((x - self.mu) / self.sd)


def unit_max(
    window_logits: torch.Tensor, unit_idx: torch.Tensor, n_units: int
) -> torch.Tensor:
    """Max over the windows of each unit (differentiable). Every unit must own at least one window."""
    out = window_logits.new_full((n_units, window_logits.shape[1]), -1e9)
    return out.scatter_reduce(
        0,
        unit_idx[:, None].expand_as(window_logits),
        window_logits,
        reduce="amax",
        include_self=False,
    )


def fit_head(
    X: np.ndarray,
    unit_idx: np.ndarray,
    Y: np.ndarray,
    *,
    kind: str = "linear",
    wd: float = 1e-3,
    lr: float = 1e-2,
    epochs: int = 100,
    snapshot_every: int = 10,
    hidden: int = 32,
    seed: int = 13,
) -> dict[int, dict]:
    """Full-batch Adam on turn-level BCE with max-over-windows pooling.

    ``X`` (Nw,H) pooled window features, ``unit_idx`` (Nw,) turn index in ``[0, U)``, ``Y`` (U,C) in {0,1}.
    Returns ``{epoch: state_dict}`` snapshots (CPU copies) so the caller can pick the epoch on a TUNING fold.
    """
    torch.manual_seed(seed)
    U, C = Y.shape
    Xt = torch.as_tensor(X, dtype=torch.float32)
    ui = torch.as_tensor(unit_idx, dtype=torch.long)
    Yt = torch.as_tensor(Y, dtype=torch.float32)
    head = Head(X.shape[1], C, kind, hidden)
    head.mu.copy_(Xt.mean(0))
    head.sd.copy_(Xt.std(0).clamp_min(1e-6))
    pos = Yt.sum(0).clamp_min(1.0)
    pos_weight = (U - pos) / pos
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=wd)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    snaps: dict[int, dict] = {}
    for ep in range(1, epochs + 1):
        head.train()
        opt.zero_grad()
        loss = lossf(unit_max(head(Xt), ui, U), Yt)
        loss.backward()
        opt.step()
        if ep % snapshot_every == 0 or ep == epochs:
            snaps[ep] = copy.deepcopy(
                {k: v.detach().cpu() for k, v in head.state_dict().items()}
            )
    return snaps


@torch.no_grad()
def head_scores(
    state: dict, X: np.ndarray, n_classes: int, kind: str, hidden: int = 32
) -> np.ndarray:
    """Per-window sigmoid scores (Nw,C) from a head state dict. Scores, NOT calibrated probabilities."""
    head = Head(X.shape[1], n_classes, kind, hidden)
    head.load_state_dict(state)
    head.eval()
    return torch.sigmoid(head(torch.as_tensor(X, dtype=torch.float32))).numpy()


def average_precision(y, s) -> float:
    """Threshold-free ranking quality (area under precision-recall, step-wise). nan if no positive."""
    y = np.asarray(y).astype(int)
    s = np.asarray(s, float)
    if y.sum() == 0:
        return float("nan")
    order = np.argsort(-s, kind="stable")
    y = y[order]
    tp = np.cumsum(y)
    prec = tp / (np.arange(len(y)) + 1)
    return float((prec * y).sum() / y.sum())


# ----------------------------------------------------------------------------- checkpoint I/O
def sha256_file(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def save_adapted(
    path,
    *,
    base_ckpt: Path,
    classes: list[str],
    state: dict,
    kind: str,
    hidden: int,
    in_dim: int,
    meta: dict,
):
    """Write a NEW adapted-head checkpoint. Refuses to overwrite an existing file."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": "mrcd-adapted-head-v1",
            "base_ckpt_path": str(base_ckpt),
            "base_ckpt_sha256": sha256_file(base_ckpt),
            "classes": list(classes),
            "head_kind": kind,
            "hidden": hidden,
            "in_dim": in_dim,
            "state": state,
            "meta": meta,
        },
        path,
    )
    return sha256_file(path)


def load_adapted(path, *, expected_sha256: str | None = None) -> tuple[dict, str]:
    """Return ``(checkpoint dict, sha256)``; verifies that the stored base checkpoint hash still matches.

    The file is read ONCE: the bytes that are hashed are the bytes that are deserialized (no
    hash-then-reopen race). Deserialization uses ``weights_only=True`` so an arbitrary pickle is never
    executed, with or without a hash. With ``expected_sha256`` a mismatch raises ``ValueError`` before
    any deserialization.
    """
    data = Path(path).read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256.lower():
        raise ValueError(
            f"adapted-head checkpoint {Path(path).name} has sha256 {digest}, expected {expected_sha256}"
        )
    ck = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
    if ck.get("format") != "mrcd-adapted-head-v1":
        raise ValueError("not an adapted-head checkpoint")
    base = Path(ck["base_ckpt_path"])
    if base.exists() and sha256_file(base) != ck["base_ckpt_sha256"]:
        raise ValueError(
            "base checkpoint hash differs from the one this head was trained on"
        )
    return ck, digest


# ----------------------------------------------------------------------------- Vc (experimental)
#: Frozen Vc head: MLP(32) on the pooled base representation, human DIHANA turn labels (folds 1-3),
#: epoch/thresholds chosen on fold 4. See results/mejoras_20261030/MEJORAS.md. It is opt-in.
VC_RELPATH = "results/mejoras_20261030/checkpoints/ckpt_head_dihana_f123_zero.pt"
VC_SHA256 = "776c72f8e064e42c569e3046f5d237de03aec4a810cabfdef5054d234066ab4a"
#: classes the engine can emit that Vc does NOT cover, with the reason (no compatible human reference)
VC_UNCOVERED = {
    "block": "no compatible human reference (DIHANA '<b>' is ambiguous; n=2 in the original evaluation)",
    "rhetorical_pause": "no human reference with pause onset/offset and function",
    "neutral_pause": "no human reference with pause onset/offset and function",
    "revision": "a human reference exists but the head was not trained for it (V0 detects almost none, see MEJORAS.md)",
}


@dataclass(frozen=True)
class LoadedHead:
    """A hash-verified adapted head, ready to score pooled window vectors."""

    name: str
    path: str
    sha256: str
    base_sha256: str
    classes: tuple[str, ...]
    kind: str
    hidden: int
    state: dict
    thresholds: dict
    meta: dict

    def window_scores(self, pooled: np.ndarray) -> np.ndarray:
        """Per-window sigmoid scores (W, C). Scores, NOT calibrated probabilities."""
        return head_scores(self.state, pooled, len(self.classes), self.kind, self.hidden)

    def unit_decision(self, scores: np.ndarray) -> dict[str, dict]:
        """Turn-level rule used at selection time: max over windows >= per-class threshold."""
        out = {}
        for i, c in enumerate(self.classes):
            thr = float(self.thresholds[c]["threshold"])
            score = float(scores[:, i].max()) if len(scores) else float("nan")
            out[c] = {
                "score": score,
                "threshold": thr,
                "present": bool(score >= thr),
                "experimental": True,
                "calibrated": False,
                "reference_tune_f1": self.thresholds[c].get("tune_f1_at_threshold"),
            }
        return out

    def provenance(self) -> dict:
        m = self.meta
        return {
            "head": self.name,
            "experimental": True,
            "checkpoint": Path(self.path).name,
            "sha256": self.sha256,
            "base_checkpoint_sha256": self.base_sha256,
            "head_kind": self.kind,
            "classes": list(self.classes),
            "train_folds": m.get("train_folds"),
            "selection_fold": m.get("selection_fold"),
            "n_train_units": m.get("n_train_units"),
            "task": "turn-level presence of the class anywhere in the unit (max over windows)",
            "label_source": m.get("label_source"),
            "domain": "DIHANA: telephone, peninsular Spanish, train-information dialogues",
            "scores_are_calibrated_probabilities": False,
        }


def load_head(
    path, *, expected_sha256: str, base_sha256: str | None = None, name: str = "custom"
) -> LoadedHead:
    """Load an adapted head. The file hash MUST be given and match; ``base_sha256`` (the checkpoint the
    engine runs on) must equal the base the head was trained on."""
    if not expected_sha256:
        raise ValueError("expected_sha256 is required: unpinned head checkpoints are not loaded")
    ck, digest = load_adapted(path, expected_sha256=expected_sha256)
    if base_sha256 is not None and ck["base_ckpt_sha256"] != base_sha256:
        raise ValueError(
            "this head was trained on a different base checkpoint "
            f"({ck['base_ckpt_sha256'][:12]}... vs {base_sha256[:12]}...)"
        )
    meta = ck.get("meta", {})
    missing = [c for c in ck["classes"] if c not in meta.get("thresholds", {})]
    if missing:
        raise ValueError(f"head checkpoint has no frozen threshold for {missing}")
    return LoadedHead(
        name=name,
        path=str(path),
        sha256=digest,
        base_sha256=ck["base_ckpt_sha256"],
        classes=tuple(ck["classes"]),
        kind=ck["head_kind"],
        hidden=int(ck["hidden"]),
        state=ck["state"],
        thresholds=meta["thresholds"],
        meta=meta,
    )


def load_vc(base_sha256: str | None = None, path=None) -> LoadedHead:
    """Load the frozen Vc head. The sha256 is pinned in this module; ``path`` only relocates the file."""
    p = Path(path) if path else Path(__file__).resolve().parents[1] / VC_RELPATH
    if not p.exists():
        raise FileNotFoundError(
            f"Vc checkpoint not found at {p}. It lives in the repository "
            f"({VC_RELPATH}); pass head_checkpoint=<path> to relocate it (the pinned sha256 is still enforced)."
        )
    return load_head(p, expected_sha256=VC_SHA256, base_sha256=base_sha256, name="vc")
