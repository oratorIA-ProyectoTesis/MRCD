"""Reusable checkpoint-backed prediction use case for batch and live adapters."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from core.contracts import BATCH_FEATURES, NUMERIC_FEATURES, validate_checkpoint, validate_features
from core.models.fusion import CrossModalFusion
from core.normalization import apply_stats


def load_ckpt(path: Path, device: str):
    """Load a checkpoint envelope, rejecting incompatible versioned contracts."""
    ck = torch.load(path, map_location=device, weights_only=True)
    status = validate_checkpoint(ck)
    dims = ck["dims"]
    net = CrossModalFusion(len(ck["labels"]), ac_dim=int(dims["ac"]), ling_dim=int(dims["ling"]),
                           d=int(dims["d"]), mode=ck["mode"], pros_dim=int(dims.get("pros", 0))).to(device)
    try:
        net.load_state_dict(ck["state_dict"])
    except RuntimeError as exc:
        raise ValueError("checkpoint state_dict is incompatible with this reconstructed fusion model; "
                         "retrain with scripts/train_checkpoints.py") from exc
    net.eval()
    norm = {key: (np.asarray(mu, np.float32), np.asarray(sd, np.float32))
            for key, (mu, sd) in ck["norm"].items()}
    provenance = dict(ck.get("provenance", {}), **status)
    return net, norm, ck["labels"], provenance


@torch.no_grad()
def predict(net: CrossModalFusion, norm: dict, data: dict, device: str, bs: int = 128) -> np.ndarray:
    dims = {"ac": net.ac.proj[0].in_features, "ling": net.ling.proj[0].in_features,
            "pros": net.pros.proj[0].in_features if net.pros is not None else 0}
    validate_features(data, expected_dims=dims, mode=net.mode)
    normalized = apply_stats(data, norm)
    n = len(data["ac"])
    if not n:
        return np.zeros((0, net.classifier[-1].out_features), np.float32)
    probabilities = []
    for start in range(0, n, bs):
        sl = slice(start, min(start + bs, n))
        batch = {}
        for key in BATCH_FEATURES:
            if key not in normalized:
                continue
            value = torch.as_tensor(normalized[key][sl], device=device)
            batch[key] = (value.bool() if key in ("lmask", "kmask", "has_video") else
                          value.float() if key in NUMERIC_FEATURES else value.long())
        probabilities.append(torch.softmax(net(batch), -1).cpu().numpy())
    return np.concatenate(probabilities, 0)


class InferenceEngine:
    """One prediction path; input adapters only produce the feature tensors."""

    def __init__(self, net: CrossModalFusion, norm: dict, labels: list[str],
                 provenance: dict, device: str):
        self.net, self.norm, self.labels = net, norm, labels
        self.provenance, self.device = provenance, device

    @classmethod
    def from_checkpoint(cls, path: Path, device: str) -> "InferenceEngine":
        return cls(*load_ckpt(path, device), device)

    def predict(self, data: dict, bs: int = 128) -> np.ndarray:
        return predict(self.net, self.norm, data, self.device, bs)
