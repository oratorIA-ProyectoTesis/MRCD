"""Three-mode multimodal classifier for the versioned MRCD feature tensors.

This is a new implementation of the public API exercised by the scripts and tests;
the original fusion source was absent from the repository. Historical checkpoints
must not be assumed compatible with its state dictionary.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from core.constants import PROSODY_DIM
from core.contracts import MODES

MODALITIES = ("acoustic", "syntactic", "kinesic", "prosodic")


class FeatureBranch(nn.Module):
    def __init__(self, input_dim: int, d: int):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(input_dim, d), nn.LayerNorm(d), nn.GELU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.to(x.dtype).unsqueeze(-1)
    return (x * weights).sum(1) / weights.sum(1).clamp_min(1)


class CrossModalFusion(nn.Module):
    """Acoustic queries attend to timestamped text and visual tokens.

    Missing modalities have zero gate weight. Prosody is an acoustic-derived
    branch present in all three ablation modes, including ``audio_only``.
    """

    def __init__(self, n_cls: int, ac_dim: int = 5, ling_dim: int = 12,
                 d: int = 64, mode: str = "trimodal", modality_dropout: float = 0.0,
                 pros_dim: int = PROSODY_DIM):
        super().__init__()
        if mode not in MODES:
            raise ValueError(f"unknown MRCD mode {mode!r}; expected {MODES}")
        if d % 4:
            raise ValueError("embedding width must be divisible by four attention heads")
        if not 0 <= modality_dropout <= 1:
            raise ValueError("modality_dropout must be in [0, 1]")
        self.mode, self.modality_dropout = mode, modality_dropout
        self.ac = FeatureBranch(ac_dim, d)
        self.ling = FeatureBranch(ling_dim, d)
        self.kin = FeatureBranch(12, d)
        self.pros = FeatureBranch(pros_dim, d) if pros_dim else None
        self.word_position = nn.Embedding(100, d)
        self.visual_position = nn.Embedding(30, d)
        self.cross = nn.MultiheadAttention(d, 4, batch_first=True)
        self.null_token = nn.Parameter(torch.zeros(1, 1, d))
        self.gate = nn.Linear(d, 1)
        self.classifier = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, n_cls))

    def forward(self, batch: dict[str, torch.Tensor], return_attn: bool = False):
        audio = batch["ac"]
        if audio.ndim != 3:
            raise ValueError("ac must have shape [batch, time, features]")
        n = audio.shape[0]
        # Ten acoustic frames per visual frame, while retaining fine-grained
        # acoustic information in the pooled channel representation.
        ac_seq = self.ac(F.avg_pool1d(audio.transpose(1, 2), kernel_size=10, stride=10).transpose(1, 2))
        ac_vec = ac_seq.mean(1)
        tokens, token_masks = [], []
        reps = [ac_vec]
        present = [torch.ones(n, dtype=torch.bool, device=audio.device)]

        if self.mode != "audio_only" and "ling" in batch:
            lmask = batch.get("lmask", torch.ones(batch["ling"].shape[:2], dtype=torch.bool, device=audio.device)).bool()
            lpos = batch.get("lpos", torch.zeros_like(lmask, dtype=torch.long)).long()
            ling = self.ling(batch["ling"]) + self.word_position(lpos.clamp(0, 99))
            l_present = lmask.any(1)
            if self.training and self.modality_dropout:
                l_present = l_present & (torch.rand(n, device=audio.device) >= self.modality_dropout)
            lmask = lmask & l_present[:, None]
            reps.append(_masked_mean(ling, lmask))
            present.append(l_present)
            tokens.append(ling); token_masks.append(lmask)
        else:
            reps.append(torch.zeros_like(ac_vec)); present.append(torch.zeros_like(present[0]))

        if self.mode == "trimodal" and "kin" in batch:
            if "has_video" not in batch:
                raise ValueError("trimodal forward requires has_video from real-frame coverage; "
                                 "kmask alone may contain interpolated frames")
            kmask = batch.get("kmask", torch.ones(batch["kin"].shape[:2], dtype=torch.bool, device=audio.device)).bool()
            pos = torch.arange(batch["kin"].shape[1], device=audio.device).clamp_max(29)
            kin = self.kin(batch["kin"]) + self.visual_position(pos)[None]
            # Interpolation can make kmask nonempty from a sparse recording.
            # The extractor's >=24-real-frame availability flag is authoritative.
            has_video = batch["has_video"].bool()
            k_present = kmask.any(1) & has_video
            if self.training and self.modality_dropout:
                k_present = k_present & (torch.rand(n, device=audio.device) >= self.modality_dropout)
            kmask = kmask & k_present[:, None]
            reps.append(_masked_mean(kin, kmask))
            present.append(k_present)
            tokens.append(kin); token_masks.append(kmask)
        else:
            reps.append(torch.zeros_like(ac_vec)); present.append(torch.zeros_like(present[0]))

        if self.pros is not None and "pros" in batch:
            reps.append(self.pros(batch["pros"]))
            present.append(torch.ones_like(present[0]))
        else:
            reps.append(torch.zeros_like(ac_vec)); present.append(torch.zeros_like(present[0]))

        rep = torch.stack(reps, 1)
        available = torch.stack(present, 1)
        scores = self.gate(rep).squeeze(-1).masked_fill(~available, torch.finfo(rep.dtype).min)
        weights = torch.softmax(scores, dim=1)
        fused = (rep * weights.unsqueeze(-1)).sum(1)

        if tokens:
            memory = torch.cat([*tokens, self.null_token.expand(n, -1, -1)], 1)
            valid = torch.cat([*token_masks, torch.ones(n, 1, dtype=torch.bool, device=audio.device)], 1)
            attended, _ = self.cross(ac_seq, memory, memory, key_padding_mask=~valid, need_weights=False)
            any_auxiliary = torch.stack(present[1:3], 1).any(1)
            fused = fused + attended.mean(1) * any_auxiliary[:, None]

        logits = self.classifier(fused)
        if return_attn:
            return logits, {name: weights[:, i] for i, name in enumerate(MODALITIES)}
        return logits


def count_params(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)
