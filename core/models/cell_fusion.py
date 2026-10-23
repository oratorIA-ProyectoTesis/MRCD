"""Position/target/missingness-aware attention with independent type/context heads."""
import numpy as np
import torch
from torch import nn
from core.cell_contracts import CELL_MODES


def tensor_batch(data, device="cpu"):
    return {key: torch.as_tensor(value, device=device) for key, value in data.items() if isinstance(value, np.ndarray)}


class CellFusionV2(nn.Module):
    def __init__(self, mode="audio_text", d_model=32, live=False):
        super().__init__()
        if live:
            raise ValueError("live v2 unsupported: centered lookahead/tail flush not implemented")
        if mode not in CELL_MODES or type(d_model) is not int or d_model < 8 or d_model % 4:
            raise ValueError("invalid cell mode/d_model (multiple of four >=8 required)")
        self.mode, self.d_model = mode, d_model
        self.values = nn.ModuleDict({key: nn.Linear(2 * width, d_model) for key, width in
                                     (("ac", 5), ("ling", 12), ("kin", 12), ("pros", 8))})
        self.position = nn.Sequential(nn.Linear(1, d_model), nn.Tanh(), nn.Linear(d_model, d_model))
        self.modality = nn.Embedding(4, d_model)
        self.ownership = nn.Embedding(2, d_model)
        self.null = nn.Parameter(torch.zeros(1, 1, d_model))
        self.query = nn.Parameter(torch.zeros(1, 1, d_model))
        self.attention = nn.MultiheadAttention(d_model, 4, batch_first=True, dropout=0.)
        self.output = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.GELU())
        self.type_head = nn.Linear(d_model, 7)
        self.context_head = nn.Linear(d_model, 14)

    def forward(self, data):
        if len(data["ac"]) == 0:
            # D=0 is a valid temporal grid; do not enter attention's ambiguous
            # zero-batch reshape. Empty training is rejected by its own boundary.
            return dict(type_logits=self.type_head.weight.new_empty((0, 7)),
                        context_logits=self.context_head.weight.new_empty((0, 7, 2)),
                        input_evaluable=torch.empty(0, dtype=torch.bool, device=self.type_head.weight.device))
        tokens, valid, targets = [], [], []
        for modality, key in enumerate(("ac", "ling", "kin", "pros")):
            x, mask = data[key].float(), data[key + "_mask"].bool()
            if key == "ling" and self.mode == "audio_only" or key == "kin" and self.mode != "trimodal":
                mask = torch.zeros_like(mask)
            if key == "kin":
                mask = mask & data["has_video"][:, None, None]
            if key == "pros":
                x, mask = x[:, None], mask[:, None]
                position = torch.zeros(x.shape[:2], device=x.device)
                target = torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)
            else:
                pos_key, target_key = {"ac": ("ac_pos", "ac_target"), "ling": ("lpos", "ltarget"),
                                       "kin": ("kpos", "ktarget")}[key]
                position, target = data[pos_key].float(), data[target_key].bool()
            safe = torch.where(mask, x, torch.zeros_like(x))
            h = self.values[key](torch.cat([safe, mask.float()], -1))
            h = h + self.position(position[..., None]) + self.modality.weight[modality] + self.ownership(target.long())
            tokens.append(h)
            valid.append(mask.any(-1))
            targets.append(target & mask.any(-1))
        n = len(data["ac"])
        # The query explicitly summarizes owned acoustic observations, not a
        # whole-context pool; ownership also conditions every evidence token.
        own_ac = targets[0]
        pooled = (tokens[0] * own_ac[..., None]).sum(1) / own_ac.sum(1).clamp_min(1)[:, None]
        query = self.query.expand(n, -1, -1) + pooled[:, None]
        evidence = torch.cat(tokens, 1)
        available = torch.cat(valid, 1)
        null_valid = ~available.any(1)
        evidence = torch.cat([evidence, self.null.expand(n, -1, -1)], 1)
        available = torch.cat([available, null_valid[:, None]], 1)
        attended, _ = self.attention(query, evidence, evidence, key_padding_mask=~available, need_weights=False)
        representation = self.output((attended + query)[:, 0])
        evaluable = targets[0].any(1) | targets[1].any(1) | targets[2].any(1)
        return dict(type_logits=self.type_head(representation),
                    context_logits=self.context_head(representation).reshape(n, 7, 2), input_evaluable=evaluable)
