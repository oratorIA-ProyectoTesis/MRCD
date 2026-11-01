"""Spike de fusión multimodal por atención cruzada.

Q = H_acoust · W_Q ;  K, V = [H_ling ; H_kine] · W_K, W_V
CrossAttn = softmax(Q·Kᵀ / √d) · V   ->  pooling  ->  clasificador lineal

Modos de ablación:
  A  audio_only : Q=K=V=H_acoust (auto-atención; misma capacidad de cómputo)
  B  audio_text : K,V = [H_ling]
  C  trimodal   : K,V = [H_ling ; H_kine]  (+ modality dropout cinésico en entrenamiento)

Se añade un token "nulo" aprendido a K/V para que la atención esté definida
cuando una modalidad falta (sin video o ventana sin palabras).
"""

from __future__ import annotations


import torch
import torch.nn as nn

from mrcd.constants import CONTEXT_S, KINESIC_DIM, KINESIC_STEPS, PROSODY_DIM, VIDEO_FPS

MODES = ("audio_only", "audio_text", "trimodal", "kinesic_only")

# Modos que NO reciben la rama prosódica. `kinesic_only` existe para responder una
# pregunta que la ablación sola no puede: si el trimodal no supera al bimodal, ¿es
# porque el video no tiene información, o porque la fusión la ignora? (Wang, Tran y
# Feiszli, CVPR 2020). Un modo "solo video" que recibiera F0 no respondería nada.
NO_PROSODY_MODES = ("kinesic_only",)
CONTEXT_STEPS = int(round(CONTEXT_S * VIDEO_FPS))


class AcousticEncoder(nn.Module):
    """(B, 300, 5) @100 fps  ->  (B, 30, d) @10 fps."""

    def __init__(self, in_dim: int, d: int, steps: int = KINESIC_STEPS):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_dim, d, 5, padding=2),
            nn.GELU(),
            nn.Conv1d(d, d, 5, padding=2, dilation=1),
            nn.GELU(),
        )
        self.pool = nn.AdaptiveAvgPool1d(steps)
        self.pos = nn.Parameter(torch.randn(1, steps, d) * 0.02)

    def forward(self, x):
        h = self.pool(self.net(x.transpose(1, 2))).transpose(1, 2)
        return h + self.pos


class KinesicTCN(nn.Module):
    """TCN dilatada sobre (B, 30, 12) + canal de máscara -> (B, 30, d)."""

    def __init__(self, d: int, in_dim: int = KINESIC_DIM, levels: int = 3):
        super().__init__()
        layers, c = [], in_dim + 1
        for i in range(levels):
            dil = 2**i
            layers += [nn.Conv1d(c, d, 3, padding=dil, dilation=dil), nn.GELU()]
            c = d
        self.net = nn.Sequential(*layers)
        self.pos = nn.Parameter(torch.randn(1, KINESIC_STEPS, d) * 0.02)

    def forward(self, x, mask):
        x = torch.cat([x * mask.unsqueeze(-1), mask.unsqueeze(-1).float()], -1)
        return self.net(x.transpose(1, 2)).transpose(1, 2) + self.pos


class LinguisticEncoder(nn.Module):
    """Rasgos por palabra (+ posición temporal en la ventana) -> (B, L, d)."""

    def __init__(self, in_dim: int, d: int, n_pos: int = CONTEXT_STEPS):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(in_dim, d), nn.GELU(), nn.Linear(d, d))
        self.pos = nn.Embedding(n_pos, d)

    def forward(self, x, pos_idx):
        return self.proj(x) + self.pos(pos_idx.clamp(0, self.pos.num_embeddings - 1))


class CrossModalFusion(nn.Module):
    def __init__(
        self,
        n_classes: int,
        ac_dim: int = 5,
        ling_dim: int = 12,
        d: int = 64,
        heads: int = 4,
        mode: str = "trimodal",
        modality_dropout: float = 0.5,
        dropout: float = 0.1,
        pros_dim: int = PROSODY_DIM,
    ):
        super().__init__()
        assert mode in MODES
        if mode in NO_PROSODY_MODES:
            pros_dim = 0
        self.mode, self.p_drop = mode, modality_dropout
        self.ac = AcousticEncoder(ac_dim, d) if mode != "kinesic_only" else None
        self.ling = (
            LinguisticEncoder(ling_dim, d)
            if mode not in ("audio_only", "kinesic_only")
            else None
        )
        self.kin = KinesicTCN(d) if mode in ("trimodal", "kinesic_only") else None
        # Rama prosódica: presente en LOS TRES modos. Se deriva sólo del audio
        # (F0, energía, VAD), así que atribuirla a la multimodalidad inflaría
        # artificialmente la ventaja de los modos con texto o video.
        # Sin LayerNorm de entrada: normalizar entre rasgos heterogéneos
        # (segundos, semitonos, pendientes) mezcla unidades distintas y borra
        # la escala absoluta —justo lo que `f0_reset` aporta—. La z-normalización
        # por columna ya se hace con estadísticas del conjunto de entrenamiento.
        self.pros = (
            nn.Sequential(nn.Linear(pros_dim, d), nn.GELU(), nn.Linear(d, d))
            if pros_dim
            else None
        )
        self.null = nn.Parameter(torch.zeros(1, 1, d))
        self.type_emb = nn.Embedding(4, d)  # 0=nulo, 1=ling, 2=kine, 3=prosodia
        self.attn = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.norm1, self.norm2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.ffn = nn.Sequential(
            nn.Linear(d, 2 * d), nn.GELU(), nn.Dropout(dropout), nn.Linear(2 * d, d)
        )
        head_in = d * (2 if self.pros is not None else 1)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(head_in, n_classes))
        self.d = d

    def forward(self, batch: dict, return_attn: bool = False):
        if self.mode == "kinesic_only":
            # Auto-atención sobre la rama cinésica: mismo presupuesto de cómputo que
            # `audio_only`, pero con la cara como única fuente. Sin prosodia ni texto.
            q = self.kin(batch["kin"], batch["kmask"].bool())
            att, w = self.attn(
                q, q, q, need_weights=return_attn, average_attn_weights=True
            )
            h = self.norm1(q + att)
            h = self.norm2(h + self.ffn(h))
            logits = self.head(h.mean(1))
            if not return_attn:
                return logits
            types = torch.full((q.shape[1],), 2, dtype=torch.long)
            return logits, self._modality_weights(w, types)
        q = self.ac(batch["ac"])  # (B,30,d)
        B = q.shape[0]
        hp = None
        if self.pros is not None:
            pv = batch.get("pros")
            if pv is None:
                pv = torch.zeros(B, self.pros[0].in_features, device=q.device)
            hp = self.pros(pv.float())  # (B,d)
        if self.mode == "audio_only":
            parts = [q]
            pads = [torch.zeros(B, q.shape[1], dtype=torch.bool, device=q.device)]
            types = [torch.zeros(q.shape[1], dtype=torch.long)]
            if hp is not None:
                parts.append(hp.unsqueeze(1) + self.type_emb.weight[3])
                pads.append(torch.zeros(B, 1, dtype=torch.bool, device=q.device))
                types.append(torch.full((1,), 3, dtype=torch.long))
            kv, pad, types = torch.cat(parts, 1), torch.cat(pads, 1), torch.cat(types)
        else:
            parts = [self.null.expand(B, -1, -1) + self.type_emb.weight[0]]
            pads = [torch.zeros(B, 1, dtype=torch.bool, device=q.device)]
            types = [torch.zeros(1, dtype=torch.long)]
            if hp is not None:
                parts.append(hp.unsqueeze(1) + self.type_emb.weight[3])
                pads.append(torch.zeros(B, 1, dtype=torch.bool, device=q.device))
                types.append(torch.full((1,), 3, dtype=torch.long))
            hl = self.ling(batch["ling"], batch["lpos"]) + self.type_emb.weight[1]
            parts.append(hl)
            pads.append(~batch["lmask"])
            types.append(torch.ones(hl.shape[1], dtype=torch.long))
            if self.mode == "trimodal":
                kmask = batch["kmask"].clone()
                if self.training and self.p_drop > 0:  # modality dropout
                    drop = torch.rand(B, device=q.device) < self.p_drop
                    kmask[drop] = False
                hk = self.kin(batch["kin"], kmask) + self.type_emb.weight[2]
                parts.append(hk)
                pads.append(~kmask)
                types.append(torch.full((hk.shape[1],), 2))
            kv, pad, types = torch.cat(parts, 1), torch.cat(pads, 1), torch.cat(types)
        att, w = self.attn(
            q,
            kv,
            kv,
            key_padding_mask=pad,
            need_weights=return_attn,
            average_attn_weights=True,
        )
        h = self.norm1(q + att)
        h = self.norm2(h + self.ffn(h))
        pooled = h.mean(1)
        if hp is not None:  # vía directa al clasificador: la frontera prosódica
            pooled = torch.cat(
                [pooled, hp], -1
            )  # no debe depender de ganar la atención
        logits = self.head(pooled)
        if not return_attn:
            return logits
        return logits, self._modality_weights(w, types)

    @staticmethod
    def _modality_weights(w, types):
        """Aproximación de `modality_weights`: masa de atención por tipo de token.
        La masa sobre el token nulo se interpreta como dependencia de la propia señal acústica."""
        names = ((0, "acoustic"), (1, "syntactic"), (2, "kinesic"), (3, "prosodic"))
        if types is None:
            z = torch.zeros(w.shape[0])
            return {
                "acoustic": torch.ones(w.shape[0]),
                "syntactic": z,
                "kinesic": z,
                "prosodic": z,
            }
        mass = w.mean(1)  # (B, K)
        types = types.to(mass.device)
        return {
            name: mass[:, types == t].sum(1)
            if (types == t).any()
            else torch.zeros(mass.shape[0], device=mass.device)
            for t, name in names
        }


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
