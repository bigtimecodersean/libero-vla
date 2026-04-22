"""Flow-matching action head.

Learnable action queries cross-attend to VLM hidden states and decode a chunk of
actions. Training uses linear interpolation flow matching:
    x_t    = (1 - t) * noise + t * target
    v_tgt  = target - noise
    loss   = MSE( v_pred(x_t, t, ctx), v_tgt )
Inference uses Euler integration from noise (t=0) to clean actions (t=1).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _sinusoidal_time_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """t: (B,) in [0,1]. Returns (B, dim)."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0)
        * torch.arange(half, device=t.device, dtype=torch.float32)
        / max(half - 1, 1)
    )
    args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb.to(t.dtype if t.is_floating_point() else torch.float32)


class FlowMatchingActionHead(nn.Module):
    def __init__(
        self,
        vlm_dim: int,
        action_dim: int = 7,
        num_queries: int = 10,
        hidden_dim: int = 512,
        num_layers: int = 4,
        num_heads: int = 8,
        num_flow_steps: int = 10,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.num_queries = num_queries
        self.hidden_dim = hidden_dim
        self.num_flow_steps = num_flow_steps

        self.query_embed = nn.Parameter(torch.randn(num_queries, hidden_dim) * 0.02)
        self.action_in = nn.Linear(action_dim, hidden_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.ctx_proj = nn.Linear(vlm_dim, hidden_dim)

        layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.0,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, action_dim)

    def _velocity(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        ctx: torch.Tensor,
        ctx_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """x_t: (B, K, A), t: (B,), ctx: (B, L, H_vlm)."""
        B, K, _ = x_t.shape
        h = self.action_in(x_t) + self.query_embed.unsqueeze(0)
        t_emb = self.time_mlp(_sinusoidal_time_embedding(t, self.hidden_dim))
        h = h + t_emb.unsqueeze(1)
        memory = self.ctx_proj(ctx)
        out = self.decoder(
            tgt=h,
            memory=memory,
            memory_key_padding_mask=ctx_key_padding_mask,
        )
        return self.out_proj(self.norm(out))

    def loss(
        self,
        actions: torch.Tensor,
        ctx: torch.Tensor,
        ctx_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """actions: (B, K, A) normalized to [-1, 1]. ctx: (B, L, H_vlm)."""
        B = actions.size(0)
        noise = torch.randn_like(actions)
        t = torch.rand(B, device=actions.device, dtype=actions.dtype)
        t_b = t.view(B, 1, 1)
        x_t = (1.0 - t_b) * noise + t_b * actions
        v_target = actions - noise
        v_pred = self._velocity(x_t, t, ctx, ctx_key_padding_mask)
        return F.mse_loss(v_pred, v_target)

    @torch.no_grad()
    def sample(
        self,
        ctx: torch.Tensor,
        ctx_key_padding_mask: torch.Tensor | None = None,
        num_steps: int | None = None,
    ) -> torch.Tensor:
        num_steps = num_steps or self.num_flow_steps
        B = ctx.size(0)
        device = ctx.device
        dtype = ctx.dtype if ctx.is_floating_point() else torch.float32
        x = torch.randn(B, self.num_queries, self.action_dim, device=device, dtype=dtype)
        dt = 1.0 / num_steps
        for i in range(num_steps):
            t = torch.full((B,), i * dt, device=device, dtype=dtype)
            v = self._velocity(x, t, ctx, ctx_key_padding_mask)
            x = x + v * dt
        return x
