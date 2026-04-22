"""SigLIP vision encoder + MLP projector into the LLM hidden dim."""
from __future__ import annotations

import torch
import torch.nn as nn
from transformers import SiglipVisionModel


class VisionEncoder(nn.Module):
    def __init__(
        self,
        model_name: str = "google/siglip-base-patch16-224",
        out_dim: int = 896,
        freeze: bool = True,
    ):
        super().__init__()
        self.vision = SiglipVisionModel.from_pretrained(model_name)
        in_dim = self.vision.config.hidden_size
        self.projector = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )
        self.frozen = freeze
        if freeze:
            for p in self.vision.parameters():
                p.requires_grad = False
            self.vision.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.frozen:
            self.vision.eval()
        return self

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """pixel_values: (B, 3, H, W). Returns (B, num_patches, out_dim)."""
        ctx = torch.no_grad() if self.frozen else torch.enable_grad()
        with ctx:
            out = self.vision(pixel_values=pixel_values)
        feats = out.last_hidden_state  # (B, 196, 768)
        return self.projector(feats)
