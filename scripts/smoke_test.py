"""Smoke test: build LIBEROVLA, run forward + backward on a synthetic batch,
print shapes, loss, parameter counts, and a sampled action chunk.

Run:
    python scripts/smoke_test.py
"""
from __future__ import annotations

import os
import sys
import time

import torch
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.model.vla import LIBEROVLA, VLAConfig  # noqa: E402


def main() -> None:
    torch.manual_seed(0)

    with open(os.path.join(ROOT, "config.yaml"), "r") as f:
        cfg_dict = yaml.safe_load(f)
    cfg = VLAConfig.from_dict(cfg_dict)
    print(f"[cfg] llm={cfg.llm_name}  vision={cfg.vision_name}  K={cfg.chunk_size}")

    device = torch.device("cpu")
    print(f"[device] {device}")

    t0 = time.time()
    model = LIBEROVLA(cfg).to(device)
    model.train()
    print(f"[build] model built in {time.time() - t0:.1f}s")

    total = sum(p.numel() for p in model.parameters())
    trainable = model.num_trainable_params()
    print(f"[params] total={total/1e6:.1f}M  trainable={trainable/1e6:.2f}M")

    # Synthetic batch
    B = 2
    pixel_values = torch.randn(B, 3, 224, 224, device=device)
    instructions = [
        "pick up the red block and place it on the plate",
        "open the top drawer",
    ]
    tok = model.tokenizer(
        instructions,
        padding="max_length",
        truncation=True,
        max_length=cfg_dict["data"]["max_text_len"],
        return_tensors="pt",
    )
    input_ids = tok.input_ids.to(device)
    attention_mask = tok.attention_mask.to(device)
    actions = torch.rand(B, cfg.chunk_size, cfg.action_dim, device=device) * 2 - 1

    print(
        f"[batch] pixel={tuple(pixel_values.shape)} "
        f"input_ids={tuple(input_ids.shape)} actions={tuple(actions.shape)}"
    )

    # Forward
    t0 = time.time()
    out = model(
        pixel_values=pixel_values,
        input_ids=input_ids,
        attention_mask=attention_mask,
        actions=actions,
    )
    fwd_time = time.time() - t0
    loss = out["loss"]
    hidden = out["hidden"]
    print(f"[fwd] hidden={tuple(hidden.shape)} loss={loss.item():.4f} ({fwd_time:.1f}s)")

    # Backward
    t0 = time.time()
    loss.backward()
    bwd_time = time.time() - t0

    grad_norm_sq = 0.0
    n_with_grad = 0
    for p in model.parameters():
        if p.grad is not None:
            grad_norm_sq += p.grad.detach().pow(2).sum().item()
            n_with_grad += 1
    grad_norm = grad_norm_sq ** 0.5
    print(
        f"[bwd] grad_norm={grad_norm:.4f} "
        f"tensors_with_grad={n_with_grad} ({bwd_time:.1f}s)"
    )

    # Sample at inference
    model.eval()
    with torch.no_grad():
        sampled = model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
    print(
        f"[sample] actions={tuple(sampled['actions'].shape)} "
        f"range=[{sampled['actions'].min().item():.3f}, {sampled['actions'].max().item():.3f}]"
    )
    print("[ok] smoke test passed")


if __name__ == "__main__":
    main()
