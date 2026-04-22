"""Qwen2 backbone with LoRA + prefix-LM attention mask over [vision, text]."""
from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, Qwen2Model


def build_prefix_lm_mask(
    num_vision: int,
    text_attention_mask: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build a 4D additive attention mask for a [vision | text] sequence.

    Rules:
      - vision <-> vision: bidirectional
      - text   -> vision: full access
      - text   -> text:   causal
      - vision -> text:   blocked
      - padded text keys are masked out everywhere

    Returns a mask of shape (B, 1, L, L) with 0 for "attend" and -inf for "mask".
    """
    B, T = text_attention_mask.shape
    L = num_vision + T
    device = text_attention_mask.device
    neg_inf = torch.finfo(dtype).min

    mask = torch.full((B, L, L), neg_inf, device=device, dtype=dtype)

    # vision <-> vision bidirectional
    mask[:, :num_vision, :num_vision] = 0.0

    # text -> vision full
    mask[:, num_vision:, :num_vision] = 0.0

    # text -> text causal
    causal = torch.zeros((T, T), device=device, dtype=dtype)
    causal.masked_fill_(
        torch.triu(torch.ones((T, T), device=device, dtype=torch.bool), diagonal=1),
        neg_inf,
    )
    mask[:, num_vision:, num_vision:] = causal

    # pad keys blocked for all queries
    pad_keys = (text_attention_mask == 0).unsqueeze(1)  # (B, 1, T)
    mask[:, :, num_vision:] = mask[:, :, num_vision:].masked_fill(pad_keys, neg_inf)

    return mask.unsqueeze(1)  # (B, 1, L, L)


class VLMBackbone(nn.Module):
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2-0.5B",
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        lora_target_modules: Iterable[str] = ("q_proj", "k_proj", "v_proj", "o_proj"),
    ):
        super().__init__()
        # bf16 on CUDA (fast, standard for Qwen2 training); fp32 on CPU
        # (autocast doesn't support some ops on CPU, and the smoke test uses CPU).
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        base = Qwen2Model.from_pretrained(
            model_name,
            attn_implementation="eager",
            torch_dtype=dtype,
        )
        # Freeze base; LoRA adapters will be added on top
        for p in base.parameters():
            p.requires_grad = False

        lora_cfg = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=list(lora_target_modules),
            bias="none",
            task_type=None,
        )
        self.llm = get_peft_model(base, lora_cfg)
        self.hidden_size = base.config.hidden_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def embed_text(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.llm.get_input_embeddings()(input_ids)

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask_4d: torch.Tensor,
    ) -> torch.Tensor:
        """Run the backbone on precomputed embeddings with a custom 4D mask.

        inputs_embeds:        (B, L, H)
        attention_mask_4d:    (B, 1, L, L) additive mask
        Returns:              (B, L, H)
        """
        out = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask_4d,
            use_cache=False,
            output_hidden_states=False,
            return_dict=True,
        )
        return out.last_hidden_state
