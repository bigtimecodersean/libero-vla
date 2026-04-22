"""Full LIBERO-VLA: SigLIP -> MLP -> Qwen2 (LoRA, prefix-LM) -> Flow matching.

Optionally consumes a wrist camera and a 5-D proprio vector in addition to the
agentview image. When enabled, the prefix sequence becomes:

    [proprio_token, agent_vision_tokens, wrist_vision_tokens, text_tokens]

and the prefix LM mask treats the whole non-text portion as bidirectional.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn

from .action_head import FlowMatchingActionHead
from .vision_encoder import VisionEncoder
from .vlm_backbone import VLMBackbone, build_prefix_lm_mask


@dataclass
class VLAConfig:
    vision_name: str = "google/siglip-base-patch16-224"
    vision_freeze: bool = True

    llm_name: str = "Qwen/Qwen2-0.5B"
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: tuple = ("q_proj", "k_proj", "v_proj", "o_proj")

    action_dim: int = 7
    chunk_size: int = 10
    num_flow_steps: int = 10
    action_hidden_dim: int = 512
    action_num_layers: int = 4
    action_num_heads: int = 8

    use_wrist: bool = False
    use_proprio: bool = False
    proprio_dim: int = 5

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> "VLAConfig":
        m = cfg["model"]
        d = cfg.get("data", {})
        return cls(
            vision_name=m["vision"]["name"],
            vision_freeze=m["vision"].get("freeze", True),
            llm_name=m["llm"]["name"],
            lora_r=m["llm"]["lora"]["r"],
            lora_alpha=m["llm"]["lora"]["alpha"],
            lora_dropout=m["llm"]["lora"]["dropout"],
            lora_target_modules=tuple(m["llm"]["lora"]["target_modules"]),
            action_dim=m["action_head"]["action_dim"],
            chunk_size=m["action_head"]["chunk_size"],
            num_flow_steps=m["action_head"]["num_flow_steps"],
            action_hidden_dim=m["action_head"]["hidden_dim"],
            action_num_layers=m["action_head"]["num_layers"],
            action_num_heads=m["action_head"]["num_heads"],
            use_wrist=bool(d.get("use_wrist", False)),
            use_proprio=bool(d.get("use_proprio", False)),
            proprio_dim=int(d.get("proprio_dim", 5)),
        )


class LIBEROVLA(nn.Module):
    def __init__(self, config: VLAConfig):
        super().__init__()
        self.config = config

        self.vlm = VLMBackbone(
            model_name=config.llm_name,
            lora_r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            lora_target_modules=config.lora_target_modules,
        )
        llm_dim = self.vlm.hidden_size

        self.vision = VisionEncoder(
            model_name=config.vision_name,
            out_dim=llm_dim,
            freeze=config.vision_freeze,
        )

        if config.use_proprio:
            self.proprio_mlp = nn.Sequential(
                nn.Linear(config.proprio_dim, llm_dim),
                nn.SiLU(),
                nn.Linear(llm_dim, llm_dim),
            )

        self.action_head = FlowMatchingActionHead(
            vlm_dim=llm_dim,
            action_dim=config.action_dim,
            num_queries=config.chunk_size,
            hidden_dim=config.action_hidden_dim,
            num_layers=config.action_num_layers,
            num_heads=config.action_num_heads,
            num_flow_steps=config.num_flow_steps,
        )

    @property
    def tokenizer(self):
        return self.vlm.tokenizer

    def encode(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        wrist_pixel_values: torch.Tensor | None = None,
        proprio: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run vision + text through the backbone.

        Returns:
            hidden: (B, L, H) with L = num_prefix + T
            key_padding_mask: (B, L) True = padding position
        """
        text_embeds = self.vlm.embed_text(input_ids)  # (B, T, H)
        llm_dtype = text_embeds.dtype

        prefix_parts: list[torch.Tensor] = []

        if self.config.use_proprio:
            assert proprio is not None, "use_proprio=True but proprio=None"
            # proprio_mlp is fp32; feed it fp32 input and cast the output to
            # match the LLM dtype. Under autocast this is all handled anyway.
            mlp_dtype = next(self.proprio_mlp.parameters()).dtype
            prop_tok = self.proprio_mlp(proprio.to(mlp_dtype))
            if prop_tok.dtype != llm_dtype:
                prop_tok = prop_tok.to(llm_dtype)
            prefix_parts.append(prop_tok.unsqueeze(1))  # (B, 1, H)

        agent_feats = self.vision(pixel_values)  # (B, V, H)
        if agent_feats.dtype != llm_dtype:
            agent_feats = agent_feats.to(llm_dtype)
        prefix_parts.append(agent_feats)

        if self.config.use_wrist:
            assert wrist_pixel_values is not None, (
                "use_wrist=True but wrist_pixel_values=None"
            )
            wrist_feats = self.vision(wrist_pixel_values)
            if wrist_feats.dtype != llm_dtype:
                wrist_feats = wrist_feats.to(llm_dtype)
            prefix_parts.append(wrist_feats)

        prefix = torch.cat(prefix_parts, dim=1)  # (B, P, H)
        seq = torch.cat([prefix, text_embeds], dim=1)

        num_prefix = prefix.size(1)
        mask4d = build_prefix_lm_mask(
            num_vision=num_prefix,
            text_attention_mask=attention_mask,
            dtype=seq.dtype,
        )
        hidden = self.vlm(inputs_embeds=seq, attention_mask_4d=mask4d)

        # Action head weights stay fp32; autocast will redo the bf16 cast under
        # `torch.autocast` during training. Off autocast (eg. smoke test),
        # explicit cast is required so the fp32 linear layers don't see bf16.
        action_head_dtype = next(self.action_head.parameters()).dtype
        if hidden.dtype != action_head_dtype:
            hidden = hidden.to(action_head_dtype)

        B = attention_mask.size(0)
        prefix_valid = torch.ones(
            (B, num_prefix), dtype=torch.bool, device=attention_mask.device
        )
        text_valid = attention_mask.bool()
        valid = torch.cat([prefix_valid, text_valid], dim=1)
        key_padding_mask = ~valid
        return hidden, key_padding_mask

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        actions: torch.Tensor | None = None,
        wrist_pixel_values: torch.Tensor | None = None,
        proprio: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        hidden, kpm = self.encode(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            wrist_pixel_values=wrist_pixel_values,
            proprio=proprio,
        )
        if actions is not None:
            loss = self.action_head.loss(actions, hidden, ctx_key_padding_mask=kpm)
            return {"loss": loss, "hidden": hidden}
        pred = self.action_head.sample(hidden, ctx_key_padding_mask=kpm)
        return {"actions": pred, "hidden": hidden}

    def num_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
