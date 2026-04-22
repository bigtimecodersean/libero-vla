"""Stage 3: SFT training loop for LIBERO-VLA.

Trains LoRA adapters + vision projector + action head on LIBERO demos with a
flow-matching objective. Frozen pieces (SigLIP, Qwen2 base) never get gradients.

Usage:
    python -m src.training.train --config config.yaml
    python -m src.training.train --config config.yaml --wandb
    python -m src.training.train --config config.yaml --resume checkpoints/last.pt
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.libero_dataset import (  # noqa: E402
    LiberoCollator,
    LiberoDataset,
    build_joint_datasets,
)
from src.model.vla import LIBEROVLA, VLAConfig  # noqa: E402


# ---------------------------------------------------------------------------
# Optimizer / schedule helpers
# ---------------------------------------------------------------------------

def trainable_param_groups(model: nn.Module, weight_decay: float) -> list[dict]:
    """Split trainable params into decay / no-decay groups.

    Everything with ndim >= 2 (linear/conv weights) gets weight decay; biases
    and 1D tensors (LayerNorm, embeddings, learnable queries) don't.
    """
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim >= 2:
            decay.append(p)
        else:
            no_decay.append(p)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def cosine_with_warmup(step: int, warmup: int, total: int) -> float:
    if step < warmup:
        return step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def trainable_state_dict(model: LIBEROVLA) -> dict[str, torch.Tensor]:
    """Only the parts that actually receive gradients."""
    return {k: v.detach().cpu() for k, v in model.state_dict().items()
            if any(name == k for name, p in model.named_parameters() if p.requires_grad)}


def save_checkpoint(
    path: Path,
    model: LIBEROVLA,
    optimizer: torch.optim.Optimizer,
    step: int,
    cfg: VLAConfig,
    action_stats: dict[str, list[float]],
    train_cfg: dict,
    proprio_stats: dict[str, list[float]] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    blob = {
        "step": step,
        "model_trainable": trainable_state_dict(model),
        "optimizer": optimizer.state_dict(),
        "vla_config": asdict(cfg),
        "action_stats": action_stats,
        "proprio_stats": proprio_stats,
        "train_config": train_cfg,
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(blob, tmp)
    tmp.replace(path)


def load_checkpoint(
    path: Path,
    model: LIBEROVLA,
    optimizer: torch.optim.Optimizer | None = None,
) -> int:
    blob = torch.load(path, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(blob["model_trainable"], strict=False)
    # It's expected that most keys are missing (frozen base weights live in the
    # from_pretrained load, not the checkpoint). Warn if any were unexpected.
    if unexpected:
        print(f"[ckpt] unexpected keys: {unexpected[:5]}...")
    if optimizer is not None and "optimizer" in blob:
        optimizer.load_state_dict(blob["optimizer"])
    return int(blob.get("step", 0))


# ---------------------------------------------------------------------------
# Eval
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    model: LIBEROVLA,
    loader: DataLoader,
    device: torch.device,
    max_batches: int | None = None,
) -> float:
    model.eval()
    total, n = 0.0, 0
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        batch = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
                 for k, v in batch.items()}
        out = model(
            pixel_values=batch["pixel_values"],
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            actions=batch["actions"],
            wrist_pixel_values=batch.get("wrist_pixel_values"),
            proprio=batch.get("proprio"),
        )
        total += out["loss"].item()
        n += 1
    model.train()
    return total / max(1, n)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--suite", nargs="+", default=["libero_spatial"])
    parser.add_argument("--output", default="checkpoints")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--wandb", action="store_true", help="Enable wandb logging")
    parser.add_argument("--wandb-project", default="libero-vla")
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Override config.yaml max_steps")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--eval-batches", type=int, default=20)
    parser.add_argument("--save-last-every", type=int, default=500)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg_dict = yaml.safe_load(f)
    tr = cfg_dict["training"]
    if args.max_steps is not None:
        tr["max_steps"] = args.max_steps

    torch.manual_seed(tr["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    amp_dtype = torch.bfloat16 if use_amp else torch.float32
    print(f"[setup] device={device} amp={use_amp} dtype={amp_dtype}")

    # --------- Data ---------
    data_root = cfg_dict["data"]["root"]
    chunk = cfg_dict["data"]["action_chunk"]
    image_size = cfg_dict["data"]["image_size"]
    max_text_len = cfg_dict["data"]["max_text_len"]
    use_wrist = bool(cfg_dict["data"].get("use_wrist", False))
    use_proprio = bool(cfg_dict["data"].get("use_proprio", False))
    augment = bool(cfg_dict["data"].get("augment", False))

    suites = args.suite
    print(f"[data] loading {suites} from {data_root} "
          f"(wrist={use_wrist} proprio={use_proprio} augment={augment})")
    t0 = time.time()

    if len(suites) == 1:
        train_ds = LiberoDataset(
            data_root=data_root,
            suite=suites[0],
            chunk_size=chunk,
            image_size=image_size,
            split="train",
            use_wrist=use_wrist,
            use_proprio=use_proprio,
            augment=augment,
        )
        val_ds = LiberoDataset(
            data_root=data_root,
            suite=suites[0],
            chunk_size=chunk,
            image_size=image_size,
            split="val",
            action_stats=train_ds.stats,
            proprio_stats=train_ds.proprio_stats_dict,
            use_wrist=use_wrist,
            use_proprio=use_proprio,
            augment=False,
        )
        action_stats = train_ds.stats
        proprio_stats = train_ds.proprio_stats_dict
    else:
        train_ds, val_ds, action_stats, proprio_stats = build_joint_datasets(
            data_root=data_root,
            suites=suites,
            chunk_size=chunk,
            image_size=image_size,
            use_wrist=use_wrist,
            use_proprio=use_proprio,
            augment=augment,
        )

    print(f"[data] train={len(train_ds)} val={len(val_ds)} ({time.time() - t0:.1f}s)")

    # --------- Model ---------
    cfg = VLAConfig.from_dict(cfg_dict)
    print("[model] building LIBEROVLA ...")
    t0 = time.time()
    model = LIBEROVLA(cfg).to(device)
    model.train()
    print(
        f"[model] built in {time.time() - t0:.1f}s, "
        f"trainable={model.num_trainable_params() / 1e6:.2f}M"
    )

    collator = LiberoCollator(tokenizer=model.tokenizer, max_text_len=max_text_len)
    train_loader = DataLoader(
        train_ds,
        batch_size=tr["batch_size"],
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collator,
        drop_last=True,
        pin_memory=use_amp,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=tr["batch_size"],
        shuffle=False,
        num_workers=max(1, args.num_workers // 2),
        collate_fn=collator,
        drop_last=False,
        pin_memory=use_amp,
        persistent_workers=args.num_workers > 0,
    )

    # --------- Optimizer ---------
    param_groups = trainable_param_groups(model, weight_decay=tr["weight_decay"])
    optimizer = torch.optim.AdamW(
        param_groups, lr=tr["lr"], betas=(0.9, 0.95), eps=1e-8
    )

    start_step = 0
    if args.resume:
        resume_path = Path(args.resume)
        print(f"[ckpt] resuming from {resume_path}")
        start_step = load_checkpoint(resume_path, model, optimizer)
        print(f"[ckpt] resumed at step {start_step}")

    # --------- wandb ---------
    wandb = None
    if args.wandb:
        import wandb as _wandb  # type: ignore
        wandb = _wandb
        suite_label = "+".join(suites) if len(suites) > 1 else suites[0]
        run_name = args.wandb_run_name or f"{suite_label}-{int(time.time())}"
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config={**cfg_dict, "suites": suites, "max_steps": tr["max_steps"]},
        )
        wandb.watch(model, log=None)  # we log grad_norm manually

    # --------- Train loop ---------
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    step = start_step
    max_steps = tr["max_steps"]
    warmup = tr["warmup_steps"]
    log_every = tr["log_every"]
    grad_clip = tr["grad_clip"]
    grad_accum = tr["grad_accum"]
    base_lr = tr["lr"]

    print(f"[train] starting from step {step} -> {max_steps}")
    t_log = time.time()
    loss_accum = 0.0
    samples_since_log = 0
    optimizer.zero_grad(set_to_none=True)

    train_iter = iter(train_loader)
    micro_step = 0

    while step < max_steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        batch = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
                 for k, v in batch.items()}

        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            out = model(
                pixel_values=batch["pixel_values"],
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                actions=batch["actions"],
                wrist_pixel_values=batch.get("wrist_pixel_values"),
                proprio=batch.get("proprio"),
            )
            loss = out["loss"] / grad_accum

        loss.backward()
        loss_accum += loss.item() * grad_accum
        samples_since_log += batch["actions"].size(0)
        micro_step += 1

        if micro_step % grad_accum != 0:
            continue

        # LR schedule on the optimizer step boundary
        lr_scale = cosine_with_warmup(step, warmup, max_steps)
        for pg in optimizer.param_groups:
            pg["lr"] = base_lr * lr_scale

        grad_norm = torch.nn.utils.clip_grad_norm_(
            (p for p in model.parameters() if p.requires_grad), grad_clip
        )
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        step += 1

        if step % log_every == 0:
            dt = time.time() - t_log
            samples_per_sec = samples_since_log / dt if dt > 0 else 0.0
            avg_loss = loss_accum / log_every
            print(
                f"[step {step:>6}/{max_steps}] "
                f"loss={avg_loss:.4f}  "
                f"lr={base_lr * lr_scale:.2e}  "
                f"grad={grad_norm:.2f}  "
                f"{samples_per_sec:.1f} sa/s"
            )
            if wandb is not None:
                wandb.log(
                    {
                        "train/loss": avg_loss,
                        "train/lr": base_lr * lr_scale,
                        "train/grad_norm": float(grad_norm),
                        "train/samples_per_sec": samples_per_sec,
                        "train/step": step,
                    },
                    step=step,
                )
            loss_accum = 0.0
            samples_since_log = 0
            t_log = time.time()

        if step % args.eval_every == 0 or step == max_steps:
            t0 = time.time()
            val_loss = evaluate(model, val_loader, device, max_batches=args.eval_batches)
            print(f"[eval  {step:>6}] val_loss={val_loss:.4f}  ({time.time() - t0:.1f}s)")
            if wandb is not None:
                wandb.log({"val/loss": val_loss, "val/step": step}, step=step)

        if step % args.save_last_every == 0 or step == max_steps:
            last_path = output_dir / "last.pt"
            save_checkpoint(
                last_path,
                model,
                optimizer,
                step,
                cfg,
                action_stats,
                tr,
                proprio_stats=proprio_stats,
            )
            print(f"[ckpt] saved {last_path}")

    final_path = output_dir / "final.pt"
    save_checkpoint(
        final_path,
        model,
        optimizer,
        step,
        cfg,
        action_stats,
        tr,
        proprio_stats=proprio_stats,
    )
    print(f"[done] final checkpoint: {final_path}")
    if wandb is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
