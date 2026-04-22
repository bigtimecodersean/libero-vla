"""Data sanity test.

1. Build LiberoDataset on libero_spatial (assumes scripts/download_data.py has run).
2. Print dataset stats: num files, num samples, action ranges.
3. Build a DataLoader with the tokenizer collator and print batch shapes.
4. Run one forward pass through LIBEROVLA to confirm the data plugs in.

    python scripts/data_test.py
"""
from __future__ import annotations

import os
import sys
import time

import torch
import yaml
from torch.utils.data import DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.data.libero_dataset import LiberoCollator, LiberoDataset  # noqa: E402
from src.model.vla import LIBEROVLA, VLAConfig  # noqa: E402


def main() -> None:
    torch.manual_seed(0)

    with open(os.path.join(ROOT, "config.yaml"), "r") as f:
        cfg_dict = yaml.safe_load(f)
    cfg = VLAConfig.from_dict(cfg_dict)

    suite = "libero_spatial"
    data_root = cfg_dict["data"]["root"]
    chunk_size = cfg.chunk_size
    max_text_len = cfg_dict["data"]["max_text_len"]

    print(f"[data] root={data_root} suite={suite} K={chunk_size}")
    t0 = time.time()
    ds = LiberoDataset(
        data_root=data_root,
        suite=suite,
        chunk_size=chunk_size,
        image_size=cfg_dict["data"]["image_size"],
    )
    print(f"[data] built in {time.time() - t0:.1f}s")
    print(f"[data] files={len(ds.files)} samples={len(ds)}")
    print(f"[data] action_min={ds.action_min.round(3).tolist()}")
    print(f"[data] action_max={ds.action_max.round(3).tolist()}")
    for path, instr in zip(ds.files[:3], ds.instructions[:3]):
        print(f"[data] {os.path.basename(str(path))[:60]}...")
        print(f"         -> '{instr}'")

    # Inspect one raw sample
    s = ds[0]
    print(
        f"[sample] pixel={tuple(s['pixel_values'].shape)} "
        f"range=[{s['pixel_values'].min():.2f}, {s['pixel_values'].max():.2f}] "
        f"actions={tuple(s['actions'].shape)} "
        f"actions_range=[{s['actions'].min():.2f}, {s['actions'].max():.2f}]"
    )
    print(f"[sample] instruction='{s['instruction']}'")

    # Build a model just to borrow its tokenizer (cached after smoke_test run)
    print("[model] loading tokenizer via LIBEROVLA ...")
    t0 = time.time()
    model = LIBEROVLA(cfg)
    print(f"[model] built in {time.time() - t0:.1f}s")
    tokenizer = model.tokenizer

    collator = LiberoCollator(tokenizer=tokenizer, max_text_len=max_text_len)
    # Small batch for CPU sanity; training uses cfg_dict["training"]["batch_size"]
    loader = DataLoader(
        ds,
        batch_size=4,
        shuffle=True,
        num_workers=0,
        collate_fn=collator,
        drop_last=True,
    )
    print(f"[loader] batch_size={loader.batch_size} num_batches={len(loader)}")

    t0 = time.time()
    batch = next(iter(loader))
    load_time = time.time() - t0
    print(
        f"[batch] pixel_values={tuple(batch['pixel_values'].shape)} "
        f"input_ids={tuple(batch['input_ids'].shape)} "
        f"actions={tuple(batch['actions'].shape)}  ({load_time:.1f}s)"
    )
    print(f"[batch] first 3 instructions: {batch['instructions'][:3]}")

    # End-to-end forward through the VLA to prove the data plugs in
    model.train()
    t0 = time.time()
    out = model(
        pixel_values=batch["pixel_values"],
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        actions=batch["actions"],
    )
    print(
        f"[fwd] hidden={tuple(out['hidden'].shape)} "
        f"loss={out['loss'].item():.4f} ({time.time() - t0:.1f}s)"
    )
    print("[ok] data sanity passed")


if __name__ == "__main__":
    main()
