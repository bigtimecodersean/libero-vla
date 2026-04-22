"""LIBERO HDF5 dataset and collator.

Each sample is a (image_t, [wrist_t], [proprio_t], instruction, actions_t:t+K)
tuple:
    pixel_values        : (3, 224, 224) float32, SigLIP-normalized to [-1, 1]
    wrist_pixel_values  : (3, 224, 224) float32, SigLIP-normalized to [-1, 1]
    proprio             : (proprio_dim,) float32, per-dim normalized to [-1, 1]
    actions             : (K, 7)         float32, per-dim normalized to [-1, 1]
    instruction         : str            (derived from filename; LIBERO encodes
                                          the task in the filename)

HDF5 layout (LIBERO/robosuite convention):
    file["data/demo_{i}/obs/agentview_rgb"]  (T, 128, 128, 3) uint8, vflipped
    file["data/demo_{i}/obs/eye_in_hand_rgb"] (T, 128, 128, 3) uint8, vflipped
    file["data/demo_{i}/obs/ee_pos"]          (T, 3)  float64
    file["data/demo_{i}/obs/gripper_states"]  (T, 2)  float64
    file["data/demo_{i}/actions"]             (T, 7)  float32

Action + proprio stats (per-dim 1st/99th percentile) are computed once and
cached to data/libero/<suite>/{action_stats,proprio_stats}.json.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, Dataset
from torchvision import transforms as T


# SigLIP was pretrained with mean=0.5, std=0.5, i.e. pixel values in [-1, 1].
SIGLIP_MEAN = 0.5
SIGLIP_STD = 0.5

# Proprio = [ee_pos (3), gripper_states (2)]. Orientation is dropped for Run A
# because the HDF5 encodes it as 3D (euler) while the sim env returns 4D quat,
# so aligning the two cleanly would add work without much payoff.
PROPRIO_DIM = 5


def instruction_from_filename(path: str) -> str:
    """Turn a LIBERO HDF5 filename into the natural-language instruction.

    Examples:
        pick_up_the_black_bowl_on_the_plate_demo.hdf5
            -> "pick up the black bowl on the plate"
        KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_demo.hdf5
            -> "turn on the stove and put the moka pot on it"
    """
    name = os.path.basename(path)
    if name.endswith("_demo.hdf5"):
        name = name[: -len("_demo.hdf5")]
    elif name.endswith(".hdf5"):
        name = name[: -len(".hdf5")]
    parts = name.split("_")
    i = 0
    while i < len(parts) and parts[i] and parts[i].isupper():
        i += 1
    instruction = " ".join(parts[i:]) if i < len(parts) else " ".join(parts)
    return instruction.replace("_", " ").strip().lower()


def build_augmentation() -> T.Compose:
    return T.Compose([
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
        T.RandomResizedCrop(224, scale=(0.85, 1.0), ratio=(0.95, 1.05)),
        T.RandomErasing(p=0.1, scale=(0.02, 0.10)),
    ])


def preprocess_image(img_hwc_uint8: np.ndarray, image_size: int = 224) -> torch.Tensor:
    """(H, W, 3) uint8 -> (3, image_size, image_size) float32 SigLIP-normalized.

    LIBERO renders with MuJoCo, which outputs a vertically flipped image; we
    flip it back before resizing.
    """
    img = np.ascontiguousarray(img_hwc_uint8[::-1])  # un-flip
    t = torch.from_numpy(img).float().div_(255.0)  # (H, W, 3) in [0, 1]
    t = t.permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
    t = F.interpolate(t, size=(image_size, image_size), mode="bilinear", align_corners=False)
    t = t.squeeze(0)  # (3, image_size, image_size)
    return (t - SIGLIP_MEAN) / SIGLIP_STD


class LiberoDataset(Dataset):
    """
    split='train' uses all demos except the last `val_holdout` per file.
    split='val'   uses the last `val_holdout` demos per file.
    split='all'   uses everything (useful for final training).

    Action + proprio stats are always computed from the train split and cached,
    so val uses the same normalization as train.
    """

    def __init__(
        self,
        data_root: str | os.PathLike,
        suite: str = "libero_spatial",
        chunk_size: int = 10,
        image_size: int = 224,
        split: str = "train",
        val_holdout: int = 5,
        action_stats: dict[str, list[float]] | None = None,
        proprio_stats: dict[str, list[float]] | None = None,
        cache_stats: bool = True,
        use_wrist: bool = True,
        use_proprio: bool = True,
        augment: bool = False,
    ):
        assert split in ("train", "val", "all"), split
        self.augment_fn = build_augmentation() if augment else None
        self.root = Path(data_root) / suite
        self.suite = suite
        self.chunk_size = chunk_size
        self.image_size = image_size
        self.split = split
        self.val_holdout = val_holdout
        self.use_wrist = use_wrist
        self.use_proprio = use_proprio

        self.files: list[Path] = sorted(self.root.glob("*.hdf5"))
        if not self.files:
            raise FileNotFoundError(
                f"No .hdf5 files under {self.root}. "
                f"Run scripts/download_data.py --suites {suite} first."
            )
        self.instructions: list[str] = [instruction_from_filename(str(p)) for p in self.files]

        action_stats_path = self.root / "action_stats.json"
        proprio_stats_path = self.root / "proprio_stats.json"
        if action_stats is None and cache_stats and action_stats_path.exists():
            with open(action_stats_path, "r") as f:
                action_stats = json.load(f)
        if (
            use_proprio
            and proprio_stats is None
            and cache_stats
            and proprio_stats_path.exists()
        ):
            with open(proprio_stats_path, "r") as f:
                proprio_stats = json.load(f)

        self.index: list[tuple[int, str, int]] = []
        collected_actions: list[np.ndarray] = [] if action_stats is None else None  # type: ignore[assignment]
        collected_proprio: list[np.ndarray] = (
            [] if (use_proprio and proprio_stats is None) else None  # type: ignore[assignment]
        )

        for fi, path in enumerate(self.files):
            with h5py.File(path, "r") as f:
                data_grp = f["data"]
                demo_keys = sorted(
                    data_grp.keys(),
                    key=lambda k: int(k.split("_")[-1]) if k.startswith("demo_") else 0,
                )
                n_demos = len(demo_keys)
                if split == "train":
                    use_keys = demo_keys[: max(0, n_demos - val_holdout)]
                elif split == "val":
                    use_keys = demo_keys[max(0, n_demos - val_holdout) :]
                else:
                    use_keys = demo_keys

                for demo_key in use_keys:
                    demo = data_grp[demo_key]
                    T = int(demo["actions"].shape[0])
                    n_samples = T - chunk_size + 1
                    if n_samples <= 0:
                        continue
                    for t in range(n_samples):
                        self.index.append((fi, demo_key, t))
                    if collected_actions is not None:
                        collected_actions.append(demo["actions"][:])
                    if collected_proprio is not None:
                        ee_pos = demo["obs"]["ee_pos"][:]           # (T, 3)
                        gripper = demo["obs"]["gripper_states"][:]  # (T, 2)
                        collected_proprio.append(
                            np.concatenate([ee_pos, gripper], axis=1)
                        )

        if action_stats is None:
            all_a = np.concatenate(collected_actions, axis=0)  # (N, 7)
            q01 = np.quantile(all_a, 0.01, axis=0)
            q99 = np.quantile(all_a, 0.99, axis=0)
            action_stats = {"min": q01.tolist(), "max": q99.tolist()}
            if cache_stats:
                with open(action_stats_path, "w") as f:
                    json.dump(action_stats, f, indent=2)

        if use_proprio and proprio_stats is None:
            all_p = np.concatenate(collected_proprio, axis=0)  # (N, 5)
            q01 = np.quantile(all_p, 0.01, axis=0)
            q99 = np.quantile(all_p, 0.99, axis=0)
            proprio_stats = {"min": q01.tolist(), "max": q99.tolist()}
            if cache_stats:
                with open(proprio_stats_path, "w") as f:
                    json.dump(proprio_stats, f, indent=2)

        self.action_min = np.array(action_stats["min"], dtype=np.float32)
        self.action_max = np.array(action_stats["max"], dtype=np.float32)
        if use_proprio:
            assert proprio_stats is not None
            self.proprio_min = np.array(proprio_stats["min"], dtype=np.float32)
            self.proprio_max = np.array(proprio_stats["max"], dtype=np.float32)
        else:
            self.proprio_min = self.proprio_max = None  # type: ignore[assignment]

        # lazy per-worker HDF5 handle cache
        self._handles: dict[int, h5py.File] = {}

    # Avoid pickling open HDF5 handles when DataLoader forks workers
    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_handles"] = {}
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._handles = {}

    @property
    def stats(self) -> dict[str, list[float]]:
        return {"min": self.action_min.tolist(), "max": self.action_max.tolist()}

    @property
    def proprio_stats_dict(self) -> dict[str, list[float]] | None:
        if not self.use_proprio:
            return None
        return {"min": self.proprio_min.tolist(), "max": self.proprio_max.tolist()}

    def _file(self, fi: int) -> h5py.File:
        h = self._handles.get(fi)
        if h is None:
            h = h5py.File(self.files[fi], "r")
            self._handles[fi] = h
        return h

    def normalize_actions(self, a: np.ndarray) -> np.ndarray:
        rng = np.maximum(self.action_max - self.action_min, 1e-6)
        a = 2.0 * (a - self.action_min) / rng - 1.0
        return np.clip(a, -1.0, 1.0)

    def denormalize_actions(self, a: np.ndarray) -> np.ndarray:
        rng = self.action_max - self.action_min
        return (a + 1.0) / 2.0 * rng + self.action_min

    def normalize_proprio(self, p: np.ndarray) -> np.ndarray:
        rng = np.maximum(self.proprio_max - self.proprio_min, 1e-6)
        p = 2.0 * (p - self.proprio_min) / rng - 1.0
        return np.clip(p, -1.0, 1.0)

    def _apply_augment(self, img: torch.Tensor) -> torch.Tensor:
        img_01 = img * SIGLIP_STD + SIGLIP_MEAN
        img_01 = self.augment_fn(img_01)
        return (img_01 - SIGLIP_MEAN) / SIGLIP_STD

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int) -> dict[str, Any]:
        fi, demo_key, t = self.index[i]
        f = self._file(fi)
        demo = f["data"][demo_key]
        img = demo["obs"]["agentview_rgb"][t]  # (128, 128, 3) uint8
        actions = demo["actions"][t : t + self.chunk_size].astype(np.float32)

        pixel_values = preprocess_image(img, self.image_size)
        if self.augment_fn is not None:
            pixel_values = self._apply_augment(pixel_values)
        actions_norm = self.normalize_actions(actions)

        sample: dict[str, Any] = {
            "pixel_values": pixel_values,
            "actions": torch.from_numpy(actions_norm),
            "instruction": self.instructions[fi],
        }

        if self.use_wrist:
            wrist_img = demo["obs"]["eye_in_hand_rgb"][t]
            wrist_pv = preprocess_image(wrist_img, self.image_size)
            if self.augment_fn is not None:
                wrist_pv = self._apply_augment(wrist_pv)
            sample["wrist_pixel_values"] = wrist_pv

        if self.use_proprio:
            ee_pos = demo["obs"]["ee_pos"][t].astype(np.float32)        # (3,)
            gripper = demo["obs"]["gripper_states"][t].astype(np.float32)  # (2,)
            proprio = np.concatenate([ee_pos, gripper], axis=0)
            sample["proprio"] = torch.from_numpy(self.normalize_proprio(proprio))

        return sample


class LiberoCollator:
    """Stacks tensors and tokenizes instructions."""

    def __init__(self, tokenizer, max_text_len: int = 32):
        self.tokenizer = tokenizer
        self.max_text_len = max_text_len

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        pixel_values = torch.stack([b["pixel_values"] for b in batch], dim=0)
        actions = torch.stack([b["actions"] for b in batch], dim=0)
        instructions = [b["instruction"] for b in batch]
        tok = self.tokenizer(
            instructions,
            padding="max_length",
            truncation=True,
            max_length=self.max_text_len,
            return_tensors="pt",
        )
        out: dict[str, Any] = {
            "pixel_values": pixel_values,
            "input_ids": tok.input_ids,
            "attention_mask": tok.attention_mask,
            "actions": actions,
            "instructions": instructions,
        }
        if "wrist_pixel_values" in batch[0]:
            out["wrist_pixel_values"] = torch.stack(
                [b["wrist_pixel_values"] for b in batch], dim=0
            )
        if "proprio" in batch[0]:
            out["proprio"] = torch.stack([b["proprio"] for b in batch], dim=0)
        return out


def build_joint_datasets(
    data_root: str | os.PathLike,
    suites: list[str],
    chunk_size: int = 10,
    image_size: int = 224,
    val_holdout: int = 5,
    use_wrist: bool = True,
    use_proprio: bool = True,
    augment: bool = False,
) -> tuple[ConcatDataset, ConcatDataset, dict, dict | None]:
    """Build train/val ConcatDatasets spanning multiple suites.

    Action and proprio stats are computed per-suite then merged (global
    min of all per-suite mins, global max of all per-suite maxes) so
    normalisation is consistent across suites.
    """
    train_datasets: list[LiberoDataset] = []
    for suite in suites:
        ds = LiberoDataset(
            data_root=data_root,
            suite=suite,
            chunk_size=chunk_size,
            image_size=image_size,
            split="train",
            use_wrist=use_wrist,
            use_proprio=use_proprio,
            augment=augment,
        )
        train_datasets.append(ds)

    global_a_min = np.minimum.reduce([d.action_min for d in train_datasets])
    global_a_max = np.maximum.reduce([d.action_max for d in train_datasets])
    global_action_stats = {"min": global_a_min.tolist(), "max": global_a_max.tolist()}

    global_proprio_stats = None
    if use_proprio:
        global_p_min = np.minimum.reduce([d.proprio_min for d in train_datasets])
        global_p_max = np.maximum.reduce([d.proprio_max for d in train_datasets])
        global_proprio_stats = {"min": global_p_min.tolist(), "max": global_p_max.tolist()}

    for ds in train_datasets:
        ds.action_min = global_a_min
        ds.action_max = global_a_max
        if use_proprio:
            ds.proprio_min = global_p_min
            ds.proprio_max = global_p_max

    val_datasets: list[LiberoDataset] = []
    for suite in suites:
        v_ds = LiberoDataset(
            data_root=data_root,
            suite=suite,
            chunk_size=chunk_size,
            image_size=image_size,
            split="val",
            action_stats=global_action_stats,
            proprio_stats=global_proprio_stats,
            use_wrist=use_wrist,
            use_proprio=use_proprio,
            augment=False,
        )
        val_datasets.append(v_ds)

    return (
        ConcatDataset(train_datasets),
        ConcatDataset(val_datasets),
        global_action_stats,
        global_proprio_stats,
    )
