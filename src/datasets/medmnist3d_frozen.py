# Copyright (c) 2026
# All rights reserved.
#
# MedMNIST3D dataset loader with labels for downstream classification.

from __future__ import annotations

from logging import getLogger
from math import trunc
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


logger = getLogger()


class MedMNIST3DClassification(Dataset):
    """Load a single MedMNIST3D split with labels from a .npz file."""

    def __init__(
        self,
        npz_path: str,
        split: str = "train",
        transform=None,
        training: bool = True,
    ) -> None:
        self.npz_path = Path(npz_path)
        self.split = split
        self.transform = transform
        self.training = training

        if not self.npz_path.exists():
            raise FileNotFoundError(f"MedMNIST3D npz not found: {self.npz_path}")

        with np.load(self.npz_path, allow_pickle=False) as npz:
            images_key = f"{split}_images"
            labels_key = f"{split}_labels"
            if images_key not in npz or labels_key not in npz:
                raise KeyError(f"Missing keys: {images_key}/{labels_key} in {self.npz_path}")
            self.images = npz[images_key]
            self.labels = npz[labels_key]

        if self.images.shape[0] != self.labels.shape[0]:
            raise ValueError("Images and labels have different lengths")

        logger.info(f"Loaded {self.images.shape[0]} samples from {self.npz_path} ({split})")

    def __len__(self) -> int:
        return self.images.shape[0]

    def __getitem__(self, idx: int):
        vol = self.images[idx]
        label = self.labels[idx]

        if vol.ndim == 4 and vol.shape[-1] == 1:
            vol = vol[..., 0]
        elif vol.ndim == 4 and vol.shape[0] == 1:
            vol = vol[0]
        elif vol.ndim != 3:
            raise ValueError(f"Unexpected volume shape: {vol.shape}")

        if self.transform is not None:
            use_transforms = self.transform(training=self.training)
            vol = use_transforms(vol, seed=None)

        if isinstance(vol, np.ndarray):
            vol = torch.from_numpy(vol)
        if not torch.is_floating_point(vol):
            vol = vol.float()
        if vol.ndim == 3:
            vol = vol.unsqueeze(0)

        if isinstance(label, np.ndarray):
            label = label.reshape(-1)
            label = int(label[0]) if label.size > 0 else int(label)
        label = torch.tensor([label], dtype=torch.long)

        return vol, label


def make_medmnist3d_dataloader(
    npz_path: str,
    split: str,
    transform,
    batch_size: int,
    num_workers: int = 12,
    training: bool = False,
    persistent_workers: bool = True,
    prefetch_factor: int = 4,
    pin_memory: bool = True,
    drop_last: bool = False,
):
    dataset = MedMNIST3DClassification(
        npz_path=npz_path,
        split=split,
        transform=transform,
        training=training,
    )

    loader_kwargs = {
        'batch_size': batch_size,
        'shuffle': training,    # 绘图方便找展示清晰图, 记得训练时改回training
        'num_workers': num_workers,
        'pin_memory': pin_memory,
        'drop_last': drop_last,
    }

    if num_workers > 0:
        loader_kwargs['persistent_workers'] = persistent_workers
        loader_kwargs['prefetch_factor'] = prefetch_factor

    loader = DataLoader(dataset, **loader_kwargs)
    return loader
