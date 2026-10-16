# Copyright (c) 2026
# All rights reserved.
#
# MedMNIST3D dataset loader for npz volumes.

from __future__ import annotations

from logging import getLogger
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


logger = getLogger()


class MedMNIST3DDataset(Dataset):
    """Load MedMNIST 3D volumes from one or more npz files.

    Each npz is expected to contain keys like train_images/val_images/test_images.
    This dataset uses the union of all images across provided splits.
    Labels are not used for self-supervised pretraining.
    """

    def __init__(
        self,
        data_paths: List[str],
        split_keys: Tuple[str, ...] = ("train_images",),
        transform=None,
    ) -> None:
        self.data_paths = [str(p) for p in data_paths]
        self.split_keys = split_keys
        self.transform = transform

        self._arrays: List[np.ndarray] = []
        self._cum_lengths: List[int] = []

        total = 0
        for p in self.data_paths:
            path = Path(p)
            if not path.exists():
                raise FileNotFoundError(f"MedMNIST3D npz not found: {path}")
            with np.load(path, allow_pickle=False) as npz:
                for key in self.split_keys:
                    if key in npz:
                        arr = npz[key]
                        if arr.ndim < 3:
                            continue
                        self._arrays.append(arr)
                        total += arr.shape[0]
                        self._cum_lengths.append(total)
                    print(f"Loaded {arr.shape[0]} volumes from {path} key {key}, cumulative total: {total}")
        if total == 0:
            raise ValueError("No images found in provided npz files.")

        logger.info(f"Loaded MedMNIST3D volumes: {total}")

    def __len__(self) -> int:
        return self._cum_lengths[-1]

    def _locate(self, index: int) -> Tuple[np.ndarray, int]:
        for arr, cum in zip(self._arrays, self._cum_lengths):
            if index < cum:
                prev = cum - arr.shape[0]
                return arr, index - prev
        raise IndexError("Index out of range")

    def __getitem__(self, index: int):
        arr, local_idx = self._locate(index)
        vol = arr[local_idx]

        # Normalize shape to (D, H, W)
        if vol.ndim == 4 and vol.shape[-1] == 1:
            vol = vol[..., 0]
        elif vol.ndim == 4 and vol.shape[0] == 1:
            vol = vol[0]
        elif vol.ndim != 3:
            raise ValueError(f"Unexpected volume shape: {vol.shape}")

        if self.transform is not None:
            use_transforms = self.transform(training=True)
            vol = use_transforms(vol, seed=None)

        if isinstance(vol, np.ndarray):
            vol = torch.from_numpy(vol)
        if not torch.is_floating_point(vol):
            vol = vol.float()

        if vol.ndim == 3:
            vol = vol.unsqueeze(0)

        return vol


def make_medmnist3d_dataset(
    data_paths,
    batch_size,
    transform=None,
    collator=None,
    pin_mem=True,
    num_workers=8,
    world_size=1,
    rank=0,
    drop_last=True,
    persistent_workers=False,
):
    dataset = MedMNIST3DDataset(
        data_paths=data_paths,
        transform=transform,
    )
    logger.info('MedMNIST3D dataset created')

    dist_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset=dataset,
        num_replicas=world_size,
        rank=rank,
    )

    data_loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=collator,
        sampler=dist_sampler,
        batch_size=batch_size,
        drop_last=drop_last,
        pin_memory=pin_mem,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
    )
    logger.info('MedMNIST3D data loader created')

    return dataset, data_loader, dist_sampler
