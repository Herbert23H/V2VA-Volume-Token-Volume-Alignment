# Copyright (c) 2026
# All rights reserved.
#
# MRNet dataset loader for knee MRI volumes (axial / coronal / sagittal).

from __future__ import annotations

from logging import getLogger
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


logger = getLogger()


class MRNetDataset(Dataset):
    """Load MRNet knee MRI volumes from three view folders (axial/coronal/sagittal).

    Each folder contains .npy files named by zero-padded index (0000.npy ~ N.npy).
    Volumes from all views are pooled together as individual samples.
    Labels are not used for self-supervised pretraining.

    If label_path is provided (e.g., '/path/to/train_label.xlsx'), only the
    sample IDs listed in the first column of the xlsx will be included,
    effectively restricting the dataset to a specific split (e.g., train set).
    """

    def __init__(
        self,
        data_paths: List[str],
        transform=None,
        views: Tuple[str, ...] = ("axial", "coronal", "sagittal"),
        label_path: str | None = None,
    ) -> None:
        self.data_paths = [str(p) for p in data_paths]
        self.views = views
        self.transform = transform
        self.label_path = label_path

        # Parse label file to get valid IDs (first column), if provided
        valid_ids: set[int] | None = None
        if label_path is not None:
            import pandas as pd
            df = pd.read_excel(label_path)
            ids = df.iloc[:, 0].astype(int).tolist()
            valid_ids = set(ids)
            logger.info(
                f"Restricting MRNetDataset to {len(valid_ids)} samples from {label_path}"
            )

        # Collect all volume file paths
        self._files: List[str] = []
        for data_dir in self.data_paths:
            for view in self.views:
                view_dir = Path(data_dir) / view
                if not view_dir.exists():
                    logger.warning(f"View directory not found: {view_dir}")
                    continue
                npy_files = sorted(view_dir.glob("*.npy"))
                for f in npy_files:
                    stem = int(Path(f).stem)  # e.g. 0000 -> 0
                    if valid_ids is None or stem in valid_ids:
                        self._files.append(str(f))

        if len(self._files) == 0:
            raise ValueError(
                f"No .npy files found in {self.data_paths} under views {self.views}"
            )

        logger.info(f"Loaded MRNet volumes: {len(self._files)} from {len(self.data_paths)} root(s)")

    def __len__(self) -> int:
        return len(self._files)

    def __getitem__(self, index: int):
        path = self._files[index]
        vol = np.load(path)

        # Normalize shape to (D, H, W)
        if vol.ndim == 4 and vol.shape[-1] == 1:
            vol = vol[..., 0]
        elif vol.ndim == 4 and vol.shape[0] == 1:
            vol = vol[0]
        elif vol.ndim != 3:
            raise ValueError(f"Unexpected volume shape at {path}: {vol.shape}")

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


def make_mrnet_dataset(
    data_paths,
    batch_size,
    transform=None,
    collator=None,
    pin_mem=True,
    num_workers=12,
    world_size=1,
    rank=0,
    drop_last=True,
    persistent_workers=False,
    views=("axial", "coronal", "sagittal"),
    label_path: str | None = None,
):
    dataset = MRNetDataset(
        data_paths=data_paths,
        transform=transform,
        views=views,
        label_path=label_path,
    )
    logger.info('MRNet dataset created')

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
    logger.info('MRNet data loader created')

    return dataset, data_loader, dist_sampler


class MRNetClassification(Dataset):
    """MRNet downstream classification dataset.

    Each sample = 3 views (axial, coronal, sagittal) of the same patient,
    loaded together with binary labels for multiple tasks.
    """

    def __init__(
        self,
        data_path: str,
        label_path: str,
        transform=None,
        views: Tuple[str, ...] = ("axial", "coronal", "sagittal"),
        training: bool = True,
        defect: Tuple[str, ...] = ("abnormal", "acl", "meniscus"),
    ) -> None:
        self.data_dir = Path(data_path)
        self.transform = transform
        self.views = views
        self.training = training
        self.defect = defect

        # Read labels from xlsx
        import pandas as pd
        df = pd.read_excel(label_path)
        # Ensure IDs are integers
        df.iloc[:, 0] = df.iloc[:, 0].astype(int)
        self._ids = df.iloc[:, 0].tolist()
        self._labels = torch.tensor(df[list(defect)].values, dtype=torch.long)

        logger.info(
            f"Loaded MRNet classification: {len(self._ids)} samples, "
            f"defect={defect}, from {label_path}"
        )

    def __len__(self) -> int:
        return len(self._ids)

    def __getitem__(self, idx: int):
        pid = self._ids[idx]
        label = self._labels[idx]
        seed = np.random.randint(0, 1_000_000) if self.training else 0

        vols = []
        for view in self.views:
            path = self.data_dir / view / f"{pid:04d}.npy"
            vol = np.load(str(path))
            if vol.ndim == 4 and vol.shape[-1] == 1:
                vol = vol[..., 0]
            elif vol.ndim == 4 and vol.shape[0] == 1:
                vol = vol[0]

            if self.transform is not None:
                use_transforms = self.transform(training=self.training)
                # print("shape before transform:", vol.shape)
                vol = use_transforms(vol, seed=seed)
                # print("shape after transform:", vol.shape)

            if isinstance(vol, np.ndarray):
                vol = torch.from_numpy(vol)
            if not torch.is_floating_point(vol):
                vol = vol.float()
            if vol.ndim == 3:
                vol = vol.unsqueeze(0)
            vols.append(vol)

        # vols: list of 3 tensors, each (1, D, H, W)
        return tuple(vols), label


def make_mrnet_dataloader(
    data_path: str,
    label_path: str,
    transform,
    batch_size: int,
    defect: tuple = ("abnormal", "acl", "meniscus"),
    training: bool = True,
    num_workers: int = 12,
    persistent_workers: bool = True,
    prefetch_factor: int = 4,
    pin_memory: bool = True,
    drop_last: bool = False,
    views: tuple = ("axial", "coronal", "sagittal"),
):
    dataset = MRNetClassification(
        data_path=data_path,
        label_path=label_path,
        transform=transform,
        views=views,
        training=training,
        defect=defect,
    )

    loader_kwargs = {
        'batch_size': batch_size,
        'shuffle': training,
        'num_workers': num_workers,
        'pin_memory': pin_memory,
        'drop_last': drop_last,
    }
    if num_workers > 0:
        loader_kwargs['persistent_workers'] = persistent_workers
        loader_kwargs['prefetch_factor'] = prefetch_factor

    loader = torch.utils.data.DataLoader(dataset, **loader_kwargs)
    return loader
