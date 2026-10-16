# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from pathlib import Path

from logging import getLogger


_GLOBAL_SEED = 0
logger = getLogger()


def init_mri_data(
    batch_size,
    transform=None,
    data=None,
    collator=None,
    pin_mem=True,
    num_workers=8,
    world_size=1,
    rank=0,
    root_path=None,
    drop_last=True,
    persistent_workers=True,
):
        
    if data.lower() == 'mridataset': 
        from src.datasets.mri_dataset import make_mridataset
        dataset, data_loader, dist_sampler = make_mridataset(
            data_paths=root_path,
            batch_size=batch_size,
            transform=transform,
            collator=collator,
            pin_mem=pin_mem,
            num_workers=num_workers,
            world_size=world_size,
            rank=rank,
            drop_last=drop_last,
            persistent_workers=persistent_workers,
        )
    elif data.lower() == 'medmnist3d':
        from src.datasets.medmnist3d_dataset import make_medmnist3d_dataset
        dataset, data_loader, dist_sampler = make_medmnist3d_dataset(
            data_paths=root_path,
            batch_size=batch_size,
            transform=transform,
            collator=collator,
            pin_mem=pin_mem,
            num_workers=num_workers,
            world_size=world_size,
            rank=rank,
            drop_last=drop_last,
            persistent_workers=persistent_workers,
        )
    elif data.lower() == 'mrnet':
        from src.datasets.mrnet_dataset import make_mrnet_dataset

        # Derive train label path from first data root
        if isinstance(root_path, (list, tuple)) and len(root_path) > 0:
            _train_label = str(Path(root_path[0]) / 'train_label.xlsx')
        elif isinstance(root_path, str):
            _train_label = str(Path(root_path) / 'train_label.xlsx')
        else:
            _train_label = None
        logger.info(f'MRNet pretraining restricted to train split via {_train_label}')

        dataset, data_loader, dist_sampler = make_mrnet_dataset(
            data_paths=root_path,
            batch_size=batch_size,
            transform=transform,
            collator=collator,
            pin_mem=pin_mem,
            num_workers=num_workers,
            world_size=world_size,
            rank=rank,
            drop_last=drop_last,
            persistent_workers=persistent_workers,
            label_path=_train_label,
        )
    else:
        raise NotImplementedError(f'Dataset {data} not implemented for MRI!')
    return (data_loader, dist_sampler)
