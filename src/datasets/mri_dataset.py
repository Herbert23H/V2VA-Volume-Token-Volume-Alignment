# Copyright (c) Herbert Hao, Xiamen University, School of Aerospace Engineering.
# All rights reserved.
#

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import nibabel as nib
import numpy as np
import torch
from logging import getLogger
from torch.utils.data import Dataset

from app.vjepa.transforms_mri import MakeMRITransforms


_GLOBAL_SEED = 0
logger = getLogger()

class MRIDataset(Dataset):
    """ MRI dataset for handling multiple modalities (PD, T1, T2). """

    def __init__(
        self,
        data_paths,
        transform = None,
        modalities=('pdsag', 't1sag', 't2sag'),
        filter_invalid=True,
    ):
        """
        :param data_paths: List of patient folder paths.
        :param modalities: Tuple of modalities to load (e.g., PD, T1, T2).
        :param transform: Transformations to apply to the data.
        :param filter_invalid: Whether to filter out invalid files.
        """
        self.data_paths = data_paths
        self.modalities = modalities
        self.transform = transform
        self.filter_invalid = filter_invalid

        self.samples = []
        for dataset_folder in self.data_paths:  # multi-dataset
            for patient_folder in os.listdir(dataset_folder):
                folder = os.path.join(dataset_folder, patient_folder)   
                modality_files = [
                    os.path.join(folder, f"{modality}.nii.gz") for modality in self.modalities
                ]
                if all(os.path.exists(f) for f in modality_files):
                    self.samples.append(modality_files)
                elif self.filter_invalid:
                    logger.warning(f"Invalid folder skipped: {patient_folder}")
                    # # 删除空folder
                    # os.rmdir(folder)

        logger.info(f"Loaded {len(self.samples)} valid patient folders.")

    def __getitem__(self, index):
        modality_files = self.samples[index]    # # 二重列表，每个元素是一个患者的三个模态文件路径列表
        data = []

        # 为当前调用生成一个随机种子
        seed = np.random.randint(0, 1e6)

        for file in modality_files: 
            volume = nib.load(file).get_fdata()
            # Ensure data is 3D and has depth of 20
            # if volume.shape[-1] != 20:  # 深度维在最后
            #     raise ValueError(f"Invalid depth for file {file}: {volume.shape[-1]}")
            # Apply transformations before stacking
            if self.transform:  # 同一组内每个序列采样同样的变换
                # print("current seed:", seed)
                use_transforms = self.transform(training=True)       # 实例化
                volume = use_transforms(volume, seed)   # 传递随机种子

            # save transformed niigz for checking
            # volume_save = volume.numpy().astype(np.float32)
            # debug_dir = os.environ.get("V2VA_NII_DEBUG_DIR", "outputs/nii_debug")
            # fname = file.split('/')[-1].replace('.nii.gz', '_transformed.nii.gz')
            # nib.save(nib.Nifti1Image(volume_save, np.eye(4)),
            #          f"{debug_dir}/pretrain_{file.split('/')[-2]}{fname}")
            data.append(volume)

        # Stack modalities to form (3, 20, H, W)
        data = np.stack(data, axis=0)  # Shape: (3, 20, H, W)
        # data = torch.from_numpy(data).unsqueeze(0)  # 添加 batch 维度
        # print(f"Data shape after stacking modalities: {data.shape}")  # (1, 3, 20, H, W)
        return data

    def __len__(self):
        return len(self.samples)


def make_mridataset(
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
    dataset = MRIDataset(
        data_paths=data_paths,
        transform=transform,
    )
    logger.info('MRIDataset created')

    dist_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset=dataset,
        num_replicas=world_size,    # 参与分布式训练的总进程数（或 GPU 数量）
        rank=rank,                  # 当前进程的唯一标识符（从 0 到 world_size-1）
    )   # 分布式采样器，将数据集划分为多个子集，为每个进程生成其处理的数据索引

    data_loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=collator,
        sampler=dist_sampler,
        batch_size=batch_size,
        drop_last=drop_last,
        pin_memory=pin_mem,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        # shuffle=False,  # 分布式采样器内含shuffle逻辑
    )
    logger.info('MRIDataset data loader created')

    return dataset, data_loader, dist_sampler


if __name__ == "__main__":

    # 目前只有一个数据集，也支持多个数据集以列表格式输入
    data_paths = [os.environ.get('V2VA_MRI_HEALTHY_NII_DIR', 'data/mri/healthy_nii')]

    # # 模拟分布式环境
    # world_size = 4  # 模拟 4 GPU

    
    dataset, data_loader, dist_sampler = make_mridataset(
        data_paths=data_paths,
        batch_size=2,
        transform=MakeMRITransforms,
        collator=None,
        pin_mem=True,
        num_workers=1,
        world_size=1,   # 一卡
        rank=0,
        drop_last=True,
        persistent_workers=False,
    )
    for i, data in enumerate(data_loader):
        print(f"Batch {i}: data shape = {data.shape}")
        if i == 2:
            break


    # for rank in range(world_size):
    #     print(f"\nTesting for rank {rank}:")


    #     # 打印每个进程采样到的数据
    #     for batch in data_loader:
    #         print(f"Rank {rank}, Batch: {batch}")
