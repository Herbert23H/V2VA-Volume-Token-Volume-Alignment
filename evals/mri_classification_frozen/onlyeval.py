import os
from tracemalloc import start

# -- FOR DISTRIBUTED TRAINING ENSURE ONLY 1 DEVICE VISIBLE PER PROCESS
try:
    # -- WARNING: IF DOING DISTRIBUTED TRAINING ON A NON-SLURM CLUSTER, MAKE
    # --          SURE TO UPDATE THIS TO GET LOCAL-RANK ON NODE, OR ENSURE
    # --          THAT YOUR JOBS ARE LAUNCHED WITH ONLY 1 DEVICE VISIBLE
    # --          TO EACH PROCESS
    os.environ['CUDA_VISIBLE_DEVICES'] = os.environ['SLURM_LOCALID']
except Exception:
    pass

import logging
import pprint

import time
import copy
import ast
import numpy as np
import csv

import torch
import torch.multiprocessing as mp
import torch.nn.functional as F

from torch.nn.parallel import DistributedDataParallel

from src.datasets.data_manager_mri import (
    init_mri_data
)
from src.utils.distributed import (
    init_distributed,
    AllReduce
)
from src.utils.schedulers import (
    WarmupCosineSchedule,
    CosineWDSchedule,
)
from src.utils.logging import (
    AverageMeter,
    CSVLogger
)

from evals.mri_classification_frozen.utils import init_model_frozen, MaskAggregation
from app.vjepa.transforms_mri import MakeMRITransforms
from src.models.attentive_pooler import MultiHeadAttentiveClassifier


logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

_GLOBAL_SEED = 0
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True

pp = pprint.PrettyPrinter(indent=4)


def main(args_eval, resume_preempt=False):
    # ----------------------------------------------------------------------- #
    #  PASSED IN PARAMS FROM CONFIG FILE
    # ----------------------------------------------------------------------- #

    # -- PRETRAIN
    args_pretrain = args_eval.get('pretrain')
    model_name = args_pretrain.get('model_name', None)
    patch_size = args_pretrain.get('patch_size', None)
    pretrain_folder = args_pretrain.get('folder', None)
    ckp_fname = args_pretrain.get('checkpoint', None)
    tag = args_pretrain.get('write_tag', None)
    pretrained_path = os.path.join(pretrain_folder, ckp_fname)
    # Optional [for Video model]:
    depth_patch_size = args_pretrain.get('depth_patch_size', 1)
    use_swin_backbone = args_pretrain.get('use_swin_backbone', False)
    swin_backbone_name = args_pretrain.get('swin_backbone_name', 'swin_tiny_3d')
    swin_embed_dim = args_pretrain.get('swin_embed_dim', 96)
    swin_window_size = args_pretrain.get('swin_window_size', (7, 7, 7))
    swin_patch_size = args_pretrain.get('swin_patch_size', (4, 4, 4))
    swin_depths = args_pretrain.get('swin_depths', (2, 2, 6, 2))
    swin_num_heads = args_pretrain.get('swin_num_heads', (3, 6, 12, 24))
    swin_drop_rate = args_pretrain.get('swin_drop_rate', 0.0)
    swin_attn_drop_rate = args_pretrain.get('swin_attn_drop_rate', 0.0)
    swin_drop_path_rate = args_pretrain.get('swin_drop_path_rate', 0.0)
    swin_patch_norm = args_pretrain.get('swin_patch_norm', False)
    swin_use_checkpoint = args_pretrain.get('swin_use_checkpoint', False)
    swin_use_v2 = args_pretrain.get('swin_use_v2', False)

    # -- DATA
    args_data = args_eval.get('data')
    dataset_path = args_data.get('dataset_path')
    dataset_type = args_data.get('dataset_type')
    mask_size = args_data.get('mask_size')
    pin_mem = args_data.get('pin_mem', False)
    num_workers = args_data.get('num_workers', 1)

    # -- OPTIMIZATION
    args_opt = args_eval.get('optimization')
    resolution = args_opt.get('resolution', 320)
    batch_size = args_opt.get('batch_size')
    attend_across_segments = args_opt.get('attend_across_segments', False)
    num_epochs = args_opt.get('num_epochs')
    use_bfloat16 = args_opt.get('use_bfloat16')
    if use_swin_backbone:
        raise ValueError('onlyeval pipeline is MaskAggregation-only and does not support Swin backbone')
    
    # -- EXPERIMENT-ID/TAG (optional)
    eval_tag = args_eval.get('tag', None)
    
    # ----------------------------------------------------------------------- #
    # ----------------------------------------------------------------------- # 
    torch.backends.cudnn.benchmark = True

    try:                                                
        mp.set_start_method('spawn')
    except Exception:
        pass

    if not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:0')
        torch.cuda.set_device(device)

    world_size, rank = init_distributed()
    logger.info(f'Initialized (rank/world-size) {rank}/{world_size}')

    # -- log/checkpointing paths
    folder = os.path.join(pretrain_folder, 'MRI_frozen/')
    if eval_tag is not None:
        folder = os.path.join(folder, eval_tag)
    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)
    log_file = os.path.join(folder, f'{tag}_r{rank}_onlyeval.csv')

    # -- make csv_logger
    if rank == 0:
        csv_logger = CSVLogger(log_file,
                               ('%d', 'epoch'),
                               ('%.5f', 'loss'),
                               ('%.5f', 'acc'))

    # -- frozen pretrained model
    encoder, predictor = init_model_frozen(
        device,
        pretrained = pretrained_path,
        model_name = model_name,
        patch_size=16,
        crop_size=320,
        num_slices=20,
        depth_patch_size=depth_patch_size,
        use_sdpa=False,
        use_SiLU=False,
        tight_SiLU=True,
        uniform_power=False,
        use_swin_backbone=use_swin_backbone,
        swin_backbone_name=swin_backbone_name,
        swin_embed_dim=swin_embed_dim,
        swin_window_size=swin_window_size,
        swin_patch_size=swin_patch_size,
        swin_depths=swin_depths,
        swin_num_heads=swin_num_heads,
        swin_drop_rate=swin_drop_rate,
        swin_attn_drop_rate=swin_attn_drop_rate,
        swin_drop_path_rate=swin_drop_path_rate,
        swin_patch_norm=swin_patch_norm,
        swin_use_checkpoint=swin_use_checkpoint,
        swin_use_v2=swin_use_v2,
    )
    target_encoder = copy.deepcopy(encoder)

    deviation_model = MaskAggregation(  # 联同predictor、target encoder封装
        device,
        encoder,
        predictor,
        target_encoder,
        mask_size=ast.literal_eval(mask_size),
    ).to(device)
    deviation_model.eval()
    deviation_model = DistributedDataParallel(deviation_model, static_graph=True)
    for p in deviation_model.parameters():
        p.requires_grad = False


    # -- dataloaders
    (unsupervised_loader,
     unsupervised_sampler) = init_mri_data(
        root_path=dataset_path,
        batch_size = batch_size,
        transform=MakeMRITransforms,
        data='MRIDataset',
        pin_mem=pin_mem,
        num_workers=num_workers,
        world_size=world_size,
        rank=rank,
        drop_last=True,
        persistent_workers=False,
    )

    ipe = len(unsupervised_loader)
    logger.info(f'Dataloader created... iterations per epoch: {ipe}')

    record_one_epoch(
        device,
        deviation_model=deviation_model,
        data_loader=unsupervised_loader,
        dataset_type=dataset_type,
        use_bfloat16=use_bfloat16,
    )



def eval_one_epoch(
    device,
    deviation_model,
    data_loader,
    use_bfloat16,
):
    
    autocast_dtype = (torch.bfloat16 if use_bfloat16 else torch.float16)
    for itr, data in enumerate(data_loader):
        start_time=time.time()
        with torch.autocast(device_type="cuda", dtype=autocast_dtype):
            # Load data and put on GPU
            img = data
            img = img.to(device, dtype=torch.float16 if use_bfloat16 else torch.float32, non_blocking=True)

            # Forward and prediction
            with torch.no_grad():
                dz = deviation_model(img)    # torch.Size([1, 8000, 1024])
                print("dz:",dz.shape)

            # 基本统计：逐维均值与方差
            dz_flat = dz.reshape(-1, dz.size(-1))  # (N_patches, D)
            mean_per_dim = dz_flat.mean(dim=0)
            std_per_dim = dz_flat.std(dim=0)
            print("mean_per_dim(mean,std):", mean_per_dim.mean().item(), std_per_dim.mean().item())
            print("std_per_dim min/max:", std_per_dim.min().item(), std_per_dim.max().item())

            # 范数分布（每patch特征向量的L2范数）
            norms = dz_flat.norm(dim=1)
            print("norms(mean,std,min,max):", norms.mean().item(), norms.std().item(), norms.min().item(), norms.max().item())

            # 余弦相似度（随机抽样若干对）
            idx = torch.randint(0, dz_flat.size(0), (1024,))
            pairs = torch.randint(0, dz_flat.size(0), (1024,))
            cos = F.cosine_similarity(dz_flat[idx], dz_flat[pairs], dim=1)
            print("cos(mean,std,min,max):", cos.mean().item(), cos.std().item(), cos.min().item(), cos.max().item())

            # PCA/SVD谱（子样本以节省计算）
            sub = dz_flat[:: max(1, dz_flat.size(0)//4000)]  # ≤4000样本
            sub = sub - sub.mean(dim=0, keepdim=True)
            U, S, V = torch.linalg.svd(sub, full_matrices=False)
            energy = (S**2)
            print("SVD: top5 singular values:", S[:5].tolist())
            print("Explained energy ratio top1:", (energy[0]/energy.sum()).item())
            print("Effective rank (energy阈值0.99):", (energy.cumsum(0)/energy.sum() < 0.99).sum().item())

            # 简单塌陷判据
            eps = 1e-6
            collapsed_dims = (std_per_dim < 1e-3).sum().item()
            high_cos_ratio = (cos > 0.99).float().mean().item()
            print(f"collapsed_dims: {collapsed_dims} / {dz.size(-1)}, high_cos_ratio: {high_cos_ratio:.4f}")


        devidation_time = time.time()-start_time
        print(f"Deviation model time: {devidation_time:.4f}s")

    return 


def record_one_epoch(
    device,
    deviation_model,
    data_loader,
    dataset_type,
    use_bfloat16,
):

    outdir = os.environ.get("V2VA_ONLYEVAL_OUTDIR", "experiments/onlyeval")
    os.makedirs(outdir, exist_ok=True)
    stats_rows = []
    dim_means_list, dim_stds_list = [], []
    global_idx = 0

    autocast_dtype = (torch.bfloat16 if use_bfloat16 else torch.float16)
    for itr, data in enumerate(data_loader):
        start_time=time.time()
        with torch.autocast(device_type="cuda", dtype=autocast_dtype):
            # Load data and put on GPU
            img = data
            img = img.to(device, dtype=torch.float16 if use_bfloat16 else torch.float32, non_blocking=True)

            # Forward and prediction
            with torch.no_grad():
                dz = deviation_model(img)    # torch.Size([B, N_patches, D])
                print("dz:",dz.shape)

            # 按样本统计与累积存储
            B, _, D = dz.shape
            for b in range(B):
                dz_b = dz[b].reshape(-1, D)  # (N_patches, D)

                mean_per_dim = dz_b.mean(dim=0)
                std_per_dim = dz_b.std(dim=0)

                norms = dz_b.norm(dim=1)
                n_pairs = min(1024, dz_b.size(0))
                idx = torch.randint(0, dz_b.size(0), (n_pairs,), device=dz_b.device)
                pairs = torch.randint(0, dz_b.size(0), (n_pairs,), device=dz_b.device)
                cos = F.cosine_similarity(dz_b[idx], dz_b[pairs], dim=1)

                sub = dz_b[:: max(1, dz_b.size(0)//4000)]
                sub = sub - sub.mean(dim=0, keepdim=True)
                U, S, V = torch.linalg.svd(sub, full_matrices=False)
                energy = (S**2)
                top1_ratio = (energy[0]/energy.sum()).item()
                eff_rank = int((energy.cumsum(0)/energy.sum() < 0.99).sum().item())

                collapsed_dims = int((std_per_dim < 1e-3).sum().item())
                high_cos_ratio = float((cos > 0.99).float().mean().item())

                # 累积维度向量
                dim_means_list.append(mean_per_dim.detach().cpu().numpy())
                dim_stds_list.append(std_per_dim.detach().cpu().numpy())

                # 累积标量（写CSV）
                stats_rows.append({
                    "sample_idx": global_idx,
                    "mean_mean": float(mean_per_dim.mean().item()),
                    "std_mean": float(std_per_dim.mean().item()),
                    "std_min": float(std_per_dim.min().item()),
                    "std_max": float(std_per_dim.max().item()),
                    "norm_mean": float(norms.mean().item()),
                    "norm_std": float(norms.std().item()),
                    "norm_min": float(norms.min().item()),
                    "norm_max": float(norms.max().item()),
                    "cos_mean": float(cos.mean().item()),
                    "cos_std": float(cos.std().item()),
                    "cos_min": float(cos.min().item()),
                    "cos_max": float(cos.max().item()),
                    "svd_top1_ratio": float(top1_ratio),
                    "svd_eff_rank": eff_rank,
                    "collapsed_dims": collapsed_dims,
                    "high_cos_ratio": high_cos_ratio,
                })
                global_idx += 1

            # ...existing code...
            dz_flat = dz.reshape(-1, dz.size(-1))  # (N_patches, D)
            mean_per_dim = dz_flat.mean(dim=0)
            std_per_dim = dz_flat.std(dim=0)
            print("mean_per_dim(mean,std):", mean_per_dim.mean().item(), std_per_dim.mean().item())
            print("std_per_dim min/max:", std_per_dim.min().item(), std_per_dim.max().item())
            # ...existing code...

        devidation_time = time.time()-start_time
        print(f"Deviation model time: {devidation_time:.4f}s")

        
    # 循环结束后写盘
    if stats_rows:
        csv_path = os.path.join(outdir, f"{dataset_type}_per_sample_stats.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(stats_rows[0].keys()))
            writer.writeheader()
            writer.writerows(stats_rows)

    if dim_means_list:
        npz_path = os.path.join(outdir, f"{dataset_type}_per_sample_dim_stats.npz")
        np.savez_compressed(
            npz_path,
            mean=np.stack(dim_means_list, axis=0),  # [N_samples, D]
            std=np.stack(dim_stds_list, axis=0),    # [N_samples, D]
        )



