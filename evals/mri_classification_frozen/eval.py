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
import csv

import time
import copy
import ast
import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.multiprocessing as mp
import torch.nn.functional as F

from torch.nn.parallel import DistributedDataParallel

from src.datasets.data_manager import (
    init_data,
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
from src.datasets.mri_dataset_frozen import make_dataloader
from src.datasets.medmnist3d_frozen import make_medmnist3d_dataloader
from src.datasets.mrnet_dataset import make_mrnet_dataloader
from app.vjepa.transforms_mri import MakeMRITransforms, MakeMedMNIST3DTransforms, MakeMRNetTransforms
from src.models.attentive_pooler import (
    MultiHeadAttentiveClassifier,
    PlaneAwareMultiHeadClassifier,
    TokenToVolume3DClassifier,
    MRNetTriplePlaneClassifier,
    MRNetTriplePlanePerTaskClassifier,
)


logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

_GLOBAL_SEED = 0
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True

pp = pprint.PrettyPrinter(indent=4)


def build_three_plane_volumes(vol):
        """
        输入 vol: [B, C, D, H, W]
        输出三个平面视角体数据（都保持 [B, C, *, *, *]）:
            - axial:    [B, C, D, H, W]      (D 作深度)
            - coronal:  [B, C, H, D, W]      (H 作深度)
            - sagittal: [B, C, W, D, H]      (W 作深度)
        """
        axial = vol
        coronal = vol.permute(0, 1, 3, 2, 4).contiguous()
        sagittal = vol.permute(0, 1, 4, 2, 3).contiguous()
        return axial, coronal, sagittal


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
    load_pretrained_ckpt = args_pretrain.get('load_pretrained', True)
    tag = args_pretrain.get('write_tag', None)
    use_sdpa = args_pretrain.get('use_sdpa', True)
    use_SiLU = args_pretrain.get('use_silu', False)
    tight_SiLU = args_pretrain.get('tight_silu', True)
    uniform_power = args_pretrain.get('uniform_power', False)
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
    if isinstance(ckp_fname, str) and ckp_fname.strip().lower() in ('', 'none', 'null'):
        ckp_fname = None
    if (not load_pretrained_ckpt) or (ckp_fname is None):
        pretrained_path = None
    else:
        pretrained_path = os.path.join(pretrain_folder, ckp_fname)
    # Optional [for Video model]:
    depth_patch_size = args_pretrain.get('depth_patch_size', 1)
    unfreeze_raw = args_pretrain.get('unfreeze', 0)
    unfreeze_all = False
    if isinstance(unfreeze_raw, str):
        if unfreeze_raw.strip().lower() in ('all', 'full', 'full_encoder', 'all_encoder'):
            unfreeze_all = True
            unfreeze = 9999
        else:
            unfreeze = int(unfreeze_raw)
    else:
        unfreeze = int(unfreeze_raw)
    stage = args_pretrain.get('stage', None) 

    logger.info(
        f'Pretrain loading -> enable: {load_pretrained_ckpt and (pretrained_path is not None)}, '
        f'checkpoint: {pretrained_path if pretrained_path is not None else "<random-init>"}'
    )
    logger.info(
        f'Encoder fine-tune mode -> unfreeze: {unfreeze_raw} '
        f'(full_encoder={unfreeze_all})'
    )

    # -- DATA
    args_data = args_eval.get('data')
    dataset_type = args_data.get('dataset_type', 'mridataset')
    dataset_path = args_data.get('dataset_path')
    label_path = args_data.get('label_path')
    train_data_path = [args_data.get('dataset_train')]
    dataset_npz = args_data.get('dataset_npz')
    split_train = args_data.get('split_train', 'train')
    split_val = args_data.get('split_val', 'val')
    defect = args_data.get('defect', None)
    defect_classes = args_data.get('defect_classes', None)
    mask_size = args_data.get('mask_size')
    views = args_data.get('views', ["axial", "coronal", "sagittal"])

    # -- OPTIMIZATION
    args_opt = args_eval.get('optimization')
    batch_size = args_opt.get('batch_size')
    num_epochs = args_opt.get('num_epochs')
    wd = args_opt.get('weight_decay')
    start_lr = args_opt.get('start_lr')
    lr = args_opt.get('lr')
    final_lr = args_opt.get('final_lr')
    warmup = args_opt.get('warmup')
    use_bfloat16 = args_opt.get('use_bfloat16')
    scl_weight = args_opt.get('scl_weight', 0.0)
    scl_temperature = args_opt.get('scl_temperature', 0.03)
    num_workers = args_opt.get('num_workers', 8)
    persistent_workers = args_opt.get('persistent_workers', True)
    prefetch_factor = args_opt.get('prefetch_factor', 4)
    pin_memory = args_opt.get('pin_memory', True)
    train_drop_last = args_opt.get('train_drop_last', True)
    plot_loss_curve = args_opt.get('plot_loss_curve', True)

    # -- Layer-wise LR (optional)
    use_layerwise_lr = args_opt.get('use_layerwise_lr', False)
    encoder_lr = args_opt.get('encoder_lr', lr)
    pagca_lr = args_opt.get('pagca_lr', lr)
    classifier_lr = args_opt.get('classifier_lr', lr)

    # -- Plane-Agnostic Cross Attention + Plane Gate (optional, yaml controlled)
    use_plane_path = args_opt.get('use_plane_path', False)
    use_plane_cross_attention = args_opt.get('use_plane_cross_attention', True)
    plane_cross_num_heads = args_opt.get('plane_cross_num_heads', 8)
    plane_cross_dropout = args_opt.get('plane_cross_dropout', 0.0)

    plane_gate_pool_num_queries = args_opt.get('plane_gate_pool_num_queries', 1)
    plane_gate_pool_num_heads = args_opt.get('plane_gate_pool_num_heads', 8)
    plane_gate_pool_depth = args_opt.get('plane_gate_pool_depth', 2)
    plane_gate_pool_mlp_ratio = args_opt.get('plane_gate_pool_mlp_ratio', 4.0)
    plane_gate_pool_dropout = args_opt.get('plane_gate_pool_dropout', 0.1)

    # -- Token->3D CNN classifier head (optional)
    use_token_3d_cnn_head = args_opt.get('use_token_3d_cnn_head', False)
    token_3d_use_coords = args_opt.get('token_3d_use_coords', True)
    token_3d_stem_channels = args_opt.get('token_3d_stem_channels', 128)
    token_3d_stage_channels = args_opt.get('token_3d_stage_channels', [128, 256, 384])
    token_3d_stage_depths = args_opt.get('token_3d_stage_depths', [1, 1, 1])
    token_3d_stage_strides = args_opt.get('token_3d_stage_strides', [1, 2, 2])
    token_3d_dropout = args_opt.get('token_3d_dropout', 0.1)
    token_3d_grid_size = args_opt.get('token_3d_grid_size', None)

    # -- MRNet triple plane (3 encoders + 3 classifiers, avg logits)
    use_mrnet_triple_path = args_opt.get('use_mrnet_triple_path', False)
    # -- MRNet per-task per-plane (3 views × N defects independent classifiers)
    use_mrnet_triple_per_task = args_opt.get('use_mrnet_triple_per_task', False)
    use_feature_concat = args_opt.get('use_feature_concat', False)
    mrnet_data_path = args_data.get('dataset', None)

    # -- Token head pooling mode for MultiHeadAttentiveClassifier
    classifier_pool_mode = str(args_opt.get('classifier_pool_mode', 'Null')).lower()
    if classifier_pool_mode not in ('attentive', 'gap', 'head_cls', 'swin_default', 'null'):
        raise ValueError(f"optimization.classifier_pool_mode: {classifier_pool_mode} 必须为 'attentive' | 'gap' | 'head_cls' | 'swin_default' | 'Null'")

    # attentive mode params
    attn_num_queries = args_opt.get('attn_num_queries', 1)
    attn_num_heads = args_opt.get('attn_num_heads', 8)
    attn_mlp_ratio = args_opt.get('attn_mlp_ratio', 4.0)
    attn_depth = args_opt.get('attn_depth', 2)

    # head-cls mode params
    head_cls_num_heads = args_opt.get('head_cls_num_heads', 8)
    head_cls_mlp_ratio = args_opt.get('head_cls_mlp_ratio', 4.0)
    head_cls_depth = args_opt.get('head_cls_depth', 4)

    # -- BALANCED SOFTMAX / COSFACE (LMCL-style)
    use_balanced_softmax = args_opt.get('use_balanced_softmax', False)
    balanced_softmax_momentum = args_opt.get('balanced_softmax_momentum', 0.95)
    balanced_softmax_eps = args_opt.get('balanced_softmax_eps', 1e-6)

    use_cosface = args_opt.get('use_cosface', False)
    cosface_s = args_opt.get('cosface_s', 30.0)
    cosface_m = args_opt.get('cosface_m', 0.35)

    logger.info(
        f'Loss options -> balanced_softmax: {use_balanced_softmax}, '
        f'cosface: {use_cosface} (s={cosface_s}, m={cosface_m})'
    )
    logger.info(
        f'Plane path -> enable: {use_plane_path}, '
        f'cross_attention: {use_plane_cross_attention}, '
        f'cross_heads: {plane_cross_num_heads}'
    )
    logger.info(
        f'Layer-wise LR -> enable: {use_layerwise_lr}, '
        f'encoder_lr: {encoder_lr}, pagca_lr: {pagca_lr}, classifier_lr: {classifier_lr}'
    )
    logger.info(
        f'Token-3D head -> enable: {use_token_3d_cnn_head}, '
        f'use_coords: {token_3d_use_coords}, stem: {token_3d_stem_channels}, '
        f'stage_channels: {token_3d_stage_channels}'
        f'mrnet_triple_per_task: {use_mrnet_triple_per_task}, 3_view_concat: {use_feature_concat}'
    )
    if classifier_pool_mode != 'Null':
        logger.info(
            f'Token head pooling -> mode: {classifier_pool_mode}, '
            f'attn(q={attn_num_queries}, h={attn_num_heads}, mlp={attn_mlp_ratio}, d={attn_depth}), '
            f'head_cls(h={head_cls_num_heads}, mlp={head_cls_mlp_ratio}, d={head_cls_depth})'
        )
    else:
        logger.info('Ensure classifier is configured')

    if use_plane_path and use_token_3d_cnn_head:
        raise ValueError('use_plane_path 与 use_token_3d_cnn_head 不能同时开启，请二选一')
    if use_plane_path and use_mrnet_triple_path:
        raise ValueError('use_plane_path 与 use_mrnet_triple_path 不能同时开启，请二选一')
    if use_mrnet_triple_path and use_token_3d_cnn_head:
        raise ValueError('use_mrnet_triple_path 与 use_token_3d_cnn_head 不能同时开启，请二选一')
    if use_mrnet_triple_per_task and use_token_3d_cnn_head:
        raise ValueError('use_mrnet_triple_per_task 与 use_token_3d_cnn_head 不能同时开启，请二选一')
    if use_mrnet_triple_per_task and use_plane_path:
        raise ValueError('use_mrnet_triple_per_task 与 use_plane_path 不能同时开启，请二选一')
    if use_mrnet_triple_path and use_mrnet_triple_per_task:
        raise ValueError('use_mrnet_triple_path 与 use_mrnet_triple_per_task 不能同时开启，请二选一')
    if use_feature_concat and not use_mrnet_triple_per_task:
        raise ValueError('use_feature_concat 仅可与 use_mrnet_triple_per_task 搭配使用')
    if use_swin_backbone and unfreeze == -1:
        raise ValueError('Swin backbone does not support mask aggregation mode (unfreeze = -1)')
    
    # -- EXPERIMENT-ID/TAG (optional)
    resume_checkpoint = args_eval.get('resume_checkpoint', False) or resume_preempt
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
        logger.warning('No GPU detected, using CPU. This will be slow!')
    else:
        device = torch.device('cuda:0')
        torch.cuda.set_device(device)
        print(f"Using GPU: {torch.cuda.get_device_name(device)}")

    world_size, rank = init_distributed()
    logger.info(f'Initialized (rank/world-size) {rank}/{world_size}')

    # -- log/checkpointing paths
    if dataset_npz is not None:
        dataset_name = os.path.splitext(os.path.basename(dataset_npz))[0]
    elif mrnet_data_path is not None:
        dataset_name = os.path.basename(mrnet_data_path.rstrip('/'))
    else:
        dataset_name = 'unknown'
    folder = os.path.join(pretrain_folder, dataset_name)
    if eval_tag is not None:
        folder = os.path.join(folder, eval_tag)
    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)
    log_file = os.path.join(folder, f'{tag}_r{rank}.csv')
    latest_path = os.path.join(folder, f'{tag}-latest.pth.tar')
    best_path = os.path.join(folder, f'{tag}-best.pth.tar')
    best_auc_path = os.path.join(folder, f'{tag}-best-auc.pth.tar')

    # -- make csv_logger
    if rank == 0:
        num_tasks = len(defect) if isinstance(defect, (list, tuple)) else 1
        fields = [
            ('%d', 'epoch'),
            ('%.3f', 'train_loss'),
            ('%.3f', 'val_loss'),
            ('%.3f', 'train_scl_loss'),
            ('%.3f', 'val_scl_loss'),
            ('%.4f', 'train_offdiag_cos_mean'),
            ('%.4f', 'train_offdiag_cos_max'),
            ('%.4f', 'train_fusion_entropy_norm'),
            ('%.4f', 'train_fusion_attn_max'),
            ('%.4f', 'val_offdiag_cos_mean'),
            ('%.4f', 'val_offdiag_cos_max'),
            ('%.4f', 'val_fusion_entropy_norm'),
            ('%.4f', 'val_fusion_attn_max'),
        ]
        if use_plane_path:
            fields.extend([
                ('%.4f', 'train_w_a'),
                ('%.4f', 'train_w_c'),
                ('%.4f', 'train_w_s'),
                ('%.4f', 'val_w_a'),
                ('%.4f', 'val_w_c'),
                ('%.4f', 'val_w_s'),
            ])
        for i in range(num_tasks):
            t = i + 1
            fields.extend([
            ('%.3f', f'train_precision_t{t}'),
            ('%.3f', f'train_recall_t{t}'),
            ('%.3f', f'train_f1_t{t}'),
            ('%.3f', f'train_top1_t{t}'),
            ('%.3f', f'train_top2_t{t}'),
            ('%.3f', f'val_precision_t{t}'),
            ('%.3f', f'val_recall_t{t}'),
            ('%.3f', f'val_f1_t{t}'),
            ('%.3f', f'val_top1_t{t}'),
            ('%.3f', f'val_top2_t{t}'),
            ('%.4f', f'roc_auc_t{t}'),
            ])
            for c in range(defect_classes[i]):
                c_id = c + 1
                fields.extend([
                    ('%.3f', f'train_precision_t{t}_c{c_id}'),
                    ('%.3f', f'train_recall_t{t}_c{c_id}'),
                    ('%.3f', f'train_f1_t{t}_c{c_id}'),
                    ('%.3f', f'val_precision_t{t}_c{c_id}'),
                    ('%.3f', f'val_recall_t{t}_c{c_id}'),
                    ('%.3f', f'val_f1_t{t}_c{c_id}'),
                ])
        csv_logger = CSVLogger(log_file, *fields)

    # -- frozen pretrained model
    encoder, predictor = init_model_frozen(
        device,
        pretrained=pretrained_path,    # pretrained_path
        load_pretrained_ckpt=load_pretrained_ckpt,
        model_name=model_name,
        patch_size=patch_size,
        crop_size=args_pretrain.get('crop_size', 64),
        num_slices=args_pretrain.get('num_slices', 64),
        in_chans=args_pretrain.get('in_chans', 1),
        depth_patch_size=depth_patch_size,
        use_sdpa=use_sdpa,
        use_SiLU=use_SiLU,
        tight_SiLU=tight_SiLU,
        uniform_power=uniform_power,
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
    target_encoder = copy.deepcopy(encoder) # 直接使用健康预训练权重，可能无法忠实表达实际特征
    encoder_embed_dim = encoder.embed_dim


    if unfreeze == -1:
        # 原始：偏差模型
        deviation_model = MaskAggregation(  # 联同predictor、target encoder封装
            device,
            encoder,
            predictor,
            target_encoder,
            mask_size=ast.literal_eval(mask_size),
        ).to(device)
    elif unfreeze == 0:
        # 实验1：不作差直接用特征
        # 全冻结时所有 encoder 副本权重相同且永不更新，无需 TripleEncoderWrapper
        deviation_model = target_encoder.to(device)
        deviation_model.eval()
    elif unfreeze > 0:
        # 实验2：解冻部分encoder
        # 另：
        # 1.公用部分第2、3行被注释，其他实验应解除注释！！！
        # 2.init_opt调用传入extra_modules参数，其他实验置为None！！！
        # 3.run_one_epoch里deviation model注释torch.no_grad(),其他实验应解除注释！！！
        # 4.if training里裁剪梯度部分同上！！！

        def _apply_unfreeze(enc):
            """对单个 encoder 应用 freeze/unfreeze 逻辑，返回修改后的 enc"""
            for p in enc.parameters():
                p.requires_grad = False
            if unfreeze_all:
                for p in enc.parameters():
                    p.requires_grad = True
            else:
                num_unfreeze = unfreeze
                for blk in enc.blocks[-num_unfreeze:]:
                    for p in blk.parameters():
                        p.requires_grad = True
                for p in enc.norm.parameters():
                    p.requires_grad = True
            return enc

        if use_mrnet_triple_per_task:
            if len(views) == 3:
                # 三个平面用三个独立 encoder，各自应用相同的冻结/解冻策略
                enc_a = _apply_unfreeze(copy.deepcopy(target_encoder))
                enc_c = _apply_unfreeze(copy.deepcopy(target_encoder))
                enc_s = _apply_unfreeze(copy.deepcopy(target_encoder))
                deviation_model = TripleEncoderWrapper(enc_a, enc_c, enc_s).to(device)
            else:
                # 任意数量平面：用 MultiEncoderWrapper 灵活封装
                enc_dict = {}
                for vname in views:
                    enc_dict[vname] = _apply_unfreeze(copy.deepcopy(target_encoder))
                deviation_model = MultiEncoderWrapper(enc_dict).to(device)
        else:
            _apply_unfreeze(target_encoder)
            deviation_model = target_encoder.to(device)
    else:
        raise ValueError("unfreeze 参数设置错误！应为 -1, 0 或正整数")

    # 公用部分
    # deviation_model = DistributedDataParallel(deviation_model, static_graph=True)
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        deviation_model = DistributedDataParallel(deviation_model, static_graph=True)
    else:
        logger.info('Distributed not initialized -> running deviation_model in single-process mode')

    if unfreeze <= 0:
        for p in deviation_model.parameters():
            p.requires_grad = False   


    # -- init classifier
    if use_mrnet_triple_per_task:
        classifier = MRNetTriplePlanePerTaskClassifier(
            embed_dim=encoder_embed_dim,
            defect_classes=defect_classes,
            defect_names=defect,
            patch_size=patch_size,
            tubelet_size=depth_patch_size,
            grid_size=token_3d_grid_size,
            use_coord_channels=token_3d_use_coords,
            stem_channels=token_3d_stem_channels,
            stage_channels=token_3d_stage_channels,
            stage_depths=token_3d_stage_depths,
            stage_strides=token_3d_stage_strides,
            dropout=token_3d_dropout,
            use_feature_concat=use_feature_concat,
            plane_names=views,
        ).to(device)
    elif use_mrnet_triple_path:
        classifier = MRNetTriplePlaneClassifier(
            embed_dim=encoder_embed_dim,
            defect_classes=defect_classes,
            patch_size=patch_size,
            tubelet_size=depth_patch_size,
            grid_size=token_3d_grid_size,
            use_coord_channels=token_3d_use_coords,
            stem_channels=token_3d_stem_channels,
            stage_channels=token_3d_stage_channels,
            stage_depths=token_3d_stage_depths,
            stage_strides=token_3d_stage_strides,
            dropout=token_3d_dropout,
        ).to(device)
    elif use_plane_path:
        classifier = PlaneAwareMultiHeadClassifier(
            embed_dim=encoder_embed_dim,
            defect_classes=defect_classes,
            cross_num_heads=plane_cross_num_heads,
            cross_dropout=plane_cross_dropout,
            enable_plane_cross_attention=use_plane_cross_attention,
            gate_pool_num_queries=plane_gate_pool_num_queries,
            gate_pool_num_heads=plane_gate_pool_num_heads,
            gate_pool_mlp_ratio=plane_gate_pool_mlp_ratio,
            gate_pool_depth=plane_gate_pool_depth,
            gate_pool_dropout=plane_gate_pool_dropout,
        ).to(device)
    elif use_token_3d_cnn_head:
        classifier = TokenToVolume3DClassifier(
            embed_dim=encoder_embed_dim,
            defect_classes=defect_classes,
            patch_size=patch_size,
            tubelet_size=depth_patch_size,
            grid_size=token_3d_grid_size,
            use_coord_channels=token_3d_use_coords,
            stem_channels=token_3d_stem_channels,
            stage_channels=token_3d_stage_channels,
            stage_depths=token_3d_stage_depths,
            stage_strides=token_3d_stage_strides,
            dropout=token_3d_dropout,
        ).to(device)
    else:
        classifier = MultiHeadAttentiveClassifier(
            embed_dim=encoder_embed_dim,  # 根据backbone嵌入维度调整
            defect_classes=defect_classes,
            pool_mode=classifier_pool_mode,
            num_queries=attn_num_queries,
            num_heads=attn_num_heads,
            mlp_ratio=attn_mlp_ratio,
            depth=attn_depth,
            head_cls_num_heads=head_cls_num_heads,
            head_cls_mlp_ratio=head_cls_mlp_ratio,
            head_cls_depth=head_cls_depth,
        ).to(device)
    # classifier = DistributedDataParallel(classifier, static_graph=True)
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        classifier = DistributedDataParallel(classifier, static_graph=True)
    else:
        logger.info('Distributed not initialized -> running classifier in single-process mode')

    # -- dataloaders
    if dataset_type.lower() == 'mrnet':
        train_loader = make_mrnet_dataloader(
            data_path=mrnet_data_path,
            label_path=os.path.join(mrnet_data_path, 'train_label.xlsx'),
            transform=MakeMRNetTransforms,
            batch_size=batch_size,
            defect=defect,
            training=True,
            num_workers=num_workers,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
            pin_memory=pin_memory,
            drop_last=train_drop_last,
            views=views,
        )
        val_loader = make_mrnet_dataloader(
            data_path=mrnet_data_path,
            label_path=os.path.join(mrnet_data_path, 'val_label.xlsx'),
            transform=MakeMRNetTransforms,
            batch_size=batch_size,
            defect=defect,
            training=False,
            num_workers=num_workers,
            persistent_workers=persistent_workers,
            prefetch_factor=max(2, prefetch_factor // 2),
            pin_memory=pin_memory,
            drop_last=False,
            views=views,
        )
    elif dataset_type.lower() == 'medmnist3d':
        train_loader = make_medmnist3d_dataloader(
            npz_path=dataset_npz,
            split=split_train,
            transform=MakeMedMNIST3DTransforms,
            batch_size=batch_size,
            num_workers=num_workers,
            training=True,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
            pin_memory=pin_memory,
            drop_last=train_drop_last,
        )
        val_loader = make_medmnist3d_dataloader(
            npz_path=dataset_npz,
            split=split_val,
            transform=MakeMedMNIST3DTransforms,
            batch_size=batch_size,
            num_workers=num_workers,
            training=False,
            persistent_workers=persistent_workers,
            prefetch_factor=max(2, prefetch_factor // 2),
            pin_memory=pin_memory,
            drop_last=False,
        )
    elif dataset_type.lower() == 'mridataset':
        train_loader = make_dataloader(
            training=True,
            dataset_path=dataset_path,
            label_path=label_path,
            transform=MakeMRITransforms,
            batch_size=batch_size,
            defect=defect
        )

        val_loader = make_dataloader(
            training=False,
            dataset_path=dataset_path,
            label_path=label_path,
            transform=MakeMRITransforms,
            batch_size=batch_size,
            defect=defect
        )
    else:
        raise ValueError(f'Unsupported dataset type: {dataset_type}')
    ipe = len(train_loader)
    logger.info(f'Dataloader created... iterations per epoch: {ipe}')

    # per-task EMA priors, initialized uniform (for Balanced Softmax)
    class_prior_state = [
        torch.full((n_cls,), 1.0 / n_cls, device=device)
        for n_cls in defect_classes
    ]

    # optimizer, scaler, scheduler, wd_scheduler
    deviation_model_for_opt = deviation_model.module if hasattr(deviation_model, 'module') else deviation_model
    optimizer, scaler, scheduler, wd_scheduler = init_opt(
        classifier=classifier,
        wd=wd,
        start_lr=start_lr,
        ref_lr=lr,
        final_lr=final_lr,
        iterations_per_epoch=ipe,
        warmup=warmup,
        num_epochs=num_epochs,
        use_bfloat16=use_bfloat16,
        extra_modules={'deviation_model': deviation_model_for_opt} if unfreeze > 0 else None,
        use_layerwise_lr=use_layerwise_lr,
        use_plane_path=use_plane_path,
        encoder_lr=encoder_lr,
        pagca_lr=pagca_lr,
        classifier_lr=classifier_lr,
        )

    # -- load training checkpoint
    start_epoch = 0
    if resume_checkpoint:
        if stage is None:
            deviation_model, classifier, optimizer, scaler, start_epoch = load_checkpoint(
                device=device,
                r_path=latest_path,
                deviation_model=deviation_model,
                classifier=classifier,
                opt=optimizer,
                load_opt=True,
                scaler=scaler)
            for _ in range(start_epoch*ipe):
                scheduler.step()
                wd_scheduler.step()
        elif stage == 2:
            deviation_model, classifier, _, _, _ = load_checkpoint(
                device=device,
                r_path=latest_path,
                deviation_model=deviation_model,
                classifier=classifier,
                opt=optimizer,
                load_opt=False,
                scaler=scaler)
            print(f"Stage 2: Loaded model from {latest_path}, but not optimizer/scheduler states. Starting at epoch 0 with fresh optimization.")
        else:
            raise ValueError("Invalid stage specified for checkpoint loading.")

    # -- load best metrics from CSV when resuming (rank 0 only)
    _csv_best_top1 = -1.0
    _csv_best_epoch = -1
    _csv_best_auc = -1.0
    _csv_best_auc_epoch = -1
    if resume_checkpoint and rank == 0 and os.path.exists(log_file):
        try:
            with open(log_file, 'r', newline='') as f:
                reader = csv.DictReader(f)
                if reader.fieldnames:
                    for row in reader:
                        try:
                            ep = int(float(row['epoch']))
                            if 'val_top1_t1' in row:
                                top1 = float(row['val_top1_t1'])
                                if top1 > _csv_best_top1:
                                    _csv_best_top1 = top1
                                    _csv_best_epoch = ep
                            if 'roc_auc_t1' in row:
                                auc = float(row['roc_auc_t1'])
                                if auc > _csv_best_auc:
                                    _csv_best_auc = auc
                                    _csv_best_auc_epoch = ep
                        except (ValueError, KeyError):
                            continue
            logger.info(
                f'Loaded best metrics from CSV: best_val_top1={_csv_best_top1:.3f} '
                f'(epoch {_csv_best_epoch}), best_roc_auc={_csv_best_auc:.4f} '
                f'(epoch {_csv_best_auc_epoch})'
            )
        except Exception as e:
            logger.warning(f'Failed to load best metrics from CSV: {e}')

    for i,g in enumerate(optimizer.param_groups):
        total = sum(p.numel() for p in g['params'])
        print(f"group {i} requires_grad params:", total)

    def save_checkpoint(epoch):
        """
        可能有解冻部分编码器 总是保存全部enc ckpt

        """
        save_dict = {
            'deviation_model': deviation_model.state_dict(),
            'classifier': classifier.state_dict(),
            'opt': optimizer.state_dict(),
            'scaler': None if scaler is None else scaler.state_dict(),
            'epoch': epoch,
            'batch_size': batch_size,
            'world_size': world_size,
            'lr': lr
        }
        if rank == 0:
            torch.save(save_dict, latest_path)

    def save_best_checkpoint(epoch):
        """
        保存 val_top1 最优模型
        """
        save_dict = {
            'deviation_model': deviation_model.state_dict(),
            'classifier': classifier.state_dict(),
            'opt': optimizer.state_dict(),
            'scaler': None if scaler is None else scaler.state_dict(),
            'epoch': epoch,
            'batch_size': batch_size,
            'world_size': world_size,
            'lr': lr
        }
        if rank == 0:
            torch.save(save_dict, best_path)

    def save_best_auc_checkpoint(epoch):
        """
        保存 val_roc_auc 均值最优模型
        """
        save_dict = {
            'deviation_model': deviation_model.state_dict(),
            'classifier': classifier.state_dict(),
            'opt': optimizer.state_dict(),
            'scaler': None if scaler is None else scaler.state_dict(),
            'epoch': epoch,
            'batch_size': batch_size,
            'world_size': world_size,
            'lr': lr
        }
        if rank == 0:
            torch.save(save_dict, best_auc_path)

    # TRAIN LOOP
    cm_folder = os.path.join(folder, 'confusion_matrix')
    cm_best_folder = os.path.join(folder, 'confusion_matrix_best')
    best_val_top1 = _csv_best_top1 if resume_checkpoint and _csv_best_top1 >= 0 else -1.0
    best_val_auc = _csv_best_auc if resume_checkpoint and _csv_best_auc >= 0 else -1.0
    best_epoch = _csv_best_epoch if resume_checkpoint and _csv_best_epoch >= 0 else -1
    best_auc_epoch = _csv_best_auc_epoch if resume_checkpoint and _csv_best_auc_epoch >= 0 else -1
    if rank == 0:
        os.makedirs(cm_folder, exist_ok=True)
        os.makedirs(cm_best_folder, exist_ok=True)

    loss_curve_path = os.path.join(folder, 'train_val_loss_curve.png')
    acc_curve_path = os.path.join(folder, 'train_val_acc_curve.png')
    query_curve_path = os.path.join(folder, 'query_quickcheck_curve.png')

    for epoch in range(start_epoch, num_epochs):
        logger.info('Epoch %d' % (epoch + 1))
        train_f1_scores, train_precisions, train_recalls, train_f1_per_class, train_prec_per_class, train_rec_per_class, train_avg_loss, train_avg_scl_loss, train_top1_scores, train_top2_scores, train_plane_weights, train_query_metrics, train_confusion = run_one_epoch(
            device,
            training=True,
            deviation_model=deviation_model,
            classifier=classifier,
            scaler=scaler,
            optimizer=optimizer,
            scheduler=scheduler,
            wd_scheduler=wd_scheduler,
            data_loader=train_loader,
            use_bfloat16=use_bfloat16,
            defect=defect,
            defect_classes=defect_classes,
            use_balanced_softmax=use_balanced_softmax,
            class_prior_state=class_prior_state,
            balanced_softmax_momentum=balanced_softmax_momentum,
            balanced_softmax_eps=balanced_softmax_eps,
            use_cosface=use_cosface,
            cosface_s=cosface_s,
            cosface_m=cosface_m,
            update_priors=True,     # prior logits adjust
            scl_weight=scl_weight,
            scl_temperature=scl_temperature,
            unfreeze=unfreeze,
            use_plane_path=use_plane_path,
            use_token_3d_cnn_head=use_token_3d_cnn_head,
            use_mrnet_triple_path=use_mrnet_triple_path,
            use_mrnet_triple_per_task=use_mrnet_triple_per_task,
            views=views,
            collect_confusion=True,
        )
        val_f1_scores, val_precisions, val_recalls, val_f1_per_class, val_prec_per_class, val_rec_per_class, val_avg_loss, val_avg_scl_loss, val_top1_scores, val_top2_scores, val_plane_weights, val_query_metrics, val_confusion = run_one_epoch(
            device,
            training=False,
            deviation_model=deviation_model,
            classifier=classifier,
            scaler=scaler,
            optimizer=optimizer,
            scheduler=scheduler,
            wd_scheduler=wd_scheduler,
            data_loader=val_loader,
            use_bfloat16=use_bfloat16,
            defect=defect,
            defect_classes=defect_classes,
            use_balanced_softmax=use_balanced_softmax,
            class_prior_state=class_prior_state,
            balanced_softmax_momentum=balanced_softmax_momentum,
            balanced_softmax_eps=balanced_softmax_eps,
            use_cosface=use_cosface,
            cosface_s=cosface_s,
            cosface_m=cosface_m,
            update_priors=False,    # 不更新以免泄露
            scl_weight=scl_weight,
            scl_temperature=scl_temperature,
            unfreeze=unfreeze,
            use_plane_path=use_plane_path,
            use_token_3d_cnn_head=use_token_3d_cnn_head,
            use_mrnet_triple_path=use_mrnet_triple_path,
            use_mrnet_triple_per_task=use_mrnet_triple_per_task,
            views=views,
            collect_confusion=True,
        )

        for i, (tr_f1, tr_p, tr_r, va_f1, va_p, va_r) in enumerate(
            zip(train_f1_scores, train_precisions, train_recalls,
            val_f1_scores, val_precisions, val_recalls), start=1):
            logger.info('[%5d][Task %d] train P: %.3f%% R: %.3f%% F1: %.3f%% | val P: %.3f%% R: %.3f%% F1: %.3f%%'
                % (epoch + 1, i, tr_p * 100, tr_r * 100, tr_f1 * 100,
                   va_p * 100, va_r * 100, va_f1 * 100))
        

        save_checkpoint(epoch + 1)
        print('Checkpoint saved.')

        roc_aucs = None
        if rank == 0:
            num_tasks = len(defect) if isinstance(defect, (list, tuple)) else 1
            roc_aucs = [np.nan for _ in range(num_tasks)]
        if (epoch + 1) % 1 == 0 or (epoch + 1) == num_epochs or epoch == start_epoch:
            if rank == 0:
                roc_aucs = compute_roc_auc(
                    device=device,
                    deviation_model=deviation_model,
                    classifier=classifier,
                    data_loader=val_loader,
                    defect=defect,
                    defect_classes=defect_classes,
                    use_bfloat16=use_bfloat16,
                    unfreeze=unfreeze,
                    use_plane_path=use_plane_path,
                    use_token_3d_cnn_head=use_token_3d_cnn_head,
                    use_mrnet_triple_path=use_mrnet_triple_path,
                    use_mrnet_triple_per_task=use_mrnet_triple_per_task,
                    views=views,
                )
                for t_idx, auc in enumerate(roc_aucs, start=0):
                    if auc is None:
                        logger.info(f'ROC-AUC (Task {t_idx}): undefined (single-class present)')
                    else:
                        logger.info(f'ROC-AUC (Task {t_idx}): {auc:.4f}')
        if rank == 0:
            row = [
                epoch + 1,
                train_avg_loss,
                val_avg_loss,
                train_avg_scl_loss,
                val_avg_scl_loss,
                train_query_metrics['offdiag_cos_mean'],
                train_query_metrics['offdiag_cos_max'],
                train_query_metrics['fusion_entropy_norm'],
                train_query_metrics['fusion_attn_max'],
                val_query_metrics['offdiag_cos_mean'],
                val_query_metrics['offdiag_cos_max'],
                val_query_metrics['fusion_entropy_norm'],
                val_query_metrics['fusion_attn_max'],
            ]
            if use_plane_path:
                row.extend([
                    train_plane_weights[0],
                    train_plane_weights[1],
                    train_plane_weights[2],
                    val_plane_weights[0],
                    val_plane_weights[1],
                    val_plane_weights[2],
                ])
            for t in range(num_tasks):
                row.extend([
                    train_precisions[t],
                    train_recalls[t],
                    train_f1_scores[t],
                    train_top1_scores[t],
                    train_top2_scores[t],
                    val_precisions[t],
                    val_recalls[t],
                    val_f1_scores[t],
                    val_top1_scores[t],
                    val_top2_scores[t],
                    roc_aucs[t] if roc_aucs[t] is not None else np.nan,
                ])
                for c in range(defect_classes[t]):
                    row.extend([
                        train_prec_per_class[t][c],
                        train_rec_per_class[t][c],
                        train_f1_per_class[t][c],
                        val_prec_per_class[t][c],
                        val_rec_per_class[t][c],
                        val_f1_per_class[t][c],
                    ])
            csv_logger.log(*row)
            if plot_loss_curve:
                try:
                    plot_train_val_loss_from_csv(
                        csv_path=log_file,
                        out_path=loss_curve_path,
                    )
                except Exception as e:
                    logger.warning(f'Failed to plot loss curve from CSV: {e}')
                try:
                    plot_train_val_acc_from_csv(
                        csv_path=log_file,
                        out_path=acc_curve_path,
                    )
                except Exception as e:
                    logger.warning(f'Failed to plot acc curve from CSV: {e}')
                try:
                    plot_query_quickcheck_from_csv(
                        csv_path=log_file,
                        out_path=query_curve_path,
                    )
                except Exception as e:
                    logger.warning(f'Failed to plot query quick-check curve from CSV: {e}')

        if rank == 0:
            if train_confusion is not None:
                train_labels, train_preds = train_confusion
                save_confusion_matrices(
                    train_labels,
                    train_preds,
                    defect_classes,
                    cm_folder,
                    prefix='train',
                )
            if val_confusion is not None:
                val_labels, val_preds = val_confusion
                save_confusion_matrices(
                    val_labels,
                    val_preds,
                    defect_classes,
                    cm_folder,
                    prefix='val',
                )

            if val_top1_scores is not None and len(val_top1_scores) > 0:
                val_top1_mean = float(np.mean(val_top1_scores))
                if val_top1_mean > best_val_top1:
                    best_val_top1 = val_top1_mean
                    best_epoch = epoch + 1
                    save_best_checkpoint(best_epoch)
                    for fname in os.listdir(cm_best_folder):
                        fpath = os.path.join(cm_best_folder, fname)
                        if os.path.isfile(fpath):
                            os.remove(fpath)
                    if train_confusion is not None:
                        train_labels, train_preds = train_confusion
                        save_confusion_matrices(
                            train_labels,
                            train_preds,
                            defect_classes,
                            cm_best_folder,
                            prefix=f'best_train_ep{best_epoch}',
                        )
                    if val_confusion is not None:
                        val_labels, val_preds = val_confusion
                        save_confusion_matrices(
                            val_labels,
                            val_preds,
                            defect_classes,
                            cm_best_folder,
                            prefix=f'best_val_ep{best_epoch}',
                        )

            if roc_aucs is not None:
                valid_aucs = [auc for auc in roc_aucs if auc is not None and not np.isnan(auc)]
                if len(valid_aucs) > 0:
                    val_auc_mean = float(np.mean(valid_aucs))
                    if val_auc_mean > best_val_auc:
                        best_val_auc = val_auc_mean
                        best_auc_epoch = epoch + 1
                        save_best_auc_checkpoint(best_auc_epoch)


def run_one_epoch(
    device,
    training,
    deviation_model,
    classifier,
    scaler,
    optimizer,
    scheduler,
    wd_scheduler,
    data_loader,
    use_bfloat16,
    defect,
    defect_classes,
    use_balanced_softmax,
    class_prior_state,
    balanced_softmax_momentum,
    balanced_softmax_eps,
    use_cosface,
    cosface_s,
    cosface_m,
    update_priors,
    scl_weight,
    scl_temperature,
    unfreeze,
    use_plane_path,
    use_token_3d_cnn_head,
    use_mrnet_triple_path=False,
    use_mrnet_triple_per_task=False,
    views=None,
    collect_confusion=False,
):

    epoch_start_time = time.time()

    classifier.train(mode=training)

    criterion = torch.nn.CrossEntropyLoss()
    # criterions = []
    # for i, n_cls in enumerate(defect_classes):
    #     # alpha 可先不加，后面再说
    #     criterions.append(
    #     FocalLoss(alpha=None, gamma=2.0)
    # )

    # 为每个任务维护独立的 top1/top2 meter
    task_top1_meters = [AverageMeter() for _ in range(len(defect))]
    task_top2_meters = [AverageMeter() for _ in range(len(defect))]
    loss_meter = AverageMeter()
    scl_loss_meter = AverageMeter()
    plane_w_a_meter = AverageMeter()
    plane_w_c_meter = AverageMeter()
    plane_w_s_meter = AverageMeter()
    # query 同质化快速判据
    q_offdiag_mean_meter = AverageMeter()
    q_offdiag_max_meter = AverageMeter()
    q_fusion_entropy_norm_meter = AverageMeter()
    q_fusion_maxw_meter = AverageMeter()

    def _safe_item(v):
        if v is None:
            return None
        if torch.is_tensor(v):
            return float(v.detach().item())
        return float(v)

    def _update_query_meters_from_lists(homo_list, fusion_list, bsz):
        if homo_list is not None:
            vals = [_safe_item(d.get('offdiag_cos_mean')) for d in homo_list if d is not None]
            vals = [x for x in vals if x is not None]
            if len(vals) > 0:
                q_offdiag_mean_meter.update(float(np.mean(vals)), bsz)

            vals = [_safe_item(d.get('offdiag_cos_max')) for d in homo_list if d is not None]
            vals = [x for x in vals if x is not None]
            if len(vals) > 0:
                q_offdiag_max_meter.update(float(np.mean(vals)), bsz)

        if fusion_list is not None:
            vals = [_safe_item(d.get('attn_entropy_norm_mean')) for d in fusion_list if d is not None]
            vals = [x for x in vals if x is not None]
            if len(vals) > 0:
                q_fusion_entropy_norm_meter.update(float(np.mean(vals)), bsz)

            vals = [_safe_item(d.get('attn_max_mean')) for d in fusion_list if d is not None]
            vals = [x for x in vals if x is not None]
            if len(vals) > 0:
                q_fusion_maxw_meter.update(float(np.mean(vals)), bsz)

    def _update_query_meters(model_ref, bsz):
        # 1) 非 plane 路径: MultiHeadAttentiveClassifier 暴露聚合结果
        homo_list = getattr(model_ref, 'last_query_homogeneity', None)
        fusion_list = getattr(model_ref, 'last_query_fusion_metrics', None)
        if isinstance(homo_list, list) or isinstance(fusion_list, list):
            _update_query_meters_from_lists(homo_list, fusion_list, bsz)

        # 2) plane 路径: 从 PlaneGate 的 pool/fuse 取指标（若可用）
        plane_gate = getattr(model_ref, 'plane_gate', None)
        if plane_gate is not None:
            plane_homo = []
            plane_fusion = []
            for pool_name in ('pool_a', 'pool_c', 'pool_s'):
                pooler = getattr(plane_gate, pool_name, None)
                if pooler is not None:
                    plane_homo.append(getattr(pooler, 'last_query_homogeneity', None))
            for fuse_name in ('fuse_a', 'fuse_c', 'fuse_s'):
                fuse = getattr(plane_gate, fuse_name, None)
                if fuse is not None:
                    plane_fusion.append(getattr(fuse, 'last_attn_metrics', None))
            _update_query_meters_from_lists(plane_homo, plane_fusion, bsz)
    # 初始化每个任务的 TP、FP、FN
    # 按任务/类别初始化累计计数（每个epoch重置）
    total_tp = [ [0 for _ in range(defect_classes[i])] for i in range(len(defect_classes)) ]
    total_fp = [ [0 for _ in range(defect_classes[i])] for i in range(len(defect_classes)) ]
    total_fn = [ [0 for _ in range(defect_classes[i])] for i in range(len(defect_classes)) ]
    
    all_preds = None
    all_labels = None
    if collect_confusion:
        all_preds = [[] for _ in range(len(defect))]
        all_labels = [[] for _ in range(len(defect))]

    for itr, data in enumerate(data_loader):

        if training:
            scheduler.step()
            wd_scheduler.step()

        start_time=time.time()
        grad_ctx = torch.enable_grad() if training else torch.inference_mode()
        with grad_ctx:
            with torch.amp.autocast('cuda', dtype=torch.float16, enabled=use_bfloat16):

                # Load data and put on GPU
                if use_mrnet_triple_path or use_mrnet_triple_per_task:
                    vols = list(data[0])  # tuple of N view tensors, each (B, C, D, H, W)
                    labels = data[1]
                    batch_size = vols[0].shape[0]
                    vols = [v.to(device, dtype=torch.float16 if use_bfloat16 else torch.float32, non_blocking=True) for v in vols]
                    labels = labels.to(device, non_blocking=True)
                else:
                    img, labels = data
                    batch_size = img.shape[0]
                    img = img.to(device, dtype=torch.float16 if use_bfloat16 else torch.float32, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)

                # Forward and prediction
                if use_mrnet_triple_path or use_mrnet_triple_per_task:
                    # --- Encode N views ---
                    feats = []
                    dev_raw = deviation_model.module if hasattr(deviation_model, 'module') else deviation_model
                    if isinstance(dev_raw, TripleEncoderWrapper):
                        # TripleEncoderWrapper: exactly 3 views in one call
                        if unfreeze <= 0:
                            with torch.no_grad():
                                f_a, f_c, f_s = dev_raw(vols[0], vols[1], vols[2])
                        else:
                            deviation_model.train(mode=training)
                            f_a, f_c, f_s = dev_raw(vols[0], vols[1], vols[2])
                        feats = [f_a, f_c, f_s]
                    elif isinstance(dev_raw, MultiEncoderWrapper):
                        # MultiEncoderWrapper: N views, call by name
                        for i, vname in enumerate(views):
                            if unfreeze <= 0:
                                with torch.no_grad():
                                    feats.append(dev_raw(vname, vols[i]))
                            else:
                                deviation_model.train(mode=training)
                                feats.append(dev_raw(vname, vols[i]))
                    else:
                        # Single encoder: call N times
                        for v in vols:
                            if unfreeze <= 0:
                                with torch.no_grad():
                                    feats.append(deviation_model(v))
                            else:
                                deviation_model.train(mode=training)
                                feats.append(deviation_model(v))

                    if unfreeze >= 0:
                        feats = [F.layer_norm(f, (f.size(-1),)) for f in feats]

                    tokens_dict = {vname: feat for vname, feat in zip(views, feats)}
                    outputs = classifier(tokens_dict, volume_shape=tuple(vols[0].shape[2:]))
                elif use_plane_path:
                    axial, coronal, sagittal = build_three_plane_volumes(img)
                    if unfreeze <= 0:
                        deviation_model.eval()
                        with torch.no_grad():
                            feat_a = deviation_model(axial)
                            feat_c = deviation_model(coronal)
                            feat_s = deviation_model(sagittal)
                    else:
                        deviation_model.train(mode=training)
                        feat_a = deviation_model(axial)
                        feat_c = deviation_model(coronal)
                        feat_s = deviation_model(sagittal)

                    if unfreeze >= 0:
                        feat_a = F.layer_norm(feat_a, (feat_a.size(-1),))
                        feat_c = F.layer_norm(feat_c, (feat_c.size(-1),))
                        feat_s = F.layer_norm(feat_s, (feat_s.size(-1),))

                    devidation = (feat_a + feat_c + feat_s) / 3.0
                    outputs = classifier((feat_a, feat_c, feat_s))

                    model_ref = classifier.module if hasattr(classifier, 'module') else classifier
                    plane_weights = getattr(model_ref, 'last_plane_weights', None)
                    if plane_weights is not None:
                        plane_weights_mean = plane_weights.mean(dim=0)
                        plane_w_a_meter.update(plane_weights_mean[0].item(), batch_size)
                        plane_w_c_meter.update(plane_weights_mean[1].item(), batch_size)
                        plane_w_s_meter.update(plane_weights_mean[2].item(), batch_size)
                else:
                    if unfreeze <= 0:
                        deviation_model.eval()
                        with torch.no_grad():
                            devidation = deviation_model(img)
                    else:
                        deviation_model.train(mode=training)
                        devidation = deviation_model(img)

                    # 单用encoder不作差时,外部加layer norm,使用mask aggregation注释掉
                    if unfreeze >= 0:
                        devidation = F.layer_norm(devidation, (devidation.size(-1),))

                    if use_token_3d_cnn_head:
                        outputs = classifier(devidation, volume_shape=tuple(img.shape[2:]))
                    else:
                        outputs = classifier(devidation)    # list 元素(B, num_classes_i)

                model_ref = classifier.module if hasattr(classifier, 'module') else classifier
                _update_query_meters(model_ref, batch_size)

                devidation_time = time.time()-start_time
                classifier_time = time.time()-start_time - devidation_time


            # Compute loss with optional CosFace + Balanced Softmax
            total_ce = 0.0
            for i in range(len(defect)):
                logits = outputs[i]
                label = labels[:, i]

                if use_balanced_softmax and update_priors:
                    batch_counts = torch.bincount(label, minlength=defect_classes[i]).to(device=logits.device, dtype=logits.dtype)
                    batch_probs = batch_counts / batch_size
                    class_prior_state[i] = (
                        balanced_softmax_momentum * class_prior_state[i]
                        + (1.0 - balanced_softmax_momentum) * batch_probs
                    )

                logits_for_loss = logits
                if training and use_cosface:
                    logits_for_loss = apply_cosface_margin(logits_for_loss, label, s=cosface_s, m=cosface_m)
                if training and use_balanced_softmax:
                    logits_for_loss = apply_balanced_softmax(logits_for_loss, class_prior_state[i], eps=balanced_softmax_eps)

                loss_i = criterion(logits_for_loss, label)
                total_ce += loss_i

            ce_loss = total_ce / len(defect)

            scl_loss = 0.0
            if scl_weight > 0:
                feats_supcon = F.normalize(devidation.float(), dim=-1)
                for i in range(len(defect)):
                    scl_loss += supervised_contrastive_loss(
                        feats_supcon,
                        labels[:, i],
                        temperature=scl_temperature,
                    )
                scl_loss = scl_loss / len(defect)

            loss = ce_loss + (scl_weight * scl_loss)
        loss_meter.update(loss.item(), batch_size)
        scl_loss_meter.update(scl_loss.item() if torch.is_tensor(scl_loss) else float(scl_loss), batch_size)
        # Focal loss
        # total_loss = 0.0
        # for i in range(len(defect)):
        #     logits = outputs[i] # (B, num_classes_i)
        #     label = labels[:, i] # (B,)
        #     loss_i = criterions[i](logits, label)
        #     print("pt mean:", torch.exp(-loss_i).mean().item())
        #     total_loss += loss_i
        # loss = total_loss / len(defect)
        
        if itr % 10 == 0:
            print(f"itr: {itr} loss: {loss.item()} forward model time: {devidation_time:.4f}s, Classifier time: {classifier_time:.4f}s")

        # calculate acc top1_meter
        with torch.no_grad():
            for i in range(len(defect)):
                # CosFace 里的 margin 依赖标签，只适合训练阶段注入；
                # 指标应尽量对齐推理路径，避免“训练技巧”污染评估。
                logits = outputs[i]     # 不使用logits_for_loss 
                label = labels[:, i]

                pred = torch.argmax(logits, dim=1)

                if collect_confusion:
                    all_preds[i].extend(pred.detach().cpu().tolist())
                    all_labels[i].extend(label.detach().cpu().tolist())

                # acc top1
                correct = pred.eq(label).sum().item()
                accuracy = 100.0 * correct / batch_size
                task_top1_meters[i].update(accuracy, batch_size)  # 更新对应任务的计量器

                # acc top2
                top2 = torch.topk(logits, k=2, dim=1).indices  # (B, 2)
                correct_top2 = top2.eq(label.unsqueeze(1)).any(dim=1).sum().item()
                accuracy_top2 = 100.0 * correct_top2 / batch_size
                task_top2_meters[i].update(accuracy_top2, batch_size)

                # F1 score components
                for c in range(defect_classes[i]):
                    total_tp[i][c] += ((pred == c) & (label == c)).sum().item()
                    total_fp[i][c] += ((pred == c) & (label != c)).sum().item()
                    total_fn[i][c] += ((pred != c) & (label == c)).sum().item()
                # if itr % 5 == 0:
                #     print(f"Task{i}:\nLogits:\n{logits}\nPred:{pred}\nGT:{label}\n---------------------")
        
        if training:
            if use_bfloat16:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                if unfreeze <= 0: # 实验0/1，仅裁剪classifier部分
                    torch.nn.utils.clip_grad_norm_(classifier.parameters(), 1.0)
                else: # 实验2，裁剪全部可训练参数：含解冻enc部分
                    torch.nn.utils.clip_grad_norm_(optimizer.param_groups[0]['params'], 1.0)
                    torch.nn.utils.clip_grad_norm_(optimizer.param_groups[1]['params'], 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if unfreeze <= 0: # 实验0/1，仅裁剪classifier部分
                    torch.nn.utils.clip_grad_norm_(classifier.parameters(), 1.0)
                else: # 实验2，裁剪全部可训练参数：含解冻enc部分
                    torch.nn.utils.clip_grad_norm_(optimizer.param_groups[0]['params'], 1.0)
                    torch.nn.utils.clip_grad_norm_(optimizer.param_groups[1]['params'], 1.0)
                optimizer.step()
            optimizer.zero_grad()

        if itr % 10 == 0:
            avgs = [m.avg for m in task_top1_meters]
            avgs_top2 = [m.avg for m in task_top2_meters]
            if training:
                phase = 'Train'
            else:
                phase = 'Val'
            logger.info(
            '%s itr[%5d] accs(top1):%s accs(top2):%s (loss: %.3f) [mem: %.2e]'
            % (
                phase,
                itr,
                ', '.join(f'{a:.3f}%' for a in avgs),
                ', '.join(f'{a:.3f}%' for a in avgs_top2),
                loss.item(),
                torch.cuda.max_memory_allocated() / 1024.**2
            )
            )

    # Compute F1 score for each task
    f1_scores = []
    precisions = []
    recalls = []
    f1_per_class_all = []
    precision_per_class_all = []
    recall_per_class_all = []
    
    for i in range(len(defect_classes)):
        f1_per_class = []
        precision_per_class = []
        recall_per_class = []
        
        for c in range(defect_classes[i]):
            tp = total_tp[i][c]; fp = total_fp[i][c]; fn = total_fn[i][c]
            denom = (2*tp + fp + fn)
            f1_c = (2*tp / denom) if denom > 0 else 0.0
            f1_per_class.append(f1_c)

            prec_c = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec_c = tp / (tp + fn) if (tp + fn) > 0 else 0.0

            precision_per_class.append(prec_c)
            recall_per_class.append(rec_c)

        f1_scores.append(sum(f1_per_class) / len(f1_per_class))
        precisions.append(sum(precision_per_class) / len(precision_per_class))
        recalls.append(sum(recall_per_class) / len(recall_per_class))
        f1_per_class_all.append(f1_per_class)
        precision_per_class_all.append(precision_per_class)
        recall_per_class_all.append(recall_per_class)

        # log
        logger.info(f"Task {i + 1} Precision:{precisions[i]:.3f}, Recall:{recalls[i]:.3f}, F1 Score: {f1_scores[i]:.3f}")

    top1_scores = [m.avg for m in task_top1_meters]
    top2_scores = [m.avg for m in task_top2_meters]
    plane_weights_avg = (
        plane_w_a_meter.avg if use_plane_path else np.nan,
        plane_w_c_meter.avg if use_plane_path else np.nan,
        plane_w_s_meter.avg if use_plane_path else np.nan,
    )

    if collect_confusion:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            world_size = torch.distributed.get_world_size()
            gathered_preds = [None for _ in range(world_size)]
            gathered_labels = [None for _ in range(world_size)]
            torch.distributed.all_gather_object(gathered_preds, all_preds)
            torch.distributed.all_gather_object(gathered_labels, all_labels)
            merged_preds = [[] for _ in range(len(defect))]
            merged_labels = [[] for _ in range(len(defect))]
            for gp, gl in zip(gathered_preds, gathered_labels):
                for t in range(len(defect)):
                    merged_preds[t].extend(gp[t])
                    merged_labels[t].extend(gl[t])
            all_preds = merged_preds
            all_labels = merged_labels

    epoch_time = time.time() - epoch_start_time
    logger.info(f"{'Train' if training else 'Val'} epoch time: {epoch_time:.2f}s")
    if use_plane_path:
        logger.info(
            f"{'Train' if training else 'Val'} plane weights avg: "
            f"w_a={plane_weights_avg[0]:.4f}, w_c={plane_weights_avg[1]:.4f}, w_s={plane_weights_avg[2]:.4f}"
        )

    logger.info(
        f"{'Train' if training else 'Val'} query quick-check: "
        f"offdiag_cos_mean={q_offdiag_mean_meter.avg:.4f}, "
        f"offdiag_cos_max={q_offdiag_max_meter.avg:.4f}, "
        f"fusion_entropy_norm={q_fusion_entropy_norm_meter.avg:.4f}, "
        f"fusion_attn_max={q_fusion_maxw_meter.avg:.4f}"
    )

    query_metrics_avg = {
        'offdiag_cos_mean': q_offdiag_mean_meter.avg,
        'offdiag_cos_max': q_offdiag_max_meter.avg,
        'fusion_entropy_norm': q_fusion_entropy_norm_meter.avg,
        'fusion_attn_max': q_fusion_maxw_meter.avg,
    }

    confusion_data = (all_labels, all_preds) if collect_confusion else None
    return f1_scores, precisions, recalls, f1_per_class_all, precision_per_class_all, recall_per_class_all, loss_meter.avg, scl_loss_meter.avg, top1_scores, top2_scores, plane_weights_avg, query_metrics_avg, confusion_data


def apply_balanced_softmax(logits, class_prior, eps=1e-6):
    """Balanced Softmax: add log class prior to logits before CE."""
    prior = class_prior.clamp_min(eps)
    return logits + prior.log().unsqueeze(0)


def apply_cosface_margin(logits, labels, s=30.0, m=0.35):
    """LMCL-style additive margin: target logit minus margin, then scaled by s."""
    one_hot = torch.zeros_like(logits)
    one_hot.scatter_(1, labels.view(-1, 1), 1.0)
    return s * (logits - m * one_hot)


def supervised_contrastive_loss(features, labels, temperature, eps=1e-6):
    """SupCon loss over a batch for a single task."""
    device = features.device

    # 若特征为 (B, T, D)，先做 token(patch)维池化 -> (B, D)
    if features.dim() > 2:
        features = features.mean(dim=1)

    batch_size = features.shape[0]
    if batch_size <= 1:
        return torch.tensor(0.0, device=device, dtype=features.dtype)

    # cosine similarities scaled by temperature
    sim = torch.matmul(features, features.T) / temperature
    sim = sim - sim.max(dim=1, keepdim=True)[0].detach()  # numerical stability

    # mask out self-comparisons
    logits_mask = torch.ones_like(sim) - torch.eye(batch_size, device=device)
    sim = sim + (-1e9) * (1.0 - logits_mask)

    labels = labels.contiguous().view(-1, 1)
    mask = torch.eq(labels, labels.T).float()
    mask = mask - torch.eye(batch_size, device=device)

    exp_sim = torch.exp(sim) * logits_mask
    log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + eps)

    mean_log_prob_pos = (mask * log_prob).sum(dim=1) / (mask.sum(dim=1) + eps)
    loss = -mean_log_prob_pos.mean()
    return loss


def compute_roc_auc(
    device,
    deviation_model,
    classifier,
    data_loader,
    defect,
    defect_classes,
    use_bfloat16,
    unfreeze,
    use_plane_path,
    use_token_3d_cnn_head,
    use_mrnet_triple_path=False,
    use_mrnet_triple_per_task=False,
    views=None,
):
    """Compute per-task ROC-AUC on a dataloader (typically validation)."""
    deviation_model.eval()
    classifier.eval()

    # Per-task collectors for labels and probabilities
    all_labels = [[] for _ in range(len(defect))]
    all_probs = [[] for _ in range(len(defect))]

    with torch.no_grad():
        for data in data_loader:
            if use_mrnet_triple_path or use_mrnet_triple_per_task:
                vols = list(data[0])  # tuple of N view tensors
                labels = data[1]
                vols = [v.to(device, dtype=torch.float16 if use_bfloat16 else torch.float32, non_blocking=True) for v in vols]
                labels = labels.to(device, non_blocking=True)
            else:
                img, labels = data
                img = img.to(device, dtype=torch.float16 if use_bfloat16 else torch.float32, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

            with torch.amp.autocast('cuda', dtype=torch.float16, enabled=use_bfloat16):
                if use_mrnet_triple_path or use_mrnet_triple_per_task:
                    # --- Encode N views ---
                    feats = []
                    dev_raw = deviation_model.module if hasattr(deviation_model, 'module') else deviation_model
                    if isinstance(dev_raw, TripleEncoderWrapper):
                        f_a, f_c, f_s = dev_raw(vols[0], vols[1], vols[2])
                        feats = [f_a, f_c, f_s]
                    elif isinstance(dev_raw, MultiEncoderWrapper):
                        for i, vname in enumerate(views):
                            feats.append(dev_raw(vname, vols[i]))
                    else:
                        for v in vols:
                            feats.append(deviation_model(v))
                    if unfreeze >= 0:
                        feats = [F.layer_norm(f, (f.size(-1),)) for f in feats]
                    tokens_dict = {vname: feat for vname, feat in zip(views, feats)}
                    outputs = classifier(tokens_dict, volume_shape=tuple(vols[0].shape[2:]))
                elif use_plane_path:
                    axial, coronal, sagittal = build_three_plane_volumes(img)
                    feat_a = deviation_model(axial)
                    feat_c = deviation_model(coronal)
                    feat_s = deviation_model(sagittal)

                    if unfreeze >= 0:
                        feat_a = F.layer_norm(feat_a, (feat_a.size(-1),))
                        feat_c = F.layer_norm(feat_c, (feat_c.size(-1),))
                        feat_s = F.layer_norm(feat_s, (feat_s.size(-1),))

                    outputs = classifier((feat_a, feat_c, feat_s))
                else:
                    features = deviation_model(img)

                    if unfreeze >= 0:
                        features = F.layer_norm(features, (features.size(-1),))

                    if use_token_3d_cnn_head:
                        outputs = classifier(features, volume_shape=tuple(img.shape[2:]))
                    else:
                        outputs = classifier(features)

            for i in range(len(defect)):
                logits = outputs[i]
                label = labels[:, i]

                probs = F.softmax(logits, dim=1).float().detach().cpu()
                probs = probs / probs.sum(dim=1, keepdim=True).clamp_min(1e-12)
                all_probs[i].append(probs)
                all_labels[i].append(label.detach().cpu())

    roc_aucs = []
    for i in range(len(defect)):
        if len(all_labels[i]) == 0:
            raise ValueError(f"No labels found for defect index {i}")

        labels_i = torch.cat(all_labels[i]).numpy()
        probs_i = torch.cat(all_probs[i]).numpy()
        
        try:
            if defect_classes[i] == 2:
                # Use positive-class probability for binary ROC-AUC
                roc_auc = roc_auc_score(labels_i, probs_i[:, 1])
                print(f"Task {i + 1} binary AUC: {roc_auc:.4f}")

            else:
                roc_auc = roc_auc_score(
                    labels_i,
                    probs_i,
                    multi_class='ovr',
                    average='macro'
                )
                print(f"Task {i + 1} ovr-macro AUC: {roc_auc:.4f}")
        except ValueError:
            # Happens when only one class present in labels
            raise ValueError(f"Only one class present in labels for defect index {i}, ROC-AUC undefined.")

        roc_aucs.append(roc_auc)

    return roc_aucs


def save_confusion_matrices(labels_list, preds_list, defect_classes, out_dir, prefix):
    for t in range(len(defect_classes)):
        labels_t = labels_list[t]
        preds_t = preds_list[t]

        if labels_t is None or preds_t is None or len(labels_t) == 0:
            continue

        cm = confusion_matrix(
            labels_t,
            preds_t,
            labels=list(range(defect_classes[t]))
        )

        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        ax.set(
            xticks=np.arange(cm.shape[1]),
            yticks=np.arange(cm.shape[0]),
            xticklabels=list(range(defect_classes[t])),
            yticklabels=list(range(defect_classes[t])),
            xlabel='Predicted label',
            ylabel='True label',
            title=f'Confusion Matrix ({prefix}) - Task {t + 1}'
        )

        thresh = cm.max() / 2.0 if cm.size > 0 else 0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(
                    j, i, format(cm[i, j], 'd'),
                    ha='center', va='center',
                    color='white' if cm[i, j] > thresh else 'black'
                )

        fig.tight_layout()
        out_path = os.path.join(out_dir, f'{prefix}_task{t + 1}.png')
        fig.savefig(out_path, dpi=150)
        plt.close(fig)


def plot_train_val_loss_from_csv(csv_path, out_path):
    """
    从训练 CSV 中读取 epoch/train_loss/val_loss，并绘制损失曲线。
    - 会自动跳过非数据行（例如重复表头）
    - 每次调用覆盖写出同一张图，便于观察训练进度
    """
    if not os.path.exists(csv_path):
        return

    epochs, train_losses, val_losses = [], [], []

    with open(csv_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return

        required = {'epoch', 'train_loss', 'val_loss'}
        if not required.issubset(set(reader.fieldnames)):
            return

        for row in reader:
            try:
                ep = int(float(row['epoch']))
                tr = float(row['train_loss'])
                va = float(row['val_loss'])
            except Exception:
                # 跳过重复表头/空行/异常行
                continue

            if not (np.isfinite(tr) and np.isfinite(va)):
                continue

            epochs.append(ep)
            train_losses.append(tr)
            val_losses.append(va)

    if len(epochs) == 0:
        return

    # 若有重复 epoch（如恢复训练多次写入），保留最后一次
    last_idx = {}
    for i, ep in enumerate(epochs):
        last_idx[ep] = i
    ordered_epochs = sorted(last_idx.keys())
    x = ordered_epochs
    y_train = [train_losses[last_idx[ep]] for ep in ordered_epochs]
    y_val = [val_losses[last_idx[ep]] for ep in ordered_epochs]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(x, y_train, marker='o', linewidth=1.8, markersize=3.5, label='train_loss')
    ax.plot(x, y_val, marker='o', linewidth=1.8, markersize=3.5, label='val_loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Train/Val Loss Curve')
    ax.grid(True, linestyle='--', alpha=0.35)
    ax.legend(loc='best')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_train_val_acc_from_csv(csv_path, out_path):
    """
    从训练 CSV 中读取 top1 acc 并绘制 train/val 准确率曲线。
    - 自动识别所有任务列：train_top1_tX / val_top1_tX
    - 会额外绘制所有任务的均值曲线（若存在至少一个任务）
    - 每次调用覆盖写出同一张图
    """
    if not os.path.exists(csv_path):
        return

    with open(csv_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return

        fieldnames = list(reader.fieldnames)
        if 'epoch' not in fieldnames:
            return

        train_cols = sorted(
            [k for k in fieldnames if k.startswith('train_top1_t')],
            key=lambda x: int(x.split('train_top1_t')[-1]) if x.split('train_top1_t')[-1].isdigit() else 10**9
        )
        val_cols = sorted(
            [k for k in fieldnames if k.startswith('val_top1_t')],
            key=lambda x: int(x.split('val_top1_t')[-1]) if x.split('val_top1_t')[-1].isdigit() else 10**9
        )

        # 仅保留 train/val 成对存在的任务
        task_ids = []
        for tc in train_cols:
            tid = tc.replace('train_top1_t', '')
            vc = f'val_top1_t{tid}'
            if vc in val_cols:
                task_ids.append(tid)

        if len(task_ids) == 0:
            return

        rows = []
        for row in reader:
            try:
                ep = int(float(row['epoch']))
                tr_vals = [float(row[f'train_top1_t{tid}']) for tid in task_ids]
                va_vals = [float(row[f'val_top1_t{tid}']) for tid in task_ids]
            except Exception:
                continue

            vals = np.array(tr_vals + va_vals, dtype=float)
            if not np.all(np.isfinite(vals)):
                continue

            rows.append((ep, tr_vals, va_vals))

    if len(rows) == 0:
        return

    # 若有重复 epoch（如恢复训练多次写入），保留最后一次
    last_by_epoch = {}
    for i, (ep, _, _) in enumerate(rows):
        last_by_epoch[ep] = i

    ordered_epochs = sorted(last_by_epoch.keys())
    x = ordered_epochs

    tr_by_task = {tid: [] for tid in task_ids}
    va_by_task = {tid: [] for tid in task_ids}

    for ep in ordered_epochs:
        _, tr_vals, va_vals = rows[last_by_epoch[ep]]
        for j, tid in enumerate(task_ids):
            tr_by_task[tid].append(tr_vals[j])
            va_by_task[tid].append(va_vals[j])

    # 均值曲线
    tr_mean = [float(np.mean([tr_by_task[tid][k] for tid in task_ids])) for k in range(len(x))]
    va_mean = [float(np.mean([va_by_task[tid][k] for tid in task_ids])) for k in range(len(x))]

    fig, ax = plt.subplots(figsize=(8, 5.5))

    # 先画每任务（浅色细线）
    for tid in task_ids:
        ax.plot(x, tr_by_task[tid], linewidth=1.1, alpha=0.45, linestyle='--', label=f'train_top1_t{tid}')
        ax.plot(x, va_by_task[tid], linewidth=1.1, alpha=0.45, linestyle='-', label=f'val_top1_t{tid}')

    # 再画均值（深色粗线）
    ax.plot(x, tr_mean, marker='o', linewidth=2.2, markersize=3.8, linestyle='--', label='train_top1_mean')
    ax.plot(x, va_mean, marker='o', linewidth=2.2, markersize=3.8, linestyle='-', label='val_top1_mean')

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Top1 Accuracy (%)')
    ax.set_title('Train/Val Accuracy Curve (Top1)')
    ax.grid(True, linestyle='--', alpha=0.35)

    # 图例过多时放到图外
    if len(task_ids) > 3:
        ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0), borderaxespad=0., fontsize=8)
        fig.tight_layout(rect=[0, 0, 0.82, 1])
    else:
        ax.legend(loc='best', fontsize=8)
        fig.tight_layout()

    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_query_quickcheck_from_csv(csv_path, out_path):
    """
    从训练 CSV 中读取 query 快速判据并绘图。
    需要字段：
      - train/val_offdiag_cos_mean
      - train/val_offdiag_cos_max
      - train/val_fusion_entropy_norm
      - train/val_fusion_attn_max
    """
    if not os.path.exists(csv_path):
        return

    epochs = []
    tr_off_m, tr_off_x, tr_ent, tr_amx = [], [], [], []
    va_off_m, va_off_x, va_ent, va_amx = [], [], [], []

    with open(csv_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return

        required = {
            'epoch',
            'train_offdiag_cos_mean', 'train_offdiag_cos_max', 'train_fusion_entropy_norm', 'train_fusion_attn_max',
            'val_offdiag_cos_mean', 'val_offdiag_cos_max', 'val_fusion_entropy_norm', 'val_fusion_attn_max',
        }
        if not required.issubset(set(reader.fieldnames)):
            return

        for row in reader:
            try:
                ep = int(float(row['epoch']))
                v = [
                    float(row['train_offdiag_cos_mean']),
                    float(row['train_offdiag_cos_max']),
                    float(row['train_fusion_entropy_norm']),
                    float(row['train_fusion_attn_max']),
                    float(row['val_offdiag_cos_mean']),
                    float(row['val_offdiag_cos_max']),
                    float(row['val_fusion_entropy_norm']),
                    float(row['val_fusion_attn_max']),
                ]
            except Exception:
                continue

            if not np.all(np.isfinite(np.array(v, dtype=float))):
                continue

            epochs.append(ep)
            tr_off_m.append(v[0]); tr_off_x.append(v[1]); tr_ent.append(v[2]); tr_amx.append(v[3])
            va_off_m.append(v[4]); va_off_x.append(v[5]); va_ent.append(v[6]); va_amx.append(v[7])

    if len(epochs) == 0:
        return

    # 若有重复 epoch（如恢复训练多次写入），保留最后一次
    last_idx = {}
    for i, ep in enumerate(epochs):
        last_idx[ep] = i
    ordered_epochs = sorted(last_idx.keys())

    x = ordered_epochs
    tr_off_m = [tr_off_m[last_idx[ep]] for ep in ordered_epochs]
    tr_off_x = [tr_off_x[last_idx[ep]] for ep in ordered_epochs]
    tr_ent = [tr_ent[last_idx[ep]] for ep in ordered_epochs]
    tr_amx = [tr_amx[last_idx[ep]] for ep in ordered_epochs]
    va_off_m = [va_off_m[last_idx[ep]] for ep in ordered_epochs]
    va_off_x = [va_off_x[last_idx[ep]] for ep in ordered_epochs]
    va_ent = [va_ent[last_idx[ep]] for ep in ordered_epochs]
    va_amx = [va_amx[last_idx[ep]] for ep in ordered_epochs]

    fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

    ax = axes[0]
    ax.plot(x, tr_off_m, marker='o', linewidth=1.6, markersize=3.2, label='train_offdiag_cos_mean')
    ax.plot(x, va_off_m, marker='o', linewidth=1.6, markersize=3.2, label='val_offdiag_cos_mean')
    ax.plot(x, tr_off_x, marker='o', linewidth=1.6, markersize=3.2, linestyle='--', label='train_offdiag_cos_max')
    ax.plot(x, va_off_x, marker='o', linewidth=1.6, markersize=3.2, linestyle='--', label='val_offdiag_cos_max')
    ax.set_ylabel('Cosine Similarity')
    ax.set_title('Query Homogeneity Quick-Check')
    ax.grid(True, linestyle='--', alpha=0.35)
    ax.legend(loc='best', fontsize=8)

    ax = axes[1]
    ax.plot(x, tr_ent, marker='o', linewidth=1.6, markersize=3.2, label='train_fusion_entropy_norm')
    ax.plot(x, va_ent, marker='o', linewidth=1.6, markersize=3.2, label='val_fusion_entropy_norm')
    ax.plot(x, tr_amx, marker='o', linewidth=1.6, markersize=3.2, linestyle='--', label='train_fusion_attn_max')
    ax.plot(x, va_amx, marker='o', linewidth=1.6, markersize=3.2, linestyle='--', label='val_fusion_attn_max')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Fusion Metric')
    ax.grid(True, linestyle='--', alpha=0.35)
    ax.legend(loc='best', fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def load_checkpoint(
    device,
    r_path,
    deviation_model,
    classifier,
    opt,
    load_opt,
    scaler
):
    """
    已实现加载下游解冻enc部分
    """
    try:
        checkpoint = torch.load(r_path, map_location=torch.device('cpu'))
        epoch = checkpoint['epoch']

        # -- lodding deviation_model
        pretrained_dict_0 = checkpoint['deviation_model']
        msg = deviation_model.load_state_dict(pretrained_dict_0)
        logger.info(f'loaded pretrained deviation_model from epoch {epoch} with msg: {msg}')

        # -- loading encoder
        pretrained_dict_1 = checkpoint['classifier']
        msg = classifier.load_state_dict(pretrained_dict_1)
        logger.info(f'loaded pretrained classifier from epoch {epoch} with msg: {msg}')

        # -- loading optimizer
        # opt.load_state_dict(checkpoint['opt'])
        # if scaler is not None:
        #     scaler.load_state_dict(checkpoint['scaler'])
        # logger.info(f'loaded optimizers from epoch {epoch}')
        if load_opt:
            opt.load_state_dict(checkpoint['opt'])
            if scaler is not None:
                scaler.load_state_dict(checkpoint['scaler'])
            logger.info(f'loaded optimizers from epoch {epoch}')
        else:
            logger.info(f'skip loading optimizer/scaler from checkpoint')


        logger.info(f'read-path: {r_path}')
        del checkpoint

    except Exception as e:
        logger.info(f'Encountered exception when loading checkpoint {e}')
        epoch = 0

    return deviation_model, classifier, opt, scaler, epoch

def load_pretrained(
    encoder,
    pretrained,
    checkpoint_key='target_encoder'
):
    logger.info(f'Loading pretrained model from {pretrained}')
    checkpoint = torch.load(pretrained, map_location='cpu')
    try:
        pretrained_dict = checkpoint[checkpoint_key]
    except Exception:
        pretrained_dict = checkpoint['encoder']

    pretrained_dict = {k.replace('module.', ''): v for k, v in pretrained_dict.items()}
    pretrained_dict = {k.replace('backbone.', ''): v for k, v in pretrained_dict.items()}
    for k, v in encoder.state_dict().items():
        if k not in pretrained_dict:
            logger.info(f'key "{k}" could not be found in loaded state dict')
        elif pretrained_dict[k].shape != v.shape:
            logger.info(f'key "{k}" is of different shape in model and loaded state dict')
            pretrained_dict[k] = v
    msg = encoder.load_state_dict(pretrained_dict, strict=False)
    print(encoder)
    logger.info(f'loaded pretrained model with msg: {msg}')
    logger.info(f'loaded pretrained encoder from epoch: {checkpoint["epoch"]}\n path: {pretrained}')
    del checkpoint
    return encoder

# def init_opt(
#     classifier,
#     iterations_per_epoch,
#     start_lr,
#     ref_lr,
#     warmup,
#     num_epochs,
#     wd=1e-6,
#     final_wd=1e-6,
#     final_lr=0.0,
#     use_bfloat16=False,
# ):
#     param_groups = [
#         {
#             'params': (p for n, p in classifier.named_parameters()
#                        if ('bias' not in n) and (len(p.shape) != 1))
#         }, {
#             'params': (p for n, p in classifier.named_parameters()
#                        if ('bias' in n) or (len(p.shape) == 1)),
#             'WD_exclude': True,
#             'weight_decay': 0
#         }
#     ]

#     logger.info('Using AdamW')
#     optimizer = torch.optim.AdamW(param_groups)
#     scheduler = WarmupCosineSchedule(
#         optimizer,
#         warmup_steps=int(warmup*iterations_per_epoch),
#         start_lr=start_lr,
#         ref_lr=ref_lr,
#         final_lr=final_lr,
#         T_max=int(num_epochs*iterations_per_epoch))
#     wd_scheduler = CosineWDSchedule(
#         optimizer,
#         ref_wd=wd,
#         final_wd=final_wd,
#         T_max=int(num_epochs*iterations_per_epoch))
#     scaler = torch.amp.GradScaler('cuda') if use_bfloat16 else None
#     return optimizer, scaler, scheduler, wd_scheduler


def init_opt(
    classifier,
    iterations_per_epoch,
    start_lr,
    ref_lr,
    warmup,
    num_epochs,
    wd=2e-2,
    final_wd=2e-2,
    final_lr=None,
    use_bfloat16=False,
    extra_modules=None,  # 新增，用于传入需训练的 encoder
    use_layerwise_lr=False,
    use_plane_path=False,
    encoder_lr=None,
    pagca_lr=None,
    classifier_lr=None,
):
    # 默认回退到统一 lr
    encoder_lr = ref_lr if encoder_lr is None else encoder_lr
    pagca_lr = ref_lr if pagca_lr is None else pagca_lr
    classifier_lr = ref_lr if classifier_lr is None else classifier_lr

    # 收集参数并按模块归类
    classifier_named = list(classifier.named_parameters())
    extra_named = []
    if extra_modules is not None:
        for name, module in extra_modules.items():
            extra_named += [(f"{name}.{n}", p) for n, p in module.named_parameters() if p.requires_grad]

    def _bucket_from_name(name):
        if name.startswith('deviation_model.'):
            return 'encoder'
        # plane-aware classifier 里的 PAGCA 模块
        if use_plane_path and (
            name.startswith('module.plane_cross_attention')
            or name.startswith('module.plane_gate')
            or name.startswith('plane_cross_attention')
            or name.startswith('plane_gate')
        ):
            return 'pagca'
        return 'classifier'

    all_named = classifier_named + extra_named
    buckets = {'encoder': [], 'pagca': [], 'classifier': []}
    for n, p in all_named:
        if not p.requires_grad:
            continue
        buckets[_bucket_from_name(n)].append((n, p))

    if not use_layerwise_lr:
        # 兼容旧行为：所有参数同 lr
        trainable_params = all_named
        param_groups = [
            {'params': [p for n, p in trainable_params if p.requires_grad and ('bias' not in n) and (len(p.shape) != 1)]},
            {'params': [p for n, p in trainable_params if p.requires_grad and (('bias' in n) or (len(p.shape) == 1))],
             'WD_exclude': True, 'weight_decay': 0}
        ]
    else:
        # 分层 lr：通过 lr_scale 与调度器主 lr(ref_lr) 解耦
        lr_map = {
            'encoder': encoder_lr,
            'pagca': pagca_lr,
            'classifier': classifier_lr,
        }
        param_groups = []
        for bucket_name, named_params in buckets.items():
            if len(named_params) == 0:
                continue
            bucket_lr = lr_map[bucket_name]
            lr_scale = bucket_lr / ref_lr if ref_lr > 0 else 1.0

            decay_params = [p for n, p in named_params if ('bias' not in n) and (len(p.shape) != 1)]
            no_decay_params = [p for n, p in named_params if (('bias' in n) or (len(p.shape) == 1))]

            if len(decay_params) > 0:
                param_groups.append({
                    'params': decay_params,
                    'lr_scale': lr_scale,
                    'group_name': f'{bucket_name}_decay',
                })
            if len(no_decay_params) > 0:
                param_groups.append({
                    'params': no_decay_params,
                    'WD_exclude': True,
                    'weight_decay': 0,
                    'lr_scale': lr_scale,
                    'group_name': f'{bucket_name}_no_decay',
                })

        logger.info(
            'Layer-wise optimizer groups: '
            + ', '.join([
                f"{g.get('group_name', 'group')}({sum(p.numel() for p in g['params'])} params, lr_scale={g.get('lr_scale', 1.0):.4f})"
                for g in param_groups
            ])
        )

    logger.info('Using AdamW')
    optimizer = torch.optim.AdamW(param_groups)
    scheduler = WarmupCosineSchedule(
        optimizer,
        warmup_steps=int(warmup*iterations_per_epoch),
        start_lr=start_lr,
        ref_lr=ref_lr,
        final_lr=final_lr,
        T_max=int(num_epochs*iterations_per_epoch))
    wd_scheduler = CosineWDSchedule(
        optimizer,
        ref_wd=wd,
        final_wd=final_wd,
        T_max=int(num_epochs*iterations_per_epoch))
    scaler = torch.amp.GradScaler('cuda') if use_bfloat16 else None
    return optimizer, scaler, scheduler, wd_scheduler


class TripleEncoderWrapper(torch.nn.Module):
    """
    三个独立 encoder（不共享权重），分别处理 axial / coronal / sagittal 三个平面。
    state_dict() 会自动包含三个子模块的权重，前缀为 encoder_a. / encoder_c. / encoder_s.

    forward(vol_a, vol_c, vol_s) -> (feat_a, feat_c, feat_s)
    单次前向调用即可得到三个平面的特征，便于 DDP 兼容。
    """
    def __init__(self, encoder_a, encoder_c, encoder_s):
        super().__init__()
        self.encoder_a = encoder_a
        self.encoder_c = encoder_c
        self.encoder_s = encoder_s

    def forward(self, vol_a, vol_c, vol_s):
        return self.encoder_a(vol_a), self.encoder_c(vol_c), self.encoder_s(vol_s)


class MultiEncoderWrapper(torch.nn.Module):
    """
    灵活封装 N 个独立 encoder（不共享权重），通过 view_name 索引调用。
    支持任意数量的平面视角，便于 DDP 兼容。

    forward(view_name, x) -> features
    """
    def __init__(self, encoder_dict):
        super().__init__()
        for name, enc in encoder_dict.items():
            setattr(self, f'enc_{name}', enc)
        self.view_names = list(encoder_dict.keys())

    def forward(self, view_name, x):
        return getattr(self, f'enc_{view_name}')(x)


class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        # logits: (B, C)
        # targets: (B,)
        ce_loss = torch.nn.functional.cross_entropy(
            logits, targets, reduction='none'
        )
        pt = torch.exp(-ce_loss)  # pt = p_y
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


