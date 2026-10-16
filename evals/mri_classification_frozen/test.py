import os
import copy
import ast
import json
import csv
import logging
import pprint
import numpy as np

import torch
import torch.multiprocessing as mp
import torch.nn.functional as F

from src.utils.distributed import init_distributed
from evals.mri_classification_frozen.utils import init_model_frozen, MaskAggregation
from src.datasets.mri_dataset_frozen import make_dataloader
from src.datasets.medmnist3d_frozen import make_medmnist3d_dataloader
from app.vjepa.transforms_mri import MakeMRITransforms, MakeMedMNIST3DTransforms
from src.models.attentive_pooler import (
    MultiHeadAttentiveClassifier,
    PlaneAwareMultiHeadClassifier,
    TokenToVolume3DClassifier,
)

from evals.mri_classification_frozen.eval import (
    run_one_epoch,
    compute_roc_auc,
    save_confusion_matrices,
)


logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)
pp = pprint.PrettyPrinter(indent=4)


def _safe_load_state_dict(model, state_dict, model_name):
    """Load state dict for both DDP-saved and plain model checkpoints."""
    model_keys = list(model.state_dict().keys())
    model_has_module = any(k.startswith('module.') for k in model_keys)
    ckpt_has_module = any(k.startswith('module.') for k in state_dict.keys())

    aligned_state_dict = state_dict
    if ckpt_has_module and not model_has_module:
        aligned_state_dict = {k.replace('module.', '', 1): v for k, v in state_dict.items()}
        logger.info(f'{model_name}: stripped module. prefix before strict load')
    elif (not ckpt_has_module) and model_has_module:
        aligned_state_dict = {f'module.{k}': v for k, v in state_dict.items()}
        logger.info(f'{model_name}: added module. prefix before strict load')

    try:
        msg = model.load_state_dict(aligned_state_dict, strict=True)
        logger.info(f'Loaded {model_name} with strict=True: {msg}')
        return
    except RuntimeError as e:
        logger.warning(f'{model_name} strict load failed after prefix alignment: {e}')
        msg = model.load_state_dict(aligned_state_dict, strict=False)
        logger.info(f'Loaded {model_name} with strict=False: {msg}')


def _build_ckpt_path(args_eval, dataset_npz, pretrain_folder, tag, eval_tag, args_test):
    ckpt_path = args_test.get('checkpoint_path', None)
    if ckpt_path is not None:
        return ckpt_path

    ckpt_name = args_test.get('checkpoint_name', f'{tag}-best-auc.pth.tar')
    if dataset_npz is not None:
        dataset_name = os.path.splitext(os.path.basename(dataset_npz))[0]
        folder = os.path.join(pretrain_folder, dataset_name)
    else:
        folder = pretrain_folder

    if eval_tag is not None:
        folder = os.path.join(folder, eval_tag)

    return os.path.join(folder, ckpt_name)


def main(args_eval, resume_preempt=False):
    args_pretrain = args_eval.get('pretrain')
    args_data = args_eval.get('data')
    args_opt = args_eval.get('optimization')
    args_test = args_eval.get('test', {})

    # -- PRETRAIN
    model_name = args_pretrain.get('model_name', None)
    patch_size = args_pretrain.get('patch_size', None)
    pretrain_folder = args_pretrain.get('folder', None)
    ckp_fname = args_pretrain.get('checkpoint', None)
    load_pretrained_ckpt = args_pretrain.get('load_pretrained', True)
    tag = args_pretrain.get('write_tag', None)
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

    # -- DATA
    dataset_type = args_data.get('dataset_type', 'mridataset')
    dataset_path = args_data.get('dataset_path')
    label_path = args_data.get('label_path')
    dataset_npz = args_data.get('dataset_npz', None)
    split_test = args_data.get('split_test', 'test')
    defect = args_data.get('defect', None)
    defect_classes = args_data.get('defect_classes', None)
    mask_size = args_data.get('mask_size', '(5, 80, 80)')

    # -- OPT
    batch_size = args_opt.get('batch_size', 32)
    use_bfloat16 = args_opt.get('use_bfloat16', True)
    num_workers = args_opt.get('num_workers', 8)
    persistent_workers = args_opt.get('persistent_workers', True)
    prefetch_factor = args_opt.get('prefetch_factor', 4)
    pin_memory = args_opt.get('pin_memory', True)
    scl_weight = args_opt.get('scl_weight', 0.0)
    scl_temperature = args_opt.get('scl_temperature', 0.03)
    use_balanced_softmax = args_opt.get('use_balanced_softmax', False)
    balanced_softmax_momentum = args_opt.get('balanced_softmax_momentum', 0.95)
    balanced_softmax_eps = args_opt.get('balanced_softmax_eps', 1e-6)
    use_cosface = args_opt.get('use_cosface', False)
    cosface_s = args_opt.get('cosface_s', 30.0)
    cosface_m = args_opt.get('cosface_m', 0.35)

    use_plane_path = args_opt.get('use_plane_path', False)
    use_plane_cross_attention = args_opt.get('use_plane_cross_attention', True)
    plane_cross_num_heads = args_opt.get('plane_cross_num_heads', 8)
    plane_cross_dropout = args_opt.get('plane_cross_dropout', 0.0)
    plane_gate_pool_num_queries = args_opt.get('plane_gate_pool_num_queries', 1)
    plane_gate_pool_num_heads = args_opt.get('plane_gate_pool_num_heads', 8)
    plane_gate_pool_depth = args_opt.get('plane_gate_pool_depth', 2)
    plane_gate_pool_mlp_ratio = args_opt.get('plane_gate_pool_mlp_ratio', 4.0)
    plane_gate_pool_dropout = args_opt.get('plane_gate_pool_dropout', 0.1)

    use_token_3d_cnn_head = args_opt.get('use_token_3d_cnn_head', False)
    token_3d_use_coords = args_opt.get('token_3d_use_coords', True)
    token_3d_stem_channels = args_opt.get('token_3d_stem_channels', 128)
    token_3d_stage_channels = args_opt.get('token_3d_stage_channels', [128, 256, 384])
    token_3d_stage_depths = args_opt.get('token_3d_stage_depths', [1, 1, 1])
    token_3d_stage_strides = args_opt.get('token_3d_stage_strides', [1, 2, 2])
    token_3d_dropout = args_opt.get('token_3d_dropout', 0.1)
    token_3d_grid_size = args_opt.get('token_3d_grid_size', None)

    classifier_pool_mode = str(args_opt.get('classifier_pool_mode', 'attentive')).lower()
    if classifier_pool_mode not in ('attentive', 'gap', 'head_cls', 'swin_default'):
        raise ValueError("optimization.classifier_pool_mode 必须为 'attentive' | 'gap' | 'head_cls' | 'swin_default'")

    attn_num_queries = args_opt.get('attn_num_queries', 1)
    attn_num_heads = args_opt.get('attn_num_heads', 8)
    attn_mlp_ratio = args_opt.get('attn_mlp_ratio', 4.0)
    attn_depth = args_opt.get('attn_depth', 2)

    head_cls_num_heads = args_opt.get('head_cls_num_heads', 8)
    head_cls_mlp_ratio = args_opt.get('head_cls_mlp_ratio', 4.0)
    head_cls_depth = args_opt.get('head_cls_depth', 4)

    # -- TEST
    eval_tag = args_eval.get('tag', None)
    test_batch_size = args_test.get('batch_size', batch_size)
    save_confusion = args_test.get('save_confusion', True)

    torch.backends.cudnn.benchmark = True

    try:
        mp.set_start_method('spawn')
    except Exception:
        pass

    if not torch.cuda.is_available():
        device = torch.device('cpu')
        logger.warning('No GPU detected, testing on CPU.')
    else:
        device = torch.device('cuda:0')
        torch.cuda.set_device(device)
        logger.info(f"Using GPU: {torch.cuda.get_device_name(device)}")

    world_size, rank = init_distributed()
    logger.info(f'Initialized (rank/world-size) {rank}/{world_size}')

    if use_plane_path and use_token_3d_cnn_head:
        raise ValueError('use_plane_path 与 use_token_3d_cnn_head 不能同时开启，请二选一')
    if use_swin_backbone and unfreeze == -1:
        raise ValueError('Swin backbone does not support mask aggregation mode (unfreeze = -1)')

    ckpt_path = _build_ckpt_path(args_eval, dataset_npz, pretrain_folder, tag, eval_tag, args_test)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f'Checkpoint not found: {ckpt_path}')
    logger.info(f'Test checkpoint: {ckpt_path}')

    # -- frozen pretrained model
    encoder, predictor = init_model_frozen(
        device,
        pretrained=pretrained_path,
        load_pretrained_ckpt=load_pretrained_ckpt,
        model_name=model_name,
        patch_size=patch_size,
        crop_size=args_pretrain.get('crop_size', 224),
        num_slices=args_pretrain.get('num_slices', 20),
        in_chans=args_pretrain.get('in_chans', 3),
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
    encoder_embed_dim = encoder.embed_dim

    if unfreeze == -1:
        deviation_model = MaskAggregation(
            device,
            encoder,
            predictor,
            target_encoder,
            mask_size=ast.literal_eval(mask_size),
        ).to(device)
    elif unfreeze == 0:
        deviation_model = target_encoder.to(device)
        deviation_model.eval()
    elif unfreeze > 0:
        for p in target_encoder.parameters():
            p.requires_grad = False
        if unfreeze_all:
            for p in target_encoder.parameters():
                p.requires_grad = True
        else:
            for blk in target_encoder.blocks[-unfreeze:]:
                for p in blk.parameters():
                    p.requires_grad = True
            for p in target_encoder.norm.parameters():
                p.requires_grad = True
        deviation_model = target_encoder.to(device)
    else:
        raise ValueError('unfreeze must be -1, 0, positive integer, or one of: all/full/full_encoder/all_encoder')

    if use_plane_path:
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
            embed_dim=encoder_embed_dim,
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

    # -- load ckpt (downstream classifier + deviation model)
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    if 'deviation_model' not in checkpoint or 'classifier' not in checkpoint:
        raise KeyError('Checkpoint must contain keys: deviation_model and classifier')

    _safe_load_state_dict(deviation_model, checkpoint['deviation_model'], 'deviation_model')
    _safe_load_state_dict(classifier, checkpoint['classifier'], 'classifier')

    deviation_model.eval()
    classifier.eval()

    # -- dataloader
    if dataset_type.lower() == 'medmnist3d':
        test_loader = make_medmnist3d_dataloader(
            npz_path=dataset_npz,
            split=split_test,
            transform=MakeMedMNIST3DTransforms,
            batch_size=test_batch_size,
            num_workers=num_workers,
            training=False,
            persistent_workers=persistent_workers,
            prefetch_factor=max(2, prefetch_factor // 2),
            pin_memory=pin_memory,
            drop_last=False,
        )
    else:
        test_loader = make_dataloader(
            training=False,
            dataset_path=dataset_path,
            label_path=label_path,
            transform=MakeMRITransforms,
            batch_size=test_batch_size,
            defect=defect,
            num_workers=num_workers,
        )

    class_prior_state = [
        torch.full((n_cls,), 1.0 / n_cls, device=device)
        for n_cls in defect_classes
    ]

    metrics = run_one_epoch(
        device=device,
        training=False,
        deviation_model=deviation_model,
        classifier=classifier,
        scaler=None,
        optimizer=None,
        scheduler=None,
        wd_scheduler=None,
        data_loader=test_loader,
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
        update_priors=False,
        scl_weight=scl_weight,
        scl_temperature=scl_temperature,
        unfreeze=unfreeze,
        use_plane_path=use_plane_path,
        use_token_3d_cnn_head=use_token_3d_cnn_head,
        collect_confusion=save_confusion,
    )

    (
        test_f1_scores,
        test_precisions,
        test_recalls,
        test_f1_per_class,
        test_prec_per_class,
        test_rec_per_class,
        test_avg_loss,
        test_avg_scl_loss,
        test_top1_scores,
        test_top2_scores,
        test_plane_weights,
        test_query_metrics,
        test_confusion,
    ) = metrics

    roc_aucs = compute_roc_auc(
        device=device,
        deviation_model=deviation_model,
        classifier=classifier,
        data_loader=test_loader,
        defect=defect,
        defect_classes=defect_classes,
        use_bfloat16=use_bfloat16,
        unfreeze=unfreeze,
        use_plane_path=use_plane_path,
        use_token_3d_cnn_head=use_token_3d_cnn_head,
    )

    if rank == 0:
        logger.info(f'Test avg loss: {test_avg_loss:.6f}, test avg scl loss: {test_avg_scl_loss:.6f}')
        for i in range(len(defect)):
            logger.info(
                f'Task {i + 1} | top1={test_top1_scores[i]:.3f}% top2={test_top2_scores[i]:.3f}% '
                f'P={test_precisions[i]:.4f} R={test_recalls[i]:.4f} F1={test_f1_scores[i]:.4f} '
                f'AUC={roc_aucs[i]:.4f}'
            )

        if use_plane_path:
            logger.info(
                f'Test plane weights avg: w_a={test_plane_weights[0]:.4f}, '
                f'w_c={test_plane_weights[1]:.4f}, w_s={test_plane_weights[2]:.4f}'
            )

        out_dir = args_test.get('confusion_dir', None)
        if out_dir is None:
            if dataset_npz is not None:
                dataset_name = os.path.splitext(os.path.basename(dataset_npz))[0]
                base_dir = os.path.join(pretrain_folder, dataset_name)
            else:
                base_dir = pretrain_folder
            if eval_tag is not None:
                base_dir = os.path.join(base_dir, eval_tag)
            out_dir = os.path.join(base_dir, 'test')
        os.makedirs(out_dir, exist_ok=True)

        metrics_payload = {
            'checkpoint_path': ckpt_path,
            'split_test': split_test,
            'test_avg_loss': float(test_avg_loss),
            'test_avg_scl_loss': float(test_avg_scl_loss),
            'test_top1_scores': [float(x) for x in test_top1_scores],
            'test_top2_scores': [float(x) for x in test_top2_scores],
            'test_precisions': [float(x) for x in test_precisions],
            'test_recalls': [float(x) for x in test_recalls],
            'test_f1_scores': [float(x) for x in test_f1_scores],
            'roc_aucs': [float(x) if x is not None else None for x in roc_aucs],
            'test_plane_weights': [float(x) for x in test_plane_weights],
            'test_prec_per_class': [[float(v) for v in arr] for arr in test_prec_per_class],
            'test_rec_per_class': [[float(v) for v in arr] for arr in test_rec_per_class],
            'test_f1_per_class': [[float(v) for v in arr] for arr in test_f1_per_class],
            # 'test_query_metrics': [float(x) for x in test_query_metrics],
        }

        metrics_json_path = os.path.join(out_dir, 'test_metrics.json')
        with open(metrics_json_path, 'w', encoding='utf-8') as f:
            json.dump(metrics_payload, f, ensure_ascii=False, indent=2)

        logger.info(f'Saved test metrics json to: {metrics_json_path}')

        if len(defect) > 1:
            metrics_csv_path = os.path.join(out_dir, 'test_metrics_per_task.csv')
            with open(metrics_csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        'task',
                        'top1',
                        'top2',
                        'precision',
                        'recall',
                        'f1',
                        'roc_auc',
                    ],
                )
                writer.writeheader()
                for i in range(len(defect)):
                    writer.writerow({
                        'task': i + 1,
                        'top1': float(test_top1_scores[i]),
                        'top2': float(test_top2_scores[i]),
                        'precision': float(test_precisions[i]),
                        'recall': float(test_recalls[i]),
                        'f1': float(test_f1_scores[i]),
                        'roc_auc': float(roc_aucs[i]) if roc_aucs[i] is not None else '',
                    })
            logger.info(f'Saved test metrics csv to: {metrics_csv_path}')

        if save_confusion and test_confusion is not None:
            test_labels, test_preds = test_confusion
            save_confusion_matrices(
                test_labels,
                test_preds,
                defect_classes,
                out_dir,
                prefix='test',
            )
            logger.info(f'Saved test confusion matrices to: {out_dir}')
