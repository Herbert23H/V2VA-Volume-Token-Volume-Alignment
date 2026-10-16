import ast
import copy
import json
import logging
import os
import pprint
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from app.vjepa.transforms_mri import MakeMRITransforms, MakeMedMNIST3DTransforms
from evals.mri_classification_frozen.utils import init_model_frozen, MaskAggregation
from src.datasets.mri_dataset_frozen import make_dataloader
from src.datasets.medmnist3d_frozen import make_medmnist3d_dataloader
from src.models.attentive_pooler import TokenToVolume3DClassifier


logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)
pp = pprint.PrettyPrinter(indent=2)


def _safe_load_state_dict(model, state_dict, model_name):
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


def _build_ckpt_path(args_eval, dataset_npz, pretrain_folder, tag, eval_tag, args_draw):
    ckpt_path = args_draw.get('checkpoint_path', None)
    if ckpt_path is not None:
        return ckpt_path

    ckpt_name = args_draw.get('checkpoint_name', f'{tag}-best.pth.tar')
    if dataset_npz is not None:
        dataset_name = os.path.splitext(os.path.basename(dataset_npz))[0]
        folder = os.path.join(pretrain_folder, dataset_name)
    else:
        folder = pretrain_folder

    if eval_tag is not None:
        folder = os.path.join(folder, eval_tag)

    model_tag = args_draw.get('model_tag', None)
    if model_tag is not None:
        folder = os.path.join(folder, model_tag)

    return os.path.join(folder, ckpt_name)


def _to_numpy01(x):
    x = x.astype(np.float32)
    mn = float(x.min())
    mx = float(x.max())
    if mx - mn < 1e-8:
        return np.zeros_like(x, dtype=np.float32)
    return (x - mn) / (mx - mn)


def _pca_rgb_volume(feat_4d: torch.Tensor) -> np.ndarray:
    """
    feat_4d: [C, D, H, W]
    return : [D, H, W, 3] in [0,1]
    """
    if feat_4d.dim() != 4:
        raise ValueError(f'_pca_rgb_volume expects [C,D,H,W], got {tuple(feat_4d.shape)}')

    c, d, h, w = feat_4d.shape
    x = feat_4d.reshape(c, -1).transpose(0, 1).float()  # [V, C]
    x = x - x.mean(dim=0, keepdim=True)

    q = min(3, x.shape[1], x.shape[0])
    if q < 3:
        pad = torch.zeros(x.shape[0], 3 - q, device=x.device, dtype=x.dtype)
        if q > 0:
            u, s, v = torch.pca_lowrank(x, q=q)
            z = x @ v[:, :q]
            z = torch.cat([z, pad], dim=1)
        else:
            z = pad
    else:
        u, s, v = torch.pca_lowrank(x, q=3)
        z = x @ v[:, :3]

    z = z.detach().cpu().numpy()
    for i in range(3):
        z[:, i] = _to_numpy01(z[:, i])

    rgb = z.reshape(d, h, w, 3)
    return rgb


def _plot_pca_before_after(before_feat: torch.Tensor, after_feat: torch.Tensor, save_path: str, title: str):
    """
    before_feat: [C0, D, H, W]
    after_feat : [C1, D, H, W]
    """
    rgb_b = _pca_rgb_volume(before_feat)
    rgb_a = _pca_rgb_volume(after_feat)

    d, h, w, _ = rgb_b.shape
    cd, ch, cw = d // 2, h // 2, w // 2

    fig, axes = plt.subplots(2, 3, figsize=(12, 7))

    axes[0, 0].imshow(rgb_b[cd])
    axes[0, 0].set_title('Before Stem | Axial')
    axes[0, 1].imshow(rgb_b[:, ch, :, :])
    axes[0, 1].set_title('Before Stem | Coronal')
    axes[0, 2].imshow(rgb_b[:, :, cw, :])
    axes[0, 2].set_title('Before Stem | Sagittal')

    axes[1, 0].imshow(rgb_a[cd])
    axes[1, 0].set_title('After Stem | Axial')
    axes[1, 1].imshow(rgb_a[:, ch, :, :])
    axes[1, 1].set_title('After Stem | Coronal')
    axes[1, 2].imshow(rgb_a[:, :, cw, :])
    axes[1, 2].set_title('After Stem | Sagittal')

    for ax in axes.flatten():
        ax.axis('off')

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=180)
    plt.close(fig)


def _norm_cam(cam: torch.Tensor) -> torch.Tensor:
    cam = cam - cam.min()
    den = cam.max().clamp_min(1e-6)
    cam = cam / den
    return cam


def _as_gray_volume(input_vol: torch.Tensor) -> np.ndarray:
    """
    input_vol: [C, D, H, W] 或 [D, H, W]
    return   : [D, H, W] in [0,1]
    """
    if input_vol.dim() == 4:
        gray = input_vol.float().mean(dim=0)
    elif input_vol.dim() == 3:
        gray = input_vol.float()
    else:
        raise ValueError(f'_as_gray_volume expects [C,D,H,W] or [D,H,W], got {tuple(input_vol.shape)}')
    gray_np = gray.detach().cpu().numpy().astype(np.float32)
    return _to_numpy01(gray_np)


def _overlay_rgb_on_gray(gray_2d: np.ndarray, rgb_2d: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    gray_2d = np.clip(gray_2d.astype(np.float32), 0.0, 1.0)
    rgb_2d = np.clip(rgb_2d.astype(np.float32), 0.0, 1.0)
    gray3 = np.repeat(gray_2d[..., None], 3, axis=-1)
    return (1.0 - alpha) * gray3 + alpha * rgb_2d


def _plot_pca_before_after_overlay(
    input_vol: torch.Tensor,
    patch_embed_feat: Optional[torch.Tensor],
    before_feat: torch.Tensor,
    after_feat: torch.Tensor,
    save_path: str,
    title: str,
    alpha: float = 0.7,
):
    """
    input_vol  : [C, D, H, W] (原图体数据)
    patch_embed_feat: [C2, D, H, W]，patch embed 后、进入 encoder 前
    before_feat: [C0, D, H, W]
    after_feat : [C1, D, H, W]
    输出：原图灰度底图 + PCA 半透明叠加（3正交切面，patch_embed/before/after）
    """
    gray = _as_gray_volume(input_vol)
    rgb_p = _pca_rgb_volume(patch_embed_feat) if patch_embed_feat is not None else None
    rgb_b = _pca_rgb_volume(before_feat)
    rgb_a = _pca_rgb_volume(after_feat)

    def _resize_rgb(rgb_vol: np.ndarray, target_shape: Tuple[int, int, int]) -> np.ndarray:
        td, th, tw = target_shape
        if rgb_vol.shape[:3] == (td, th, tw):
            return rgb_vol
        vol_t = torch.from_numpy(rgb_vol).permute(3, 0, 1, 2).unsqueeze(0).float()
        vol_t = F.interpolate(
            vol_t,
            size=(td, th, tw),
            mode='trilinear',
            align_corners=False,
        )
        vol_np = vol_t.squeeze(0).permute(1, 2, 3, 0).clamp(0, 1).cpu().numpy()
        return vol_np

    d, h, w = gray.shape
    if rgb_p is not None:
        rgb_p = _resize_rgb(rgb_p, (d, h, w))
    rgb_b = _resize_rgb(rgb_b, (d, h, w))
    rgb_a = _resize_rgb(rgb_a, (d, h, w))
    cd, ch, cw = d // 2, h // 2, w // 2

    # axial/coronal/sagittal 切面
    g_ax, g_co, g_sa = gray[cd], gray[:, ch, :], gray[:, :, cw]
    if rgb_p is not None:
        p_ax, p_co, p_sa = rgb_p[cd], rgb_p[:, ch, :, :], rgb_p[:, :, cw, :]
    b_ax, b_co, b_sa = rgb_b[cd], rgb_b[:, ch, :, :], rgb_b[:, :, cw, :]
    a_ax, a_co, a_sa = rgb_a[cd], rgb_a[:, ch, :, :], rgb_a[:, :, cw, :]

    if rgb_p is not None:
        ov_p = [
            _overlay_rgb_on_gray(g_ax, p_ax, alpha=alpha),
            _overlay_rgb_on_gray(g_co, p_co, alpha=alpha),
            _overlay_rgb_on_gray(g_sa, p_sa, alpha=alpha),
        ]

    ov_b = [
        _overlay_rgb_on_gray(g_ax, b_ax, alpha=alpha),
        _overlay_rgb_on_gray(g_co, b_co, alpha=alpha),
        _overlay_rgb_on_gray(g_sa, b_sa, alpha=alpha),
    ]
    ov_a = [
        _overlay_rgb_on_gray(g_ax, a_ax, alpha=alpha),
        _overlay_rgb_on_gray(g_co, a_co, alpha=alpha),
        _overlay_rgb_on_gray(g_sa, a_sa, alpha=alpha),
    ]

    n_rows = 3 if rgb_p is not None else 2
    # fig, axes = plt.subplots(n_rows, 3, figsize=(12, 3.6 * n_rows))
    fig, axes = plt.subplots(n_rows, 3, figsize=(12, 7))    # 不画pe
    names = ['Axial', 'Coronal', 'Sagittal']
    row = 0
    if rgb_p is not None:
        for i in range(3):
            axes[row, i].imshow(ov_p[i])
            axes[row, i].set_title(f'Patch Embed | {names[i]}')
        row += 1
    for i in range(3):
        axes[row, i].imshow(ov_b[i])
        axes[row, i].set_title(f'Before Stem | {names[i]}')
        axes[row + 1, i].imshow(ov_a[i])
        axes[row + 1, i].set_title(f'After Stem | {names[i]}')

    for ax in axes.flatten():
        ax.axis('off')

    # fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=180)
    plt.close(fig)


def _extract_patch_embed_volume(encoder: torch.nn.Module, input_vol: torch.Tensor):
    """
    提取 patch embed 后、进入 encoder blocks 之前的 3D 特征。
    returns:
      vol_patch_embed: [C, D, H, W]
    """
    if input_vol.dim() != 4:
        raise ValueError(f'_extract_patch_embed_volume expects [C,D,H,W], got {tuple(input_vol.shape)}')

    ref_param = next(encoder.parameters())
    x = input_vol.unsqueeze(0).to(device=ref_param.device, dtype=ref_param.dtype)
    with torch.no_grad():
        tokens = encoder.patch_embed(x)

    bsz, n, e = tokens.shape
    d, h, w = input_vol.shape[1:]
    tubelet_size = int(getattr(encoder, 'tubelet_size', 1))
    patch_size = int(getattr(encoder, 'patch_size', 1))
    grid_d = max(1, d // tubelet_size)
    grid_h = max(1, h // patch_size)
    grid_w = max(1, w // patch_size)

    if grid_d * grid_h * grid_w != n:
        raise ValueError(
            f'patch_embed token count mismatch: N={n}, expected grid=({grid_d},{grid_h},{grid_w})'
        )

    vol = tokens.transpose(1, 2).contiguous().view(bsz, e, grid_d, grid_h, grid_w)
    return vol[0].detach().cpu()


def _compute_stage_cams(
    classifier: TokenToVolume3DClassifier,
    tokens: torch.Tensor,
    volume_shape: Tuple[int, int, int],
    task_idx: int,
    target_class: Optional[int] = None,
) -> List[torch.Tensor]:
    """
    返回 list[cam_stage_i], 每个 cam shape [D, H, W], 且已上采样到同一分辨率。
    """
    num_stages = len(classifier.stages)
    acts: List[Optional[torch.Tensor]] = [None] * num_stages
    grads: List[Optional[torch.Tensor]] = [None] * num_stages
    hooks = []

    def _make_fwd_hook(idx: int):
        def _fwd_hook(_m, _inp, out):
            acts[idx] = out
        return _fwd_hook

    def _make_bwd_hook(idx: int):
        def _bwd_hook(_m, _gin, gout):
            grads[idx] = gout[0]
        return _bwd_hook

    for idx, st in enumerate(classifier.stages):
        hooks.append(st.register_forward_hook(_make_fwd_hook(idx)))
        hooks.append(st.register_full_backward_hook(_make_bwd_hook(idx)))

    classifier.zero_grad(set_to_none=True)
    outputs = classifier(tokens, volume_shape=volume_shape)
    logits = outputs[task_idx]

    if target_class is None:
        target_class = int(torch.argmax(logits[0]).item())

    score = logits[0, target_class]
    score.backward()

    for h in hooks:
        h.remove()

    if any(a is None for a in acts) or any(g is None for g in grads):
        raise RuntimeError('Grad-CAM hook capture failed for one or more stages.')

    stage_cams = []
    for a, g in zip(acts, grads):
        a0 = a[0]  # [C, D, H, W]
        g0 = g[0]  # [C, D, H, W]
        alpha = g0.mean(dim=(1, 2, 3), keepdim=True)
        cam = torch.relu((alpha * a0).sum(dim=0))
        cam = _norm_cam(cam)
        stage_cams.append(cam)

    base_dhw = stage_cams[0].shape
    upsampled = []
    for cam in stage_cams:
        if tuple(cam.shape) != tuple(base_dhw):
            cam_u = F.interpolate(
                cam.unsqueeze(0).unsqueeze(0),
                size=base_dhw,
                mode='trilinear',
                align_corners=False,
            ).squeeze(0).squeeze(0)
            cam = _norm_cam(cam_u)
        upsampled.append(cam.detach().cpu())

    return upsampled


def _plot_gradcam_isosurfaces(
    stage_cams: List[torch.Tensor],
    save_path: str,
    title: str,
    input_vol: Optional[torch.Tensor] = None,
):
    """
    将 Grad-CAM 改为：原图灰度底图 + CAM 半透明叠加（每个 stage 一行，三正交切面）。
    stage_cams: list of tensors on CPU, each [D,H,W], values 0..1
    如果提供了 input_vol，则使用其灰度体作为背景并与 CAM 对齐/重采样。
    三方向切片索引由各方向在所有 stage 上的平均激活共同决定，
    以保证不同 stage 展示同一切片的变化过程。
    """
    colors = ['#1f77b4', '#ff7f0e', '#d62728']
    alphas = [0.5, 0.55, 0.6]

    def _resize_cam(cam_np: np.ndarray, target_shape: Tuple[int, int, int]) -> np.ndarray:
        td, th, tw = target_shape
        if cam_np.shape == (td, th, tw):
            return cam_np
        vol_t = torch.from_numpy(cam_np).unsqueeze(0).unsqueeze(0).float()
        vol_t = F.interpolate(vol_t, size=(td, th, tw), mode='trilinear', align_corners=False)
        vol_np = vol_t.squeeze(0).squeeze(0).clamp(0, 1).cpu().numpy()
        return vol_np

    # 准备灰度体
    if input_vol is not None:
        gray_vol = _as_gray_volume(input_vol)
    else:
        # 若无原图，使用第一个 cam 的尺寸作为背景
        d0, h0, w0 = stage_cams[0].shape
        gray_vol = np.zeros((d0, h0, w0), dtype=np.float32)

    d, h, w = gray_vol.shape

    n_stage = min(3, len(stage_cams))
    fig_h = 3.6 * n_stage
    fig, axes = plt.subplots(n_stage, 3, figsize=(12, fig_h))
    if n_stage == 1:
        axes = axes.reshape(1, 3)

    names = ['Axial', 'Coronal', 'Sagittal']

    # 可选的 colormap 列表，会在子图间循环使用以区分每张子图的配色
    cmaps = ['magma']    #, 'viridis', 'plasma', 'inferno', 'cividis', 'coolwarm'
 
    # 先把每个 stage 的 CAM 对齐并归一化，后续统一选切片索引
    cam_norms: List[np.ndarray] = []
    for si in range(n_stage):
        cam_np = stage_cams[si].numpy()
        if cam_np.shape != gray_vol.shape:
            cam_np = _resize_cam(cam_np, gray_vol.shape)

        cam_np = cam_np.astype(np.float32)
        mn = float(cam_np.min())
        mx = float(cam_np.max())
        if mx - mn < 1e-8:
            cam_norm = np.zeros_like(cam_np, dtype=np.float32)
        else:
            cam_norm = (cam_np - mn) / (mx - mn)
        cam_norms.append(cam_norm)

    # 三方向分别在所有 stage 上求平均激活，选出共同最优切片索引
    # axial (Z / depth): 沿 (H,W) 求均值，再对 stage 求均值
    axial_scores = np.mean([cn.mean(axis=(1, 2)) for cn in cam_norms], axis=0)
    axial_start = min(16, max(0, axial_scores.shape[0] - 1))
    axial_end = min(48, axial_scores.shape[0] - 1)
    if axial_end < axial_start:
        axial_start, axial_end = 0, axial_scores.shape[0] - 1
    best_axial = int(axial_start + np.argmax(axial_scores[axial_start:axial_end + 1]))
    # coronal (Y / height): 沿 (D,W)
    coronal_scores = np.mean([cn.mean(axis=(0, 2)) for cn in cam_norms], axis=0)
    coronal_start = min(16, max(0, coronal_scores.shape[0] - 1))
    coronal_end = min(48, coronal_scores.shape[0] - 1)
    if coronal_end < coronal_start:
        coronal_start, coronal_end = 0, coronal_scores.shape[0] - 1
    best_coronal = int(coronal_start + np.argmax(coronal_scores[coronal_start:coronal_end + 1]))
    # sagittal (X / width): 沿 (D,H)
    sagittal_scores = np.mean([cn.mean(axis=(0, 1)) for cn in cam_norms], axis=0)
    sagittal_start = min(16, max(0, sagittal_scores.shape[0] - 1))
    sagittal_end = min(48, sagittal_scores.shape[0] - 1)
    if sagittal_end < sagittal_start:
        sagittal_start, sagittal_end = 0, sagittal_scores.shape[0] - 1
    best_sagittal = int(sagittal_start + np.argmax(sagittal_scores[sagittal_start:sagittal_end + 1]))

    shared_slice_idxs = [best_axial, best_coronal, best_sagittal]
    shared_plane_grays = [
        gray_vol[shared_slice_idxs[0]],
        gray_vol[:, shared_slice_idxs[1], :],
        gray_vol[:, :, shared_slice_idxs[2]],
    ]

    for si in range(n_stage):
        cam_norm = cam_norms[si]

        for pi in range(3):
            if pi == 0:
                cam_slice = cam_norm[shared_slice_idxs[0]]
                gray_slice = shared_plane_grays[0]
            elif pi == 1:
                cam_slice = cam_norm[:, shared_slice_idxs[1], :]
                gray_slice = shared_plane_grays[1]
            else:
                cam_slice = cam_norm[:, :, shared_slice_idxs[2]]
                gray_slice = shared_plane_grays[2]

            # 使用 colormap 将单通道 cam 映射为 RGB（颜色随激活强度变化）
            cmap = matplotlib.cm.get_cmap(cmaps[(si * 3 + pi) % len(cmaps)])
            rgba = cmap(cam_slice)  # [..., 4]
            rgb = rgba[..., :3].astype(np.float32)

            # 叠加系数：以 stage 为主的全局 alpha
            alpha = alphas[si % len(alphas)]
            ov = _overlay_rgb_on_gray(gray_slice, rgb, alpha=alpha)

            ax = axes[si, pi]
            ax.imshow(ov)
            ax.set_title(f'Stage {si+1} | {names[pi]} | idx={shared_slice_idxs[pi]}')
            ax.axis('off')

    # fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=180)
    plt.close(fig)


def _select_one_sample_per_label_level(data_loader, defect_classes: List[int]):
    selected: Dict[Tuple[int, int], Tuple[torch.Tensor, torch.Tensor]] = {}
    required = {(t, c) for t, n in enumerate(defect_classes) for c in range(int(n))}

    with torch.no_grad():
        for data in data_loader:
            img, labels = data
            bsz = img.shape[0]
            for b in range(bsz):
                lb = labels[b]
                if lb.dim() == 0:
                    lb = lb.unsqueeze(0)
                if lb.dim() > 1:
                    lb = lb.view(-1)
                for t in range(min(len(defect_classes), lb.numel())):
                    cls = int(lb[t].item())
                    key = (t, cls)
                    if key in required and key not in selected:
                        selected[key] = (img[b].clone(), lb.clone())
                if len(selected) == len(required):
                    return selected
    return selected


def _build_deviation_model(device, args_pretrain, pretrained_path, load_pretrained_ckpt, mask_size):
    model_name = args_pretrain.get('model_name', None)
    patch_size = args_pretrain.get('patch_size', None)
    depth_patch_size = args_pretrain.get('depth_patch_size', 1)
    unfreeze_raw = args_pretrain.get('unfreeze', 0)

    if isinstance(unfreeze_raw, str):
        if unfreeze_raw.strip().lower() in ('all', 'full', 'full_encoder', 'all_encoder'):
            unfreeze_all = True
            unfreeze = 9999
        else:
            unfreeze_all = False
            unfreeze = int(unfreeze_raw)
    else:
        unfreeze_all = False
        unfreeze = int(unfreeze_raw)

    encoder, predictor = init_model_frozen(
        device,
        pretrained=pretrained_path,
        load_pretrained_ckpt=load_pretrained_ckpt,
        model_name=model_name,
        patch_size=patch_size,
        crop_size=args_pretrain.get('crop_size', 64),
        num_slices=args_pretrain.get('num_slices', 64),
        in_chans=args_pretrain.get('in_chans', 1),
        depth_patch_size=depth_patch_size,
        use_sdpa=args_pretrain.get('use_sdpa', True),
        use_SiLU=args_pretrain.get('use_silu', False),
        tight_SiLU=args_pretrain.get('tight_silu', True),
        uniform_power=args_pretrain.get('uniform_power', False),
    )
    target_encoder = copy.deepcopy(encoder)

    if unfreeze == -1:
        if isinstance(mask_size, str):
            mask_size = ast.literal_eval(mask_size)
        deviation_model = MaskAggregation(
            device,
            encoder,
            predictor,
            target_encoder,
            mask_size=mask_size,
            num_slices=args_pretrain.get('num_slices', 64),
            crop_size=args_pretrain.get('crop_size', 64),
            depth_patch_size=depth_patch_size,
            spatial_patch_size=patch_size,
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
        raise ValueError('unfreeze must be -1, 0, positive integer, or all/full/full_encoder/all_encoder')

    return deviation_model, int(unfreeze), int(depth_patch_size), int(patch_size)


def _extract_token_volume(classifier: TokenToVolume3DClassifier, tokens: torch.Tensor, volume_shape: Tuple[int, int, int]):
    """
    复用 TokenToVolume3DClassifier 的逻辑，拿到 stem 前后的特征（用于 PCA）。
    returns:
      vol_before_stem: [C0, D, H, W]
      vol_after_stem : [C1, D, H, W]
    """
    bsz, n, e = tokens.shape
    d, h, w = classifier._resolve_grid(num_tokens=n, volume_shape=volume_shape)

    vol = tokens.transpose(1, 2).contiguous().view(bsz, e, d, h, w)
    if classifier.use_coord_channels:
        coords = classifier._build_coords(bsz, d, h, w, tokens.device, tokens.dtype)
        vol = torch.cat([vol, coords], dim=1)

    stem_out = classifier.stem(vol)
    return vol[0].detach().cpu(), stem_out[0].detach().cpu()


def _build_data_loader(args_data, args_opt, split):
    dataset_type = str(args_data.get('dataset_type', 'medmnist3d')).lower()

    num_workers = args_opt.get('num_workers', 8)
    persistent_workers = args_opt.get('persistent_workers', True)
    prefetch_factor = args_opt.get('prefetch_factor', 4)
    pin_memory = args_opt.get('pin_memory', True)

    if dataset_type == 'medmnist3d':
        return make_medmnist3d_dataloader(
            npz_path=args_data.get('dataset_npz'),
            split=split,
            transform=MakeMedMNIST3DTransforms,
            batch_size=1,
            num_workers=num_workers,
            training=False,
            persistent_workers=persistent_workers,
            prefetch_factor=max(2, prefetch_factor // 2),
            pin_memory=pin_memory,
            drop_last=False,
        )

    if dataset_type == 'mridataset':
        return make_dataloader(
            training=False,
            dataset_path=args_data.get('dataset_path'),
            label_path=args_data.get('label_path'),
            transform=MakeMRITransforms,
            batch_size=1,
            defect=args_data.get('defect'),
            num_workers=num_workers,
        )

    raise ValueError(f'Unsupported dataset_type: {dataset_type}')


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def main(args_eval, resume_preempt=False):
    args_pretrain = args_eval.get('pretrain', {})
    args_data = args_eval.get('data', {})
    args_opt = args_eval.get('optimization', {})
    args_draw = args_eval.get('draw', {})

    defect = args_data.get('defect', ['label'])
    defect_classes = args_data.get('defect_classes', [2])

    if args_opt.get('use_plane_path', False):
        raise ValueError('draw pipeline 固定 TokenToVolume3DClassifier，不支持 use_plane_path=true')

    draw_device = args_draw.get('device', None)
    if draw_device is not None:
        device = torch.device(draw_device)
    else:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    logger.info(f'Draw device: {device}')
    logger.info('Draw params:')
    pp.pprint(args_draw)

    pretrain_folder = args_pretrain.get('folder', None)
    ckp_fname = args_pretrain.get('checkpoint', None)
    load_pretrained_ckpt = args_pretrain.get('load_pretrained', True)
    tag = args_pretrain.get('write_tag', 'jepa')
    eval_tag = args_eval.get('tag', None)

    if isinstance(ckp_fname, str) and ckp_fname.strip().lower() in ('', 'none', 'null'):
        ckp_fname = None
    if (not load_pretrained_ckpt) or (ckp_fname is None):
        pretrained_path = None
    else:
        pretrained_path = os.path.join(pretrain_folder, ckp_fname)

    dataset_npz = args_data.get('dataset_npz', None)
    if dataset_npz is not None:
        dataset_name = os.path.splitext(os.path.basename(dataset_npz))[0]
    else:
        raise ValueError('dataset_npz must be provided in args_data for draw pipeline')
    split = args_draw.get('split', args_data.get('split_test', 'test'))

    ckpt_path = _build_ckpt_path(args_eval, dataset_npz, pretrain_folder, tag, eval_tag, args_draw)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f'Checkpoint not found: {ckpt_path}')

    out_dir = args_draw.get('out_dir', None)
    if out_dir is None:
        if dataset_npz is not None:
            dataset_name = os.path.splitext(os.path.basename(dataset_npz))[0]
            base = os.path.join(pretrain_folder, dataset_name)
        else:
            base = pretrain_folder
        model_tag = args_draw.get('model_tag', None)
        if model_tag is not None:
            base = os.path.join(base, model_tag)
        out_dir = os.path.join(base, 'draw')

    pca_dir = os.path.join(out_dir, 'pca')
    cam_dir = os.path.join(out_dir, 'gradcam')
    _ensure_dir(pca_dir)
    _ensure_dir(cam_dir)

    # build model (固定 TokenToVolume3DClassifier)
    deviation_model, unfreeze, depth_patch_size, patch_size = _build_deviation_model(
        device,
        args_pretrain,
        pretrained_path,
        load_pretrained_ckpt,
        args_data.get('mask_size', '(4,64,64)'),
    )

    classifier = TokenToVolume3DClassifier(
        embed_dim=768,
        defect_classes=defect_classes,
        patch_size=patch_size,
        tubelet_size=depth_patch_size,
        grid_size=args_opt.get('token_3d_grid_size', None),
        use_coord_channels=args_opt.get('token_3d_use_coords', True),
        stem_channels=args_opt.get('token_3d_stem_channels', 128),
        stage_channels=args_opt.get('token_3d_stage_channels', [128, 256, 384]),
        stage_depths=args_opt.get('token_3d_stage_depths', [1, 1, 1]),
        stage_strides=args_opt.get('token_3d_stage_strides', [1, 2, 2]),
        dropout=args_opt.get('token_3d_dropout', 0.1),
    ).to(device)

    checkpoint = torch.load(ckpt_path, map_location='cpu')
    if 'deviation_model' not in checkpoint or 'classifier' not in checkpoint:
        raise KeyError('Checkpoint must contain keys: deviation_model and classifier')

    _safe_load_state_dict(deviation_model, checkpoint['deviation_model'], 'deviation_model')
    _safe_load_state_dict(classifier, checkpoint['classifier'], 'classifier')

    deviation_model.eval()
    classifier.eval()

    # loader + sample selection
    data_loader = _build_data_loader(args_data, args_opt, split)
    selected = _select_one_sample_per_label_level(data_loader, defect_classes)

    required = {(t, c) for t, n in enumerate(defect_classes) for c in range(int(n))}
    missed = sorted(list(required.difference(set(selected.keys()))))
    if len(missed) > 0:
        logger.warning(f'Some label-level samples are missing in split={split}: {missed}')

    results = []
    for (task_idx, cls_idx), (img, label_vec) in sorted(selected.items(), key=lambda x: (x[0][0], x[0][1])):
        img = img.unsqueeze(0).to(device, dtype=torch.float32)

        with torch.no_grad():
            tokens = deviation_model(img)
            if unfreeze >= 0:
                tokens = F.layer_norm(tokens, (tokens.size(-1),))

        volume_shape = tuple(int(v) for v in img.shape[2:])

        # 1) PCA: stem前后
        patch_embed_feat = _extract_patch_embed_volume(deviation_model, img[0].detach().cpu())
        before_stem, after_stem = _extract_token_volume(classifier, tokens, volume_shape)
        pca_path = os.path.join(pca_dir, f'{dataset_name}_class{cls_idx}_pca_patchembed_stem.png')
        _plot_pca_before_after_overlay(
            input_vol=img[0].detach().cpu(),
            patch_embed_feat=None,
            before_feat=before_stem,
            after_feat=after_stem,
            save_path=pca_path,
            title=f'{dataset_name} Class {cls_idx} | PCA Before/After Stem',
        )

        # 2) Grad-CAM: 三阶段等值面
        tokens_cam = tokens.detach().clone().requires_grad_(True)
        stage_cams = _compute_stage_cams(
            classifier=classifier,
            tokens=tokens_cam,
            volume_shape=volume_shape,
            task_idx=task_idx,
            target_class=cls_idx,
        )
        cam_path = os.path.join(cam_dir, f'{dataset_name}_class{cls_idx}_gradcam_isosurfaces.png')
        _plot_gradcam_isosurfaces(
            stage_cams,
            cam_path,
            title=f'Task {task_idx} Class {cls_idx} | Multi-scale Grad-CAM Isosurfaces',
            input_vol=img[0].detach().cpu(),
        )

        results.append({
            'task': int(task_idx),
            'class': int(cls_idx),
            'labels': [int(v) for v in label_vec.view(-1).tolist()],
            'pca_path': pca_path,
            'cam_path': cam_path,
        })
        logger.info(f'Drawn sample -> task={task_idx}, class={cls_idx}')

    payload = {
        'checkpoint_path': ckpt_path,
        'split': split,
        'device': str(device),
        'num_required': len(required),
        'num_drawn': len(results),
        'missing': missed,
        'results': results,
    }

    _ensure_dir(out_dir)
    summary_path = os.path.join(out_dir, 'draw_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    logger.info(f'Draw done. Summary saved to: {summary_path}')
    return payload
