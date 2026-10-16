import logging
import sys
import warnings
import yaml


import torch

import src.models.vision_transformer as video_vit
import src.models.predictor as vit_pred
from src.models.swin_transformer import build_swin_backbone_3d
from src.models.utils.multimask import MultiMaskWrapper, PredictorMultiMaskWrapper
from src.utils.schedulers import (
    WarmupCosineSchedule,
    CosineWDSchedule)
from src.utils.tensors import trunc_normal_
from torch import nn
import torch.nn.functional as F

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()


class MaskAggregation(nn.Module):
    """
    使用 (4,64,64) patch 的 mask 大小遍历整个序列
    计算 target_encoder output - predictor output = Δz
    concatenate 得 (B, N_patches/Mask_size, pred_embed_dim)
    """

    def __init__(
        self, 
        device,
        encoder,
        predictor,
        target_encoder,
        mask_size,
        num_slices=20,
        crop_size=(320,320),
        depth_patch_size=1,
        spatial_patch_size=16
    ):
        super(MaskAggregation, self).__init__()
        self.device = device
        self.encoder = encoder
        self.predictor = predictor
        self.target_encoder = target_encoder
        self.mask_size = mask_size

        if not isinstance(crop_size, tuple):
            crop_size = (crop_size, crop_size)
        self.crop_size = crop_size
        self.height, self.width = crop_size[0] // spatial_patch_size, crop_size[1] // spatial_patch_size
        self.depth = num_slices // depth_patch_size

    def forward(self, x):
        """ Forward pass with exhaustive masking strategy for evaluation. """
        B, C, D, H, W = x.shape  # 支持 batch size > 1
        dz_list = []
        mask_D, mask_H, mask_W = self.mask_size
        n_mask_D = D // mask_D  # 4
        n_mask_H = H // mask_H  # 4
        n_mask_W = W // mask_W  # 4

        # forward pass
        with torch.no_grad():
            # print("x shape:", x.shape)
            hi = self.target_encoder(x) # 可以移动到mask遍历循环外

        # 共计 64 个 mask
        for t in range(n_mask_D):
            for h in range(n_mask_H):
                for w in range(n_mask_W):
                    # 批量创建 mask
                    mask_e = torch.ones((self.depth, self.height, self.width), dtype=torch.int32)

                    t_start = t * (mask_D // self.encoder.tubelet_size)
                    h_start = h * (mask_H // self.encoder.patch_size)
                    w_start = w * (mask_W // self.encoder.patch_size)

                    t_end = t_start + (mask_D // self.encoder.tubelet_size)
                    h_end = h_start + (mask_H // self.encoder.patch_size)
                    w_end = w_start + (mask_W // self.encoder.patch_size)
                    # print(f"Masking patches from depth {t_start} to {t_end}, height {h_start} to {h_end}, width {w_start} to {w_end}")

                    mask_e[t_start:t_end, h_start:h_end, w_start:w_end] = 0     # torch.Size([20, 20, 20])
                    mask_e = mask_e.flatten()   # torch.Size([8000])

                    mask_p = torch.argwhere(mask_e == 0).squeeze().unsqueeze(0).repeat(B,1).to(self.device)   # [B, N_masked]
                    mask_e = torch.nonzero(mask_e).squeeze().unsqueeze(0).repeat(B,1).to(self.device)        # [B, N_keep]
                    # print("mask shape:", mask_e.shape, mask_p.shape)

                    # forward pass
                    with torch.no_grad():
                        target_feats = apply_mask(hi, mask_p, concat=False)
                        # print("target_feats shape:", target_feats.shape)
                        z = self.encoder(x, mask_e)
                        # print("z shape:", z.shape)
                        pred_feats = self.predictor(z, target_feats, mask_e, mask_p)    # feat shape: torch.Size([1, 64, 1024])
                        
                        # 分别归一化再作差
                        target_feats = F.layer_norm(target_feats, (target_feats.size(-1),))
                        pred_feats = F.layer_norm(pred_feats, (pred_feats.size(-1),))
                        # print("pred_feats shape:", pred_feats.shape)

                    dz = (target_feats - pred_feats)  # (N_patches, pred_embed_dim)
                    # dz = F.layer_norm(dz, (dz.size(-1),))   # mask-wise normalization
                    dz_list.append(dz)
                    

        dz_agg = torch.cat(dz_list, dim=1)  # (B, N_patches, pred_embed_dim)
        # Aggregated Delta z shape: torch.Size([1, 8000, 1024])
        return dz_agg   # onlyval return hi, deleta @ other tasks
    

def load_pretrained(
    encoder,
    pretrained,
    checkpoint_key='target_encoder'# 原文使用target_encoder
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


def init_model_frozen(
    device,
    pretrained,
    model_name,
    load_pretrained_ckpt=True,
    patch_size=16,
    crop_size=224,
    num_slices=64,
    in_chans=3,
    depth_patch_size=1,
    use_sdpa=False,
    use_SiLU=False,
    tight_SiLU=True,
    uniform_power=False,
    use_swin_backbone=False,
    swin_backbone_name='swin_tiny_3d',
    swin_embed_dim=96,
    swin_window_size=(7, 7, 7),
    swin_patch_size=(4, 4, 4),
    swin_depths=(2, 2, 6, 2),
    swin_num_heads=(3, 6, 12, 24),
    swin_drop_rate=0.0,
    swin_attn_drop_rate=0.0,
    swin_drop_path_rate=0.0,
    swin_patch_norm=False,
    swin_use_checkpoint=False,
    swin_use_v2=False,
    pred_depth=6,
    pred_embed_dim=384,
    use_mask_tokens=True,
    num_mask_tokens=1,
    zero_init_mask_tokens=True,
):
    if use_swin_backbone:
        encoder = build_swin_backbone_3d(
            backbone_name=swin_backbone_name,
            in_chans=in_chans,
            embed_dim=swin_embed_dim,
            window_size=swin_window_size,
            patch_size=swin_patch_size,
            depths=swin_depths,
            num_heads=swin_num_heads,
            drop_rate=swin_drop_rate,
            attn_drop_rate=swin_attn_drop_rate,
            drop_path_rate=swin_drop_path_rate,
            patch_norm=swin_patch_norm,
            use_checkpoint=swin_use_checkpoint,
            use_v2=swin_use_v2,
        )
        encoder.to(device)
        predictor = None
        logger.info('Swin backbone enabled; skip loading ViT-style predictor and pretrained encoder weights.')
    else:
        encoder = video_vit.__dict__[model_name](
            img_size=crop_size,
            patch_size=patch_size,
            num_frames=num_slices,
            tubelet_size=depth_patch_size,
            in_chans=in_chans,
            uniform_power=uniform_power,
            use_sdpa=use_sdpa,
            use_SiLU=use_SiLU,
            tight_SiLU=tight_SiLU,
        )
        encoder.to(device)
        if load_pretrained_ckpt and pretrained is not None:
            encoder = load_pretrained(encoder, pretrained, checkpoint_key="target_encoder")
        else:
            logger.info('Skip loading encoder checkpoint; using random initialization.')

        predictor = vit_pred.__dict__['vit_predictor'](
            img_size=crop_size,
            use_mask_tokens=use_mask_tokens,
            patch_size=patch_size,
            num_frames=num_slices,
            tubelet_size=depth_patch_size,
            embed_dim=encoder.embed_dim,    #  不用multimaskwraper无backbone属性
            predictor_embed_dim=pred_embed_dim,
            depth=pred_depth,
            num_heads=encoder.num_heads,
            uniform_power=uniform_power,
            num_mask_tokens=num_mask_tokens,
            zero_init_mask_tokens=zero_init_mask_tokens,
            use_sdpa=use_sdpa,
        )
        predictor.to(device)
        if load_pretrained_ckpt and pretrained is not None:
            predictor = load_pretrained(predictor, pretrained, checkpoint_key='predictor')
        else:
            logger.info('Skip loading predictor checkpoint; using random initialization.')

    return encoder, predictor


def apply_mask(x, mask, concat=True):
    """
    :param x: tensor of shape [B (batch-size), N (num-patches), D (feature-dim)]
    :param mask: list of tensors of shape [B, K] containing indices of K patches in [N] to keep
    """

    # 将 mask_keep 移动到与 x 相同的设备
    mask_keep = mask.unsqueeze(-1).repeat(1, 1, x.size(-1))
    # print(x.shape, mask_keep.shape) torch.Size([1, 8000, 1024]) torch.Size([1, 64, 1024])
    all_x = torch.gather(x, dim=1, index=mask_keep)
    if not concat:
        return all_x

    return torch.cat(all_x, dim=0)
