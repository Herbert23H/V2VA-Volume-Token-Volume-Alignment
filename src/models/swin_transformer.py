import math
from typing import Sequence

import torch
import torch.nn as nn

from monai.networks.nets.swin_unetr import SwinTransformer as MonaiSwinTransformer


def _to_3tuple(value):
    if isinstance(value, int):
        return (value, value, value)
    if isinstance(value, Sequence):
        values = tuple(int(v) for v in value)
        if len(values) == 3:
            return values
    raise ValueError(f'Expected int or 3-element sequence, got {value!r}')


def _ensure_tuple_sequence(values, expected_len, name):
    if isinstance(values, int):
        return tuple([int(values)] * expected_len)
    if isinstance(values, Sequence):
        values = tuple(int(v) for v in values)
        if len(values) == expected_len:
            return values
    raise ValueError(f'{name} must have length {expected_len}, got {values!r}')


class SwinBackbone3D(nn.Module):
    """3D Swin backbone that returns the final-stage tokens as [B, N, C]."""

    def __init__(
        self,
        in_chans=1,
        embed_dim=96,
        window_size=(7, 7, 7),
        patch_size=(4, 4, 4),
        depths=(2, 2, 6, 2),
        num_heads=(3, 6, 12, 24),
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        patch_norm=False,
        use_checkpoint=False,
        spatial_dims=3,
        downsample='merging',
        use_v2=False,
    ):
        super().__init__()
        self.base_embed_dim = int(embed_dim)
        self.num_layers = len(depths)
        self.patch_size = _to_3tuple(patch_size)
        self.window_size = _to_3tuple(window_size)
        self.tubelet_size = 1
        self.backbone = MonaiSwinTransformer(
            in_chans=in_chans,
            embed_dim=self.base_embed_dim,
            window_size=self.window_size,
            patch_size=self.patch_size,
            depths=_ensure_tuple_sequence(depths, 4, 'depths'),
            num_heads=_ensure_tuple_sequence(num_heads, 4, 'num_heads'),
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            patch_norm=patch_norm,
            use_checkpoint=use_checkpoint,
            spatial_dims=spatial_dims,
            downsample=downsample,
            use_v2=use_v2,
        )
        self.embed_dim = self.backbone.num_features
        self.num_features = self.backbone.num_features
        self.num_heads = int(_ensure_tuple_sequence(num_heads, 4, 'num_heads')[-1])
        self.norm = nn.LayerNorm(self.num_features)
        self.blocks = [
            self.backbone.layers1[0],
            self.backbone.layers2[0],
            self.backbone.layers3[0],
            self.backbone.layers4[0],
        ]

    def get_num_layers(self):
        return len(self.blocks)

    def no_weight_decay(self):
        return {}

    def forward(self, x, masks=None):
        if masks is not None:
            raise ValueError('SwinBackbone3D does not support token masks')

        feature_maps = self.backbone(x, normalize=False)
        # MONAI returns [x0, x1, x2, x3, x4], where x4 may include one more
        # downsampling stage (channel doubled). We use x3 to stay aligned with
        # backbone.num_features and downstream classifier dimensions.
        last_feature = feature_maps[-2] if len(feature_maps) >= 2 else feature_maps[-1]
        if last_feature.dim() == 5:
            tokens = last_feature.flatten(2).transpose(1, 2).contiguous()
        elif last_feature.dim() == 4:
            tokens = last_feature.flatten(2).transpose(1, 2).contiguous()
        else:
            raise ValueError(f'Unexpected Swin feature shape: {tuple(last_feature.shape)}')

        return self.norm(tokens)


def build_swin_backbone_3d(
    backbone_name='swin_tiny_3d',
    in_chans=1,
    embed_dim=None,
    window_size=(7, 7, 7),
    patch_size=(4, 4, 4),
    depths=None,
    num_heads=None,
    mlp_ratio=4.0,
    qkv_bias=True,
    drop_rate=0.0,
    attn_drop_rate=0.0,
    drop_path_rate=0.0,
    patch_norm=False,
    use_checkpoint=False,
    use_v2=False,
):
    name = str(backbone_name).lower()
    presets = {
        'swin_tiny_3d': dict(embed_dim=96, depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24)),
        'swin_small_3d': dict(embed_dim=96, depths=(2, 2, 18, 2), num_heads=(3, 6, 12, 24)),
        'swin_base_3d': dict(embed_dim=112, depths=(2, 2, 20, 2), num_heads=(4, 8, 16, 32)),
    }
    if name not in presets:
        raise ValueError(f'Unsupported Swin backbone: {backbone_name!r}')

    preset = presets[name]
    return SwinBackbone3D(
        in_chans=in_chans,
        embed_dim=int(embed_dim or preset['embed_dim']),
        window_size=window_size,
        patch_size=patch_size,
        depths=depths if depths is not None else preset['depths'],
        num_heads=num_heads if num_heads is not None else preset['num_heads'],
        mlp_ratio=mlp_ratio,
        qkv_bias=qkv_bias,
        drop_rate=drop_rate,
        attn_drop_rate=attn_drop_rate,
        drop_path_rate=drop_path_rate,
        patch_norm=patch_norm,
        use_checkpoint=use_checkpoint,
        use_v2=use_v2,
    )