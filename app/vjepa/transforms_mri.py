# Copyright (c) Zhenghao Duan, Xiamen University, School of Aerospace Engineering.
# All rights reserved.
#

import torch
import torchvision.transforms as transforms
import numpy as np


class MakeMRITransforms:
    """Create MRI transforms; when called you can pass a seed for this set of transforms."""
    def __init__(
        self,
        training=None, 
        target_shape=(20, 224, 224),  # D, H, W
        rotation_degrees=15,
        max_scale=0.05,
        max_translate=0.05,
        flip_depth=True,
        flip_width=False,
    ):
        self.target_shape = target_shape
        self.rotation_degrees = rotation_degrees
        self.max_scale = max_scale
        self.max_translate = max_translate
        self.flip_depth = flip_depth
        self.flip_width = flip_width
        self.training = training

    def __call__(self, data, seed=None):
        if self.training:
            transforms = [  # 实例化各子变换
                Normalize(),    
                ResizeAndResample(self.target_shape),
                Random3DScaleTranslate(self.max_scale, self.max_translate),
                RandomFlip(self.flip_depth, self.flip_width), # 感觉会影响位置编码
                Random3DRotation(self.rotation_degrees),
            ]
        else:
            transforms = [  # 实例化各子变换
                Normalize(),    
                ResizeAndResample(self.target_shape),
            ]

        # 调整通道顺序到（D，H，W）
        data = np.transpose(data, (2, 0, 1))  # Now
        # print(f"Data shape after transpose: {data.shape}")

        # 如果输入是 NumPy 数组，转换为 PyTorch 张量
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data)

        # 确保数据是 float 类型（interpolate 需要浮点数）
        if not torch.is_floating_point(data):
            data = data.float()

        # 猜测transpose导致颠倒, 调整到和原图一致,always
        # 没必要手动调整
        # data = data.flip(2)  # 矫正上下颠倒
        # data = data.flip(1)  # 矫正左右颠倒

        # attach seed to transforms that can use it (some transform implementations check self.seed)
        if seed is not None:
            for transform in transforms:
                if hasattr(transform, '__call__') and 'seed' in transform.__call__.__code__.co_varnames:
                    data = transform(data, seed=seed)  # 传递随机种子
                else:
                    data = transform(data)

            return data
        else:
            raise ValueError("Seed must be provided for MRI transforms during training or evaluation.")


class MakeMedMNIST3DTransforms:
    """Transforms for MedMNIST 3D volumes (single-channel)."""

    def __init__(
        self,
        training=None,
        target_shape=(64, 64, 64),
    ):
        self.target_shape = target_shape
        self.training = training

    def __call__(self, data, seed=None):
        if self.training:
            transforms = [
                # Normalize(),
                # ResizeAndResample(self.target_shape),
                
                Random3DScaleTranslate(0.2, 0.2), 
                RandomFlip(),   # 加true控制flip轴
                Random3DRotation(360),

            ]
        else:
            transforms = [
                # Normalize(),
                # ResizeAndResample(self.target_shape),
            ]

        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data)

        if not torch.is_floating_point(data):
            data = data.float()

        for transform in transforms:
            if seed is not None and hasattr(transform, '__call__') and 'seed' in transform.__call__.__code__.co_varnames:
                data = transform(data, seed=seed)
            else:
                data = transform(data)

        return data

class MakeMRNetTransforms:
    """Transforms for MRNet 3D volumes (3-views)."""
    # Horizontal Flip
    # Random Contrast
    # Random Gamma
    # Random Brightness
    def __init__(
        self,
        training=None,
        target_shape=(32, 256, 256), # 检查、通道顺序
    ):
        self.target_shape = target_shape
        self.training = training

    def __call__(self, data, seed=None):
        if self.training:
            transforms = [
                Normalize256(),
                ResizeAndResample(self.target_shape),
                Random3DScaleTranslate(0.15, 0.15), 
                # RandomFlip(flip_depth=True, flip_height=True, flip_width=True),
                Random3DRotation(15),

            ]
        else:
            transforms = [
                Normalize256(),
                ResizeAndResample(self.target_shape),
            ]

        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data)

        if not torch.is_floating_point(data):
            data = data.float()

        for transform in transforms:
            if seed is not None and hasattr(transform, '__call__') and 'seed' in transform.__call__.__code__.co_varnames:
                data = transform(data, seed=seed)
            else:
                data = transform(data)

        return data
    

class Normalize:
    """ Normalize each channel of MRI data to the range [0, 1]. """

    def __call__(self, data):
        """
        :param data: Input tensor of shape (D, H, W).
        :return: Normalized tensor of the same shape.
        """
        min = data.min()
        max = data.max()
        if max > min:  # Avoid division by zero
                data = (data - min) / (max - min)
        return data
    
class Normalize256:
    """ Normalize each channel of MRI data to the range [0, 1]. """

    def __call__(self, data):
        """
        :param data: Input tensor of shape (D, H, W).
        :return: Normalized tensor of the same shape.
        """
        data = data / 255.0
        return data


class ResizeAndResample:
    """ Resize and resample MRI data to a target shape. """

    def __init__(self, target_shape):
        """
        :param target_shape: Tuple of target dimensions (D, H, W).
        """
        self.target_shape = target_shape

    def __call__(self, data):
        """
        :param data: Input tensor of shape (D, H, W).
        :return: Resized tensor of the target shape.
        """
        # factors = [t / s for t, s in zip(self.target_shape, data.shape)]
        # return torch.nn.functional.interpolate(
        #     data.unsqueeze(0).unsqueeze(0),  # Add batch and channel dimensions
        #     scale_factor=factors,
        #     mode='trilinear',
        #     align_corners=False
        # ).squeeze(0).squeeze(0)
        return torch.nn.functional.interpolate(
            data.unsqueeze(0).unsqueeze(0),  # Add batch and channel dimensions
            size=self.target_shape,          # 强制固定目标尺寸，避免取整误差
            mode='trilinear',
            align_corners=False
        ).squeeze(0).squeeze(0)

class Random3DRotation:
    """ Apply random 3D rotation to the volume along the depth axis. """
    def __init__(self, max_degrees):
        """
        :param max_degrees: Maximum rotation angle in degrees.
        """
        self.max_degrees = max_degrees

    def __call__(self, data, seed=None):
        """
        :param data: Input tensor of shape (D, H, W).
        :param seed: Random seed for reproducibility.
        :return: Rotated tensor of the same shape.
        """
        # if seed is not None:
        #     torch.manual_seed(seed)  # 设置随机种子

        if torch.rand(1).item() > 0.5:
            angle = (torch.rand(1).item() * 2 - 1) * self.max_degrees  # Random angle in [-max_degrees, max_degrees]
            # print("Rotation angle:", angle)
            device = data.device
            dtype = data.dtype
            angle_rad = torch.deg2rad(torch.tensor(angle, device=device, dtype=dtype))
            rotation_matrix = torch.tensor([
                [torch.cos(angle_rad), -torch.sin(angle_rad), 0, 0],
                [torch.sin(angle_rad), torch.cos(angle_rad), 0, 0],
                [0, 0, 1, 0]
            ], device=device, dtype=dtype).unsqueeze(0)  # Add batch dimension

            # Create affine grid for 5D input
            grid = torch.nn.functional.affine_grid(
                rotation_matrix,
                size=(1, 1, *data.shape),  # Add batch and channel dimensions
                align_corners=False
            )

            # Apply grid sampling for 5D input
            return torch.nn.functional.grid_sample(
                data.unsqueeze(0).unsqueeze(0),  # Add batch and channel dimensions
                grid,
                mode='bilinear',  # Use bilinear interpolation
                padding_mode='zeros',
                align_corners=False
            ).squeeze(0).squeeze(0)  # Remove batch and channel dimensions
        
        else:
            return data  # No rotation applied


class Random3DScaleTranslate:
    """Apply random scaling and translation to the volume."""

    def __init__(self, max_scale=0.1, max_translate=0.1):
        """
        :param max_scale: Max relative scale change (e.g., 0.05 -> scale in [0.95, 1.05]).
        :param max_translate: Max relative translation (fraction of size, e.g., 0.05).
        """
        self.max_scale = max_scale
        self.max_translate = max_translate

    def __call__(self, data, seed=None):
        """
        :param data: Input tensor of shape (D, H, W).
        :param seed: Random seed for reproducibility.
        :return: Transformed tensor of the same shape.
        """
        # if seed is not None:
        #     torch.manual_seed(seed+1)

        if torch.rand(1).item() > 0.5:
            device = data.device
            dtype = data.dtype

            # Sample scale factors
            scale = 1.0 + (torch.rand(3, device=device, dtype=dtype) * 2 - 1) * self.max_scale
            sx, sy, sz = scale.tolist()

            # Sample translations in normalized coordinates (align_corners=False)
            # translation fraction -> normalized shift in [-2*max_translate, 2*max_translate]
            t = (torch.rand(3, device=device, dtype=dtype) * 2 - 1) * (2 * self.max_translate)
            tx, ty, tz = t.tolist()

            affine = torch.tensor([
                [sx, 0.0, 0.0, tx],
                [0.0, sy, 0.0, ty],
                [0.0, 0.0, sz, tz],
            ], device=device, dtype=dtype).unsqueeze(0)

            grid = torch.nn.functional.affine_grid(
                affine,
                size=(1, 1, *data.shape),
                align_corners=False
            )

            return torch.nn.functional.grid_sample(
                data.unsqueeze(0).unsqueeze(0),
                grid,
                mode='bilinear',
                padding_mode='zeros',
                align_corners=False
            ).squeeze(0).squeeze(0)
        else:
            return data  # No scaling/translation applied


class RandomFlip:
    def __init__(self, flip_depth=False, flip_height=False, flip_width=False):
        """
        :param flip_depth: Whether to randomly flip along the depth axis.
        :param flip_height: Whether to randomly flip along the height axis.
        :param flip_width: Whether to randomly flip along the width axis.
        """
        self.flip_depth = flip_depth
        self.flip_height = flip_height
        self.flip_width = flip_width

    def __call__(self, data, seed = None):
        """
        :param data: Input tensor of shape (D, H, W).
        :param seed: Random seed for reproducibility.
        :return: Flipped tensor of the same shape.
        """
        # if seed is not None:
        #     torch.manual_seed(seed-1)
        # 对于medmnist: 0.5
        if self.flip_depth and torch.rand(1).item() > 0.8:
            data = data.flip(0)
        if self.flip_height and torch.rand(1).item() > 0.8:
            data = data.flip(1)
        if self.flip_width and torch.rand(1).item() > 0.8:
            data = data.flip(2)

        return data

