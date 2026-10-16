import math

from multiprocessing import Value

from logging import getLogger

import torch


_GLOBAL_SEED = 0
logger = getLogger()

class MaskCollator(object):

    def __init__(
            self,
            cfgs_mask,
            crop_size=(224,224),
            num_slices=20,
            patch_size=(16, 16),
            depth_patch_size=1,
    ):
        super(MaskCollator, self).__init__()

        self.mask_generators = []
        for m in cfgs_mask:
            mask_generator = _MaskGenerator(
                crop_size=crop_size,
                num_slices=num_slices,
                spatial_patch_size=patch_size,
                depth_patch_size=depth_patch_size,
                spatial_pred_mask_scale=m.get('spatial_scale'),
                depth_pred_mask_scale=m.get('depth_scale'),
                aspect_ratio=m.get('aspect_ratio'),
                npred=m.get('num_blocks'),
                max_context_slices_ratio=m.get('max_depth_keep', 1.0),
                max_keep=m.get('max_keep', None),
            )
            self.mask_generators.append(mask_generator)
    
    def step(self): # 用于跳到下一个config的mask generator
        for mask_generator in self.mask_generators:
            mask_generator.step()
    
    def __call__(self, batch):

        batch_size = len(batch)
        collated_batch = torch.utils.data.default_collate(batch)

        collated_masks_pred, collated_masks_enc = [], []
        for i, mask_generator in enumerate(self.mask_generators):
            masks_enc, masks_pred = mask_generator(batch_size)
            collated_masks_enc.append(masks_enc)
            collated_masks_pred.append(masks_pred)

        # print("long range mask ratio:", collated_masks_pred[0].shape[1] / collated_masks_enc[0].shape[1], "short range mask ratio:", collated_masks_pred[1].shape[1] / collated_masks_enc[1].shape[1])
        return collated_batch, collated_masks_enc, collated_masks_pred
    

class _MaskGenerator(object):

    def __init__(
        self,
        crop_size=(64,64),
        num_slices=64,
        spatial_patch_size=(16,16),
        depth_patch_size=4,
        spatial_pred_mask_scale=None,
        depth_pred_mask_scale=None,
        aspect_ratio=None,
        npred=None,
        max_context_slices_ratio=None,
        max_keep=None,
    ):
        super(_MaskGenerator, self).__init__()
        if not isinstance(crop_size, tuple):
            crop_size = (crop_size, crop_size)
        self.crop_size = crop_size
        self.height, self.width = crop_size[0] // spatial_patch_size, crop_size[1] // spatial_patch_size
        self.depth = num_slices // depth_patch_size

        self.spatial_patch_size = spatial_patch_size
        self.slice_patch_size = depth_patch_size

        self.aspect_ratio = aspect_ratio
        self.spatial_pred_mask_scale = spatial_pred_mask_scale
        self.depth_pred_mask_scale = depth_pred_mask_scale
        self.npred = npred
        self.max_context_depth = max(1, int(self.depth * max_context_slices_ratio))
        self.max_keep = max_keep
        self._itr_counter = Value('i', -1)

    def step(self):
        i = self._itr_counter
        with i.get_lock():
            i.value += 1
            v = i.value
        return v

    def _sample_block_size(
        self,
        generator,
        depth_scale,
        spatial_scale,
        aspect_ratio_scale
    ):
        # -- Sample temporal block mask scale
        _rand = torch.rand(1, generator=generator).item()
        min_d, max_d = depth_scale
        depth_mask_scale = min_d + _rand * (max_d - min_d)
        d = max(1, int(self.depth * depth_mask_scale))

        # -- Sample spatial block mask scale
        _rand = torch.rand(1, generator=generator).item()
        min_s, max_s = spatial_scale
        spatial_mask_scale = min_s + _rand * (max_s - min_s)
        spatial_num_keep = int(self.height * self.width * spatial_mask_scale)
        spatial_num_keep = max(1, spatial_num_keep)

        # -- Sample block aspect-ratio
        _rand = torch.rand(1, generator=generator).item()
        min_ar, max_ar = aspect_ratio_scale
        aspect_ratio = min_ar + _rand * (max_ar - min_ar)

        # -- Compute block height and width (given scale and aspect-ratio)
        h = int(round(math.sqrt(spatial_num_keep * aspect_ratio)))
        w = int(round(math.sqrt(spatial_num_keep / aspect_ratio)))
        h = max(1, h)
        w = max(1, w)
        h = min(h, self.height)
        w = min(w, self.width)

        return (d, h, w)
    
    def _sample_block_mask(self, b_size):
        d, h, w = b_size
        top = torch.randint(0, self.height - h + 1, (1,))
        left = torch.randint(0, self.width - w + 1, (1,))
        start = torch.randint(0, self.depth - d + 1, (1,))

        mask = torch.ones((self.depth, self.height, self.width), dtype=torch.int32)
        mask[start:start+d, top:top+h, left:left+w] = 0

        # Context mask will only span the first X frames 
        # (X=self.max_context_frames)
        if self.max_context_depth < self.depth:   # 硬约束，上下文mask最多延伸到max_context_duration
            mask[self.max_context_depth:, :, :] = 0  # 之后的全被置0（pred）

        # --
        return mask
    
    def __call__(self, batch_size):
        """
        Create encoder and predictor masks when collating imgs into a batch
        # 1. sample pred block size using seed
        # 2. sample several pred block locations for each image (w/o seed)
        # 3. return pred masks and complement (enc mask)
        """
        seed = self.step()
        g = torch.Generator()
        g.manual_seed(seed)
        p_size = self._sample_block_size(
            generator=g,
            depth_scale=self.depth_pred_mask_scale,
            spatial_scale=self.spatial_pred_mask_scale,
            aspect_ratio_scale=self.aspect_ratio,
        )

        collated_masks_pred, collated_masks_enc = [], []
        min_keep_enc = min_keep_pred = self.depth * self.height * self.width
        for _ in range(batch_size): 
            # batch内每一个样本都生成一对，最后enc和pred按各样本最小长度截取
            # 所以可能有些样本的mask数量会被截断，导致和小于总patch数
            empty_context = True
            while empty_context:

                mask_e = torch.ones((self.depth, self.height, self.width), dtype=torch.int32)
                for _ in range(self.npred):
                    mask_e *= self._sample_block_mask(p_size)
                mask_e = mask_e.flatten()

                mask_p = torch.argwhere(mask_e == 0).squeeze()
                mask_e = torch.nonzero(mask_e).squeeze()

                empty_context = len(mask_e) == 0
                if not empty_context:
                    min_keep_pred = min(min_keep_pred, len(mask_p))
                    min_keep_enc = min(min_keep_enc, len(mask_e))
                    collated_masks_pred.append(mask_p)
                    collated_masks_enc.append(mask_e)

        if self.max_keep is not None:   # 限制上下文信息量
            min_keep_enc = min(min_keep_enc, self.max_keep)

        collated_masks_pred = [cm[:min_keep_pred] for cm in collated_masks_pred]
        collated_masks_pred = torch.utils.data.default_collate(collated_masks_pred)
        # --
        collated_masks_enc = [cm[:min_keep_enc] for cm in collated_masks_enc]
        collated_masks_enc = torch.utils.data.default_collate(collated_masks_enc)

        return collated_masks_enc, collated_masks_pred
