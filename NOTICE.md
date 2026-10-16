# Notice: provenance and modifications

This repository is a **derivative work** of
[facebookresearch/jepa](https://github.com/facebookresearch/jepa) (V-JEPA),
Copyright (c) Meta Platforms, Inc. and affiliates, licensed under
**Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)**.
The full license text is in `LICENSE`; the original upstream README is preserved
at `docs/JEPA_README.md`.

## What was modified

Modifications and additions by the repository owner (2025–2026), on top of
upstream commit `51c59d5`:

**New / substantially extended**

- `src/models/attentive_pooler.py` — added `TokenToVolume3DClassifier`
  (token → voxel-grid re-folding + residual 3D CNN head), the plane-agnostic
  cross-attention / plane-gate pooling variants, and optional top-k token pooling.
- `src/models/swin_transformer.py` — 3D Swin backbone returning `[B, N, C]` tokens.
- `src/datasets/medmnist3d_dataset.py`, `medmnist3d_frozen.py` — MedMNIST+3D loaders.
- `src/datasets/mri_dataset.py`, `mri_dataset_frozen.py`, `dataset.py`,
  `mrnet_dataset.py` — 5-modality MRI / MRNet pipelines.
- `src/datasets/utils/video/volume_transforms.py` — volume-specific transforms.
- `src/masks/multiblock3d_mri.py` — 3D multi-block masking for medical volumes.
- `app/vjepa/train_mri.py`, `train_mri_stage2.py`, `transforms_mri.py` — 3D MRI
  pretraining entry points.
- `evals/mri_classification_frozen/` — frozen-encoder evaluation, testing and
  figure generation for 3D volumes.
- `benchmark_heads.py`, `tools/medmnist3d/`, `configs/**/medmnist*`.

**Modified upstream files**

- `src/models/predictor.py`, `src/models/vision_transformer.py`,
  `src/models/utils/{modules,patch_embed,pos_embs}.py`,
  `src/masks/{multiblock3d,random_tube}.py`, `src/utils/schedulers.py`,
  `app/main.py`, `app/scaffold.py`, `evals/scaffold.py`, `setup.py`,
  `requirements.txt`.

Upstream config files that are not relevant to medical 3D volumes
(`configs/pretrain/vith16*.yaml`, `configs/evals/vit{l,h}16_*.yaml`) were removed.

## Licensing note

Because this repository is a derivative of CC BY-NC 4.0 code, the repository as
a whole is distributed under **CC BY-NC 4.0**: attribution is required and
**commercial use is not permitted**.

If the owner intends to release their own contributions under a permissive
license (e.g. MIT), the upstream-derived files must be separated from the
owner-authored files first, because the 3D head code in
`src/models/attentive_pooler.py` is entangled with upstream code.
