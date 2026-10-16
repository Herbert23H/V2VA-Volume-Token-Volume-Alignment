# Notice: provenance and modifications

This repository is a **derivative work** of
[facebookresearch/jepa](https://github.com/facebookresearch/jepa) (V-JEPA),
Copyright (c) Meta Platforms, Inc. and affiliates, licensed under
**Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)**.
The full license text is in `LICENSE`; the original upstream README is preserved
at `docs/JEPA_README.md`.

## What was modified

Modifications and additions by the paper's authors — Zhenghao Duan, Dongyun Lin and
Weiyao Lan (School of Aerospace Engineering, Xiamen University; 2025–2026) — on top of
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

Upstream files (`app/`, `evals/`, `src/` except the files listed above as additions,
`setup.py`, `requirements.txt`, the upstream configs and `docs/`) are Copyright © Meta
Platforms, Inc. and affiliates and remain under **CC BY-NC 4.0**. That license carries two
conditions:

- **Attribution (BY)** — keep the copyright notice, keep the license text, and indicate
  that the material was modified. Satisfied here by `LICENSE`, this file and the per-file
  headers.
- **NonCommercial (NC)** — the material and adaptations of it may not be used for a
  commercial purpose.

CC BY-NC 4.0 has **no ShareAlike clause**, so we were not legally obliged to place the
authors' own contributions under the same license. We do it anyway — the whole repository
is distributed under **CC BY-NC 4.0** — because the main contribution
(`TokenToVolume3DClassifier` in `src/models/attentive_pooler.py`) sits inside an upstream
file and cannot be separated cleanly. **This is a deliberate, conservative choice**, not a
licensing requirement.

Two practical consequences worth knowing:

- Creative Commons itself recommends **against** using its licenses for software;
  "NonCommercial" is defined by the purpose of use, which is far less predictable for
  code than a software license. Any commercial use would require either a re-write that
  removes the JEPA-derived code, or a separate license from Meta.
- This restriction propagates to the published weights, which is why the Hugging Face
  model card is also `cc-by-nc-4.0`.

## Third-party components

| Component | Role here | License |
| --- | --- | --- |
| [facebookresearch/jepa](https://github.com/facebookresearch/jepa) | framework this repo derives from | CC BY-NC 4.0 |
| [MedMNIST+](https://medmnist.com/) 3D dataset | the volumes used for pretraining/evaluation | **CC BY 4.0** (data); Apache-2.0 (their code) |

MedMNIST's license reads: *"The MedMNIST dataset is licensed under Creative Commons
Attribution 4.0 International (CC BY 4.0), except DermaMNIST under CC BY-NC 4.0."* The six
3D subsets used here (Adrenal, Fracture, Nodule, Organ, Synapse, Vessel) are all CC BY 4.0;
DermaMNIST is 2D and is not used.

The dataset itself is **not** redistributed: `data/` is git-ignored and `data/README.md`
points users at the official Zenodo release. Attribution and citation for MedMNIST and for
the source paper of each subset are given in the top-level `README.md`.

## Authorship and ownership

The paper and the additions in this repository are the work of **Zhenghao Duan, Dongyun Lin
and Weiyao Lan** (School of Aerospace Engineering, Xiamen University).

This repository mixes author-owned code with third-party-licensed code. Before any
relicensing, redistribution as part of another product, or non-academic use, **all authors**
— and, where applicable, the authors' institution and any funding body — should agree.

## Paper availability

The paper's PDF is deliberately **not** hosted in this repository. The accepted version is
subject to the publisher's (Springer) terms and the official PRICAI 2026 proceedings are not
published yet; the top-level `README.md` will link the DOI once it exists.
