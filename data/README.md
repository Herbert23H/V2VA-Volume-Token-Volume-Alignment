# Data layout

No dataset is distributed with this repository (all `.npz` / `.nii.gz` / `.dcm`
files are git-ignored). Populate this directory yourself.

```
data/
└── medmnist3d/
    ├── adrenalmnist3d_64.npz
    ├── fracturemnist3d_64.npz
    ├── nodulemnist3d_64.npz
    ├── organmnist3d_64.npz
    ├── synapsemnist3d_64.npz
    └── vesselmnist3d_64.npz
```

## Expected archive format

Each `.npz` is a plain NumPy archive that must contain these keys:

| Key | Shape | Meaning |
| --- | --- | --- |
| `train_images` | `[N, 64, 64, 64, 1]` | training volumes (last dim may be omitted) |
| `train_labels` | `[N, 1]` | training labels |
| `val_images` | `[M, 64, 64, 64, 1]` | validation volumes |
| `val_labels` | `[M, 1]` | validation labels |
| `test_images` | `[K, 64, 64, 64, 1]` | test volumes |
| `test_labels` | `[K, 1]` | test labels |

This is the standard [MedMNIST+](https://medmnist.com/) 3D archive format
(`MedMNIST3D` `.npz` release) resampled to a 64³ volume, hence the `_64` suffix.
Download the official 3D archives and rename them to the names above.

## Verify your data

`tools/medmnist3d/check_data.py` inspects every archive in this directory and
reports shape / dtype / value range / label statistics and a slice-to-slice
variance check that tells you which axes are depth / height / width:

```bash
python tools/medmnist3d/check_data.py
```

## Optional: private 5-modality knee MRI

`src/datasets/mri_dataset*.py` and `src/datasets/dataset.py` also implement a
5-modality (PD/PDSAG/...) knee MRI pipeline used for internal experiments. Those
studies use non-public clinical data and are **not** reproducible from this
repository. The paths they read are parameterised through environment
variables — see the table in the top-level `README.md`.
