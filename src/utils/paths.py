# Copyright (c) Herbert Hao, Xiamen University, School of Aerospace Engineering.
# All rights reserved.
#
# Central path resolution helpers.
#
# Every path in this repository is kept relative, so the code runs from any clone.
# Use the environment variables below to relocate the dataset tree or the
# experiment output tree without editing any config file.

import os
from pathlib import Path

# ----------------------------------------------------------------------------
# Repository root. This file lives at <repo>/src/utils/paths.py -> parents[2]
# ----------------------------------------------------------------------------
REPO_ROOT = Path(
    os.environ.get('V2VA_REPO_ROOT', Path(__file__).resolve().parents[2])
).resolve()

# Root of the dataset tree. Expected layout:
#   data/
#     medmnist3d/{adrenal,fracture,nodule,organ,synapse,vessel}mnist3d_64.npz
#     mri/          # optional private 5-modality knee MRI (NOT distributed with this repo)
DATA_ROOT = Path(os.environ.get('V2VA_DATA_ROOT', REPO_ROOT / 'data'))

# Where checkpoints, run logs and metrics are written.
EXP_ROOT = Path(os.environ.get('V2VA_EXP_ROOT', REPO_ROOT / 'experiments'))


def medmnist3d_npz(name: str) -> str:
    """Absolute path of a MedMNIST+3D archive, e.g. medmnist3d_npz('organmnist3d_64')."""
    return str(DATA_ROOT / 'medmnist3d' / f'{name}.npz')


def env_str(var: str, default: str) -> str:
    """Return environment variable `var`, falling back to `default` when unset/empty."""
    value = os.environ.get(var)
    return value if value else default
