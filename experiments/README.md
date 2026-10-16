# Experiment outputs

This directory is the default location for training checkpoints, run logs and
benchmark results. It is **git-ignored**, except for the small artifacts
explicitly whitelisted in `.gitignore`:

| File | Description |
| --- | --- |
| `jepa_r0.csv` | pretraining loss curve (epoch, itr, loss, reg-loss, grad norms, timings) |
| `params-pretrain.yaml` | resolved config of the pretraining run: `ViTB / MedMNIST+3D 64³ / fine-grain` |
| `benchmark_heads.csv` | head comparison: params (M), FLOPs (G), inference latency (ms) |
| `plot_loss_curve.py` | plots `jepa_r0.csv` into `figures/loss_curve.png` |

Everything else produced here (`.pth.tar` weights, per-run CSV logs, confusion
matrices) stays local.

## Layout produced at runtime

```
experiments/
└── ViTB_medmnist3d_pretrain_64_fine-grain/
    ├── jepa-latest.pth.tar          # pretrained encoder (downloaded separately, see README)
    └── adrenalmnist3d_64/
        └── cls_CNN_unfreeze2/
            └── jepa_r0.csv          # downstream evaluation log
```

## Pretrained weights

The pretrained encoder (`jepa-latest.pth.tar`, 1.5 GB) is **not** stored in Git. It is
hosted on the Hugging Face Hub:

**👉 [Herbert23H/V2VA_Pretrain_Checkpoint](https://huggingface.co/Herbert23H/V2VA_Pretrain_Checkpoint)**

```bash
hf download Herbert23H/V2VA_Pretrain_Checkpoint jepa-latest.pth.tar \
  --local-dir experiments/ViTB_medmnist3d_pretrain_64_fine-grain
```

This places it at
`experiments/ViTB_medmnist3d_pretrain_64_fine-grain/jepa-latest.pth.tar`,
which is where the eval configs look. Or point `pretrain.folder` at wherever you keep it.
