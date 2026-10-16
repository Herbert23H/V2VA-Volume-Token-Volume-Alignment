import argparse
import math
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch


def _load_first_volume(npz_path: Path) -> np.ndarray:
    with np.load(npz_path, allow_pickle=False) as npz:
        split_key = None
        for k in ("train_images", "val_images", "test_images", "images"):
            if k in npz:
                split_key = k
                break
        if split_key is None:
            raise KeyError(f"No image key found in: {npz_path}")

        arr = npz[split_key]
        if arr.shape[0] == 0:
            raise ValueError(f"Empty split {split_key} in {npz_path}")
        vol = arr[0]

    # Normalize to [D,H,W]
    if vol.ndim == 4 and vol.shape[-1] == 1:
        vol = vol[..., 0]
    elif vol.ndim == 4 and vol.shape[0] == 1:
        vol = vol[0]
    if vol.ndim != 3:
        raise ValueError(f"Unexpected volume shape {vol.shape} in {npz_path}")

    vol = vol.astype(np.float32)
    vmin, vmax = float(vol.min()), float(vol.max())
    if vmax > vmin:
        vol = (vol - vmin) / (vmax - vmin)
    else:
        vol = np.zeros_like(vol, dtype=np.float32)
    return vol


def _downsample_to_patch_grid(vol: np.ndarray, patch_size_xyz: Tuple[int, int, int]) -> np.ndarray:
    pd, ph, pw = patch_size_xyz
    d, h, w = vol.shape
    if d % pd != 0 or h % ph != 0 or w % pw != 0:
        raise ValueError(f"Volume shape {vol.shape} is not divisible by patch size {patch_size_xyz}")

    gd, gh, gw = d // pd, h // ph, w // pw
    vol_ds = vol.reshape(gd, pd, gh, ph, gw, pw).mean(axis=(1, 3, 5))
    return vol_ds


def _sample_block_size(
    generator: torch.Generator,
    grid_shape: Tuple[int, int, int],
    depth_scale: Tuple[float, float],
    spatial_scale: Tuple[float, float],
    aspect_ratio: Tuple[float, float],
) -> Tuple[int, int, int]:
    gd, gh, gw = grid_shape

    r = torch.rand(1, generator=generator).item()
    d_scale = depth_scale[0] + r * (depth_scale[1] - depth_scale[0])
    d = max(1, int(gd * d_scale))

    r = torch.rand(1, generator=generator).item()
    s_scale = spatial_scale[0] + r * (spatial_scale[1] - spatial_scale[0])
    spatial_keep = max(1, int(gh * gw * s_scale))

    r = torch.rand(1, generator=generator).item()
    ar = aspect_ratio[0] + r * (aspect_ratio[1] - aspect_ratio[0])

    h = int(round(math.sqrt(spatial_keep * ar)))
    w = int(round(math.sqrt(spatial_keep / ar)))
    h = max(1, min(h, gh))
    w = max(1, min(w, gw))
    return d, h, w


def _sample_mask_patch(
    grid_shape: Tuple[int, int, int],
    cfg: Dict,
    seed: int,
) -> np.ndarray:
    gd, gh, gw = grid_shape
    npred = int(cfg["num_blocks"])

    g = torch.Generator()
    g.manual_seed(seed)

    b_d, b_h, b_w = _sample_block_size(
        generator=g,
        grid_shape=grid_shape,
        depth_scale=tuple(cfg["depth_scale"]),
        spatial_scale=tuple(cfg["spatial_scale"]),
        aspect_ratio=tuple(cfg["aspect_ratio"]),
    )

    # 1 = context(unmasked), 0 = predictor(masked)
    mask_e = torch.ones((gd, gh, gw), dtype=torch.int32)

    for _ in range(npred):
        top = int(torch.randint(0, gh - b_h + 1, (1,), generator=g).item())
        left = int(torch.randint(0, gw - b_w + 1, (1,), generator=g).item())
        start = int(torch.randint(0, gd - b_d + 1, (1,), generator=g).item())
        block = torch.ones((gd, gh, gw), dtype=torch.int32)
        block[start:start + b_d, top:top + b_h, left:left + b_w] = 0
        mask_e *= block

    # masked indicator
    mask_patch = (mask_e == 0).cpu().numpy()  # [D,H,W]
    return mask_patch


def _draw_cube_edges(ax, n: int) -> None:
    edges = [
        ((0, 0, 0), (n, 0, 0)), ((0, 0, 0), (0, n, 0)), ((0, 0, 0), (0, 0, n)),
        ((n, n, n), (0, n, n)), ((n, n, n), (n, 0, n)), ((n, n, n), (n, n, 0)),
        ((n, 0, 0), (n, n, 0)), ((n, 0, 0), (n, 0, n)),
        ((0, n, 0), (n, n, 0)), ((0, n, 0), (0, n, n)),
        ((0, 0, n), (n, 0, n)), ((0, 0, n), (0, n, n)),
    ]
    for p0, p1 in edges:
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]], color="black", linewidth=0.8, alpha=0.7)


def _render_one(
    vol_ds: np.ndarray,
    mask_patch: np.ndarray,
    out_path: Path,
    title: str,
) -> None:
    # Matplotlib voxels use [X,Y,Z], convert from [D,H,W]
    cube = np.transpose(vol_ds, (2, 1, 0)).astype(np.float32)      # [X,Y,Z]
    mask_cube = np.transpose(mask_patch, (2, 1, 0)).astype(bool)   # [X,Y,Z]

    unmasked_cube = cube.copy()
    unmasked_cube[mask_cube] = 0.0

    # Keep only salient points for readability
    thr = np.percentile(unmasked_cube[unmasked_cube > 0], 70) if np.any(unmasked_cube > 0) else 1.0
    show_unmasked = unmasked_cube >= thr

    xu, yu, zu = np.where(show_unmasked)
    iu = unmasked_cube[show_unmasked]

    xm, ym, zm = np.where(mask_cube)

    fig = plt.figure(figsize=(5.6, 5.6), dpi=150)
    ax = fig.add_subplot(111, projection="3d")

    if xu.size > 0:
        ax.scatter(xu + 0.5, yu + 0.5, zu + 0.5, c=iu, cmap="gray", s=7, alpha=0.24, linewidths=0)

    if xm.size > 0:
        ax.scatter(xm + 0.5, ym + 0.5, zm + 0.5, c="#d62728", s=8, alpha=0.33, linewidths=0)

    n = cube.shape[0]
    _draw_cube_edges(ax, n)

    ax.set_xlim(0, n)
    ax.set_ylim(0, n)
    ax.set_zlim(0, n)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=22, azim=-55)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_zlabel("")

    masked_ratio = float(mask_cube.mean())
    ax.set_title(f"{title}\nmasked={masked_ratio:.1%}", fontsize=9)

    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _make_contact_sheet(image_paths: List[Path], out_path: Path, ncols: int = 4) -> None:
    import matplotlib.image as mpimg

    n = len(image_paths)
    nrows = int(math.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.0, nrows * 4.0), dpi=150)
    if nrows == 1:
        axes = np.array([axes])
    axes = axes.reshape(nrows, ncols)

    for i in range(nrows * ncols):
        r, c = divmod(i, ncols)
        ax = axes[r, c]
        ax.axis("off")
        if i < n:
            img = mpimg.imread(image_paths[i])
            ax.imshow(img)
            ax.set_title(image_paths[i].stem, fontsize=8)

    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Draw 3D perspective cube mask demos for MedMNIST+3D")
    parser.add_argument("--data_dir", type=str, default=str(repo_root / "data"))
    parser.add_argument("--out_dir", type=str, default=str(repo_root / "figures" / "mask_3d_demo"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(data_dir.glob("*.npz"))
    if len(files) == 0:
        raise FileNotFoundError(f"No .npz files found in {data_dir}")

    # Match fine-grain pretrain config
    # num_slices=64, crop_size=64, patch_size=4, depth_patch_size=4 => grid 16x16x16
    patch_size_xyz = (4, 4, 4)  # (D,H,W)
    grid_shape = (16, 16, 16)

    long_cfg = {
        "name": "long",
        "aspect_ratio": [0.25, 4.0],
        "num_blocks": 8,
        "spatial_scale": [0.0625, 0.0625],
        "depth_scale": [0.75, 0.75],
    }
    short_cfg = {
        "name": "short",
        "aspect_ratio": [0.25, 4.0],
        "num_blocks": 8,
        "spatial_scale": [0.75, 0.75],
        "depth_scale": [0.0625, 0.0625],
    }

    saved_paths: List[Path] = []

    for ds_i, npz_path in enumerate(files):
        dataset_name = npz_path.stem
        vol = _load_first_volume(npz_path)            # [64,64,64]
        vol_ds = _downsample_to_patch_grid(vol, patch_size_xyz)  # [16,16,16]

        for cfg_i, cfg in enumerate((long_cfg, short_cfg)):
            seed = args.seed + ds_i * 100 + cfg_i * 17
            mask_patch = _sample_mask_patch(grid_shape=grid_shape, cfg=cfg, seed=seed)

            out_name = f"{dataset_name}_{cfg['name']}_mask_3d.png"
            out_path = out_dir / out_name

            title = f"{dataset_name} | {cfg['name']}-range mask"
            _render_one(vol_ds=vol_ds, mask_patch=mask_patch, out_path=out_path, title=title)
            saved_paths.append(out_path)
            print(f"Saved: {out_path}")

    # Optional overview board
    sheet_path = out_dir / "all_12_masks_contact_sheet.png"
    _make_contact_sheet(saved_paths, sheet_path, ncols=4)
    print(f"Saved contact sheet: {sheet_path}")
    print(f"Done. Total images: {len(saved_paths)}")


if __name__ == "__main__":
    main()
    
#  python tools/medmnist3d/plot_medmnist_mask_3d_demo.py --data_dir data --out_dir figures/mask_3d_demo