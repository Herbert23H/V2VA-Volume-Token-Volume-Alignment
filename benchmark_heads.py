import argparse
import os
import time
import yaml
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch

from evals.mri_classification_frozen.utils import init_model_frozen
from src.models.attentive_pooler import (
    MultiHeadAttentiveClassifier,
    TokenToVolume3DClassifier,
)


@dataclass
class BenchmarkResult:
    dataset: str
    method: str
    params_m: float
    flops_g: float
    time_ms: float


def _load_yaml(path: str) -> Dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def _make_input(cfg: Dict, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    pretrain = cfg.get("pretrain", {})
    crop_size = int(pretrain.get("crop_size", 64))
    num_slices = int(pretrain.get("num_slices", 64))
    in_chans = int(pretrain.get("in_chans", 1))

    x = torch.randn(
        batch_size,
        in_chans,
        num_slices,
        crop_size,
        crop_size,
        device=device,
        dtype=dtype,
    )
    return x


def _resolve_dtype(cfg: Dict, device: torch.device) -> torch.dtype:
    use_bfloat16 = cfg.get("optimization", {}).get("use_bfloat16", False)
    if device.type == "cuda" and use_bfloat16:
        return torch.bfloat16
    return torch.float32


def _apply_unfreeze_two_layers(encoder: torch.nn.Module) -> None:
    for p in encoder.parameters():
        p.requires_grad = False
    if hasattr(encoder, "blocks"):
        for blk in encoder.blocks[-2:]:
            for p in blk.parameters():
                p.requires_grad = True
    if hasattr(encoder, "norm"):
        for p in encoder.norm.parameters():
            p.requires_grad = True


def _build_models(cfg: Dict, method: str, device: torch.device) -> Tuple[torch.nn.Module, torch.nn.Module]:
    pretrain = cfg.get("pretrain", {})
    args_opt = cfg.get("optimization", {})

    encoder, _ = init_model_frozen(
        device,
        pretrained=os.path.join(pretrain.get("folder", ""), pretrain.get("checkpoint", "")),
        load_pretrained_ckpt=pretrain.get("load_pretrained", True),
        model_name=pretrain.get("model_name", "vit_base"),
        patch_size=pretrain.get("patch_size", 4),
        crop_size=pretrain.get("crop_size", 64),
        num_slices=pretrain.get("num_slices", 64),
        in_chans=pretrain.get("in_chans", 1),
        depth_patch_size=pretrain.get("depth_patch_size", 4),
        use_sdpa=pretrain.get("use_sdpa", True),
        use_SiLU=pretrain.get("use_silu", False),
        tight_SiLU=pretrain.get("tight_silu", True),
        uniform_power=pretrain.get("uniform_power", False),
    )

    encoder.eval()
    _apply_unfreeze_two_layers(encoder)

    defect_classes = cfg.get("data", {}).get("defect_classes", [2])

    if method == "token_3d_cnn":
        classifier = TokenToVolume3DClassifier(
            embed_dim=768,
            defect_classes=defect_classes,
            patch_size=pretrain.get("patch_size", 4),
            tubelet_size=pretrain.get("depth_patch_size", 4),
            grid_size=args_opt.get("token_3d_grid_size", None),
            use_coord_channels=args_opt.get("token_3d_use_coords", True),
            stem_channels=args_opt.get("token_3d_stem_channels", 128),
            stage_channels=args_opt.get("token_3d_stage_channels", [128, 256, 384]),
            stage_depths=args_opt.get("token_3d_stage_depths", [1, 1, 1]),
            stage_strides=args_opt.get("token_3d_stage_strides", [1, 2, 2]),
            dropout=args_opt.get("token_3d_dropout", 0.1),
        ).to(device)
    else:
        classifier = MultiHeadAttentiveClassifier(
            embed_dim=768,
            defect_classes=defect_classes,
            pool_mode=method,
            num_queries=args_opt.get("attn_num_queries", 1),
            num_heads=args_opt.get("attn_num_heads", 8),
            mlp_ratio=args_opt.get("attn_mlp_ratio", 4.0),
            depth=args_opt.get("attn_depth", 2),
            head_cls_num_heads=args_opt.get("head_cls_num_heads", 8),
            head_cls_mlp_ratio=args_opt.get("head_cls_mlp_ratio", 4.0),
            head_cls_depth=args_opt.get("head_cls_depth", 2),
        ).to(device)

    return encoder, classifier


def _count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _count_flops(model: torch.nn.Module, x: torch.Tensor) -> float:
    try:
        from fvcore.nn import FlopCountAnalysis
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "缺少 fvcore，请先安装: pip install fvcore"
        ) from exc

    model.eval()
    with torch.inference_mode():
        flops = FlopCountAnalysis(model, x).total()
    return float(flops)


def _measure_time(model: torch.nn.Module, x: torch.Tensor, warmup: int = 10, iters: int = 50) -> float:
    model.eval()
    with torch.inference_mode():
        for _ in range(warmup):
            _ = model(x)
        if x.device.type == "cuda":
            torch.cuda.synchronize()

        start = time.time()
        for _ in range(iters):
            _ = model(x)
        if x.device.type == "cuda":
            torch.cuda.synchronize()
        end = time.time()

    return (end - start) * 1000.0 / iters


def _make_wrapper(encoder: torch.nn.Module, classifier: torch.nn.Module, use_token_3d: bool):
    class Wrapper(torch.nn.Module):
        def __init__(self, enc, cls, use_token):
            super().__init__()
            self.enc = enc
            self.cls = cls
            self.use_token = use_token

        def forward(self, x):
            feats = self.enc(x)
            if self.use_token:
                return self.cls(feats, volume_shape=tuple(x.shape[2:]))
            return self.cls(feats)

    return Wrapper(encoder, classifier, use_token_3d)


def _discover_configs(config_dir: str) -> List[str]:
    files = []
    for name in os.listdir(config_dir):
        if name.endswith(".yaml") and "_test" not in name:
            files.append(os.path.join(config_dir, name))
    return sorted(files)


def _dataset_name(cfg: Dict) -> str:
    npz = cfg.get("data", {}).get("dataset_npz", "")
    base = os.path.basename(npz)
    return os.path.splitext(base)[0]


def run(config_dir: str, batch_size: int, warmup: int, iters: int, save_csv: str) -> None:
    device = _select_device()
    config_paths = _discover_configs(config_dir)
    methods = ["token_3d_cnn", "attentive", "gap", "head_cls"]

    results: List[BenchmarkResult] = []

    for cfg_path in config_paths:
        cfg = _load_yaml(cfg_path)
        dtype = _resolve_dtype(cfg, device)
        dataset = _dataset_name(cfg)
        for method in methods:
            encoder, classifier = _build_models(cfg, method, device)
            wrapper = _make_wrapper(encoder, classifier, use_token_3d=(method == "token_3d_cnn"))
            wrapper = wrapper.to(device=device, dtype=dtype)

            x = _make_input(cfg, batch_size, device, dtype)
            # params = _count_params(encoder) + _count_params(classifier)
            params = _count_params(classifier)

            print(f"{cfg_path}params: {params}")
            flops = _count_flops(wrapper, x)
            time_ms = _measure_time(wrapper, x, warmup=warmup, iters=iters)

            results.append(
                BenchmarkResult(
                    dataset=dataset,
                    method=method,
                    params_m=params / 1e6,
                    flops_g=flops / 1e9,
                    time_ms=time_ms,
                )
            )

    by_method: Dict[str, List[BenchmarkResult]] = {m: [] for m in methods}
    for r in results:
        by_method[r.method].append(r)

    print("\n=== Per-dataset results ===")
    for r in results:
        print(
            f"{r.dataset:25s} | {r.method:12s} | Params(M): {r.params_m:8.3f} | "
            f"FLOPs(G): {r.flops_g:8.3f} | Time(ms): {r.time_ms:8.3f}"
        )

    print("\n=== Average over 6 datasets ===")
    for method, rows in by_method.items():
        if len(rows) == 0:
            continue
        avg_params = sum(r.params_m for r in rows) / len(rows)
        avg_flops = sum(r.flops_g for r in rows) / len(rows)
        avg_time = sum(r.time_ms for r in rows) / len(rows)
        print(
            f"{method:12s} | Params(M): {avg_params:8.3f} | FLOPs(G): {avg_flops:8.3f} | "
            f"Time(ms): {avg_time:8.3f}"
        )

    if save_csv:
        os.makedirs(os.path.dirname(save_csv), exist_ok=True)
        with open(save_csv, "w") as f:
            f.write("dataset,method,params_m,flops_g,time_ms\n")
            for r in results:
                f.write(f"{r.dataset},{r.method},{r.params_m:.6f},{r.flops_g:.6f},{r.time_ms:.6f}\n")
        print(f"\nSaved: {save_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-dir",
        required=True,
        help="Directory holding the head YAML configs to benchmark.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--iters", type=int, default=96)    # 50 iters 可能不够稳定，改成 96（2x） 来获得更稳定的结果
    parser.add_argument(
        "--save-csv",
        default="experiments/benchmark_heads_headonly.csv",
    )
    args = parser.parse_args()

    run(
        config_dir=args.config_dir,
        batch_size=args.batch_size,
        warmup=args.warmup,
        iters=args.iters,
        save_csv=args.save_csv,
    )
