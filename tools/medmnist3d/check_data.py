import os
from pathlib import Path
import numpy as np


# Default: <repo>/data. Override with V2VA_DATA_ROOT (see data/README.md).
DATA_DIR = Path(
	os.environ.get("V2VA_DATA_ROOT", Path(__file__).resolve().parents[2] / "data")
)


def _format_size(num_bytes: int) -> str:
	if num_bytes < 1024:
		return f"{num_bytes} B"
	for unit in ["KB", "MB", "GB", "TB"]:
		num_bytes /= 1024.0
		if num_bytes < 1024:
			return f"{num_bytes:.2f} {unit}"
	return f"{num_bytes:.2f} PB"


def _describe_array(arr: np.ndarray) -> dict:
	return {
		"shape": tuple(arr.shape),
		"dtype": str(arr.dtype),
		"min": float(np.min(arr)) if arr.size > 0 else None,
		"max": float(np.max(arr)) if arr.size > 0 else None,
	}


def _collect_labels(npz: np.lib.npyio.NpzFile) -> dict:
	label_keys = [
		"train_labels",
		"val_labels",
		"test_labels",
		"labels",
	]
	label_info = {}
	for key in label_keys:
		if key in npz:
			labels = npz[key]
			labels_flat = labels.reshape(-1)
			label_info[key] = {
				"shape": tuple(labels.shape),
				"dtype": str(labels.dtype),
				"min": int(labels_flat.min()) if labels_flat.size > 0 else None,
				"max": int(labels_flat.max()) if labels_flat.size > 0 else None,
				"unique_count": int(np.unique(labels_flat).size) if labels_flat.size > 0 else 0,
			}
	return label_info


def _collect_images(npz: np.lib.npyio.NpzFile) -> dict:
	image_keys = [
		"train_images",
		"val_images",
		"test_images",
		"images",
	]
	image_info = {}
	for key in image_keys:
		if key in npz:
			image_info[key] = _describe_array(npz[key])
	return image_info


def _axis_change_score(volume: np.ndarray, axis: int, max_slices: int = 8) -> float:
	"""Estimate slice-to-slice change along an axis; higher means more variation."""
	if volume.ndim == 4 and volume.shape[-1] == 1:
		volume = volume[..., 0]
	if volume.ndim != 3:
		return float("nan")
	length = volume.shape[axis]
	if length < 2:
		return 0.0
	step = max(1, length // max_slices)
	diffs = []
	for i in range(0, length - step, step):
		sl1 = np.take(volume, i, axis=axis)
		sl2 = np.take(volume, i + step, axis=axis)
		diffs.append(np.mean(np.abs(sl2.astype(np.float32) - sl1.astype(np.float32))))
	return float(np.mean(diffs)) if diffs else 0.0


def _inspect_axes(npz: np.lib.npyio.NpzFile) -> dict:
	for key in ("train_images", "val_images", "test_images", "images"):
		if key in npz:
			arr = npz[key]
			if arr.ndim < 4:
				return {}
			sample = arr[0]
			if sample.ndim == 4 and sample.shape[-1] == 1:
				sample = sample[..., 0]
			if sample.ndim != 3:
				return {}
			scores = {f"axis_{i}": _axis_change_score(sample, i) for i in range(3)}
			return {"sample_shape": tuple(sample.shape), "axis_change_score": scores}
	return {}


def inspect_npz(file_path: Path) -> dict:
	info = {
		"file": str(file_path),
		"size_bytes": file_path.stat().st_size,
		"size_human": _format_size(file_path.stat().st_size),
	}
	with np.load(file_path, allow_pickle=False) as npz:
		info["keys"] = list(npz.keys())
		info["images"] = _collect_images(npz)
		info["labels"] = _collect_labels(npz)
		info["axis_check"] = _inspect_axes(npz)
	return info


def main() -> None:
	if not DATA_DIR.exists():
		raise FileNotFoundError(f"Data dir not found: {DATA_DIR}")

	npz_files = sorted(DATA_DIR.glob("*.npz"))
	if not npz_files:
		print(f"No .npz files found in {DATA_DIR}")
		return

	print(f"Found {len(npz_files)} npz files in {DATA_DIR}")
	for file_path in npz_files:
		dataset_name = file_path.stem
		info = inspect_npz(file_path)
		print("=" * 80)
		print(f"Dataset: {dataset_name}")
		print(f"File: {info['file']}")
		print(f"Size: {info['size_human']} ({info['size_bytes']} bytes)")
		print(f"Keys: {info['keys']}")

		if info["images"]:
			print("Images:")
			for key, desc in info["images"].items():
				print(f"  {key}: shape={desc['shape']}, dtype={desc['dtype']}, min={desc['min']}, max={desc['max']}")
		else:
			print("Images: <none>")

		if info["labels"]:
			print("Labels:")
			for key, desc in info["labels"].items():
				print(
					f"  {key}: shape={desc['shape']}, dtype={desc['dtype']}, "
					f"min={desc['min']}, max={desc['max']}, unique={desc['unique_count']}"
				)
		else:
			print("Labels: <none>")

		if info.get("axis_check"):
			axis = info["axis_check"]
			print("Axis check (higher change score -> likely depth axis):")
			print(f"  sample_shape={axis['sample_shape']}")
			print(f"  axis_change_score={axis['axis_change_score']}")


if __name__ == "__main__":
	main()
