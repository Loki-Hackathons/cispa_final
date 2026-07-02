#!/usr/bin/env python3
"""Analyze SprintML/MGI image statistics for adversarial L2 attack design."""

from __future__ import annotations

import os
import statistics
import sys
from pathlib import Path

import numpy as np
from datasets import load_dataset
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_PATH = SCRIPT_DIR / "dataset_metrics.txt"
DATASET_ID = "SprintML/MGI"
DEFAULT_LOCAL = Path("/p/scratch/training2625/dougnon1/Loki/MGI/data")
SAMPLE_SIZE = 50


def _resolve_token() -> str | None:
    for key in ("HF_TOKEN", "HUGGING_FACE_API_KEY", "HUGGINGFACE_HUB_TOKEN"):
        value = os.environ.get(key)
        if value:
            return value
    env_file = SCRIPT_DIR.parents[1] / ".env"
    if env_file.is_file():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            name, _, value = line.partition("=")
            if name.strip() in {
                "HF_TOKEN",
                "HUGGING_FACE_API_KEY",
                "HUGGINGFACE_HUB_TOKEN",
            }:
                return value.strip().strip('"').strip("'")
    return None


def _load_from_local(data_dir: Path):
    if not data_dir.is_dir():
        return None
    paths = sorted(data_dir.glob("*.png"))
    if not paths:
        paths = sorted(data_dir.glob("**/*.png"))
    if not paths:
        return None
    return paths


def _load_from_hf(token: str | None):
    kwargs = {"path": DATASET_ID, "split": "train"}
    if token:
        kwargs["token"] = token
    return load_dataset(**kwargs)


def _pil_to_array(img: Image.Image) -> np.ndarray:
    arr = np.asarray(img)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    return arr


def _channel_stats(arr: np.ndarray) -> dict:
    if arr.ndim != 3:
        raise ValueError(f"Expected HxWxC array, got shape {arr.shape}")
    ch_axis = 2 if arr.shape[2] in (1, 3, 4) else -1
    channels = arr.shape[ch_axis]
    means, stds, mins, maxs = [], [], [], []
    for c in range(channels):
        plane = arr[..., c] if ch_axis == 2 else arr[c]
        plane = plane.astype(np.float64)
        means.append(float(plane.mean()))
        stds.append(float(plane.std()))
        mins.append(float(plane.min()))
        maxs.append(float(plane.max()))
    return {
        "channels": channels,
        "mean": means,
        "std": stds,
        "min": mins,
        "max": maxs,
    }


def _looks_like_imagenet_norm(mean: list[float], std: list[float]) -> bool:
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]
    if len(mean) != 3:
        return False
    return all(abs(m - im) < 0.05 for m, im in zip(mean, imagenet_mean)) and all(
        abs(s - is_) < 0.05 for s, is_ in zip(std, imagenet_std)
    )


def _analyze_arrays(arrays: list[np.ndarray]) -> dict:
    shapes = {a.shape for a in arrays}
    dtypes = {str(a.dtype) for a in arrays}
    global_min = min(float(a.min()) for a in arrays)
    global_max = max(float(a.max()) for a in arrays)
    integer_like = all(np.allclose(a, np.round(a)) for a in arrays)

    per_image_stats = [_channel_stats(a) for a in arrays]
    mean_of_means = [
        statistics.mean(s["mean"][c] for s in per_image_stats)
        for c in range(per_image_stats[0]["channels"])
    ]
    mean_of_stds = [
        statistics.mean(s["std"][c] for s in per_image_stats)
        for c in range(per_image_stats[0]["channels"])
    ]

    value_range = "[0, 255] uint8 integers"
    if not integer_like:
        if 0.0 <= global_min and global_max <= 1.0:
            value_range = "[0.0, 1.0] floating-point"
        else:
            value_range = f"floating-point, observed [{global_min:.4f}, {global_max:.4f}]"
    elif global_max <= 1:
        value_range = "[0, 1] integer-like"

    h, w, c = arrays[0].shape if arrays[0].ndim == 3 else (*arrays[0].shape, 1)
    normalization_notes = []
    if _looks_like_imagenet_norm(mean_of_means, mean_of_stds):
        normalization_notes.append(
            "Per-channel stats are consistent with ImageNet normalization "
            "(mean≈[0.485,0.456,0.406], std≈[0.229,0.224,0.225])."
        )
    elif all(abs(m) < 0.15 for m in mean_of_means) and all(
        0.8 < s < 1.2 for s in mean_of_stds
    ):
        normalization_notes.append(
            "Pixel means near 0 and stds near 1 — possible zero-mean / unit-variance normalization."
        )
    elif all(100 < m < 160 for m in mean_of_means):
        normalization_notes.append(
            "Channel means in mid-range — typical of raw RGB in [0,255] without normalization."
        )
    else:
        normalization_notes.append(
            "No standard normalization signature detected in stored pixel values."
        )

    return {
        "num_samples": len(arrays),
        "unique_shapes": sorted(shapes),
        "height": h,
        "width": w,
        "channels": c,
        "dtypes": sorted(dtypes),
        "global_min": global_min,
        "global_max": global_max,
        "integer_like": integer_like,
        "value_range": value_range,
        "per_channel_mean": mean_of_means,
        "per_channel_std": mean_of_stds,
        "normalization_notes": normalization_notes,
    }


def _collect_arrays(local_paths: list[Path] | None, hf_dataset, sample_size: int) -> list[np.ndarray]:
    arrays: list[np.ndarray] = []
    if local_paths is not None:
        step = max(1, len(local_paths) // sample_size)
        for path in local_paths[::step][:sample_size]:
            arrays.append(_pil_to_array(Image.open(path).convert("RGB")))
        return arrays

    n = min(sample_size, len(hf_dataset))
    indices = np.linspace(0, len(hf_dataset) - 1, num=n, dtype=int)
    for idx in indices:
        row = hf_dataset[int(idx)]
        img = row.get("image") if isinstance(row, dict) else row
        if not isinstance(img, Image.Image):
            raise TypeError(f"Unexpected image type at index {idx}: {type(img)}")
        arrays.append(_pil_to_array(img.convert("RGB")))
    return arrays


def _format_report(metrics: dict, source: str) -> str:
    lines = [
        "SprintML/MGI — dataset image metrics",
        "=" * 48,
        f"Source: {source}",
        f"Samples analyzed: {metrics['num_samples']}",
        "",
        "Dimensions",
        f"  Height x Width x Channels: {metrics['height']} x {metrics['width']} x {metrics['channels']}",
        f"  Unique shapes in sample: {metrics['unique_shapes']}",
        f"  Color mode: RGB ({metrics['channels']} channels)",
        "",
        "Pixel value range",
        f"  Dtype(s): {', '.join(metrics['dtypes'])}",
        f"  Global min / max: {metrics['global_min']:.6f} / {metrics['global_max']:.6f}",
        f"  Integer-like storage: {metrics['integer_like']}",
        f"  Interpretation: {metrics['value_range']}",
        "",
        "Per-channel statistics (sample mean of per-image stats)",
        f"  Mean (R,G,B): {[round(x, 4) for x in metrics['per_channel_mean']]}",
        f"  Std  (R,G,B): {[round(x, 4) for x in metrics['per_channel_std']]}",
        "",
        "Normalization assessment",
    ]
    lines.extend(f"  - {note}" for note in metrics["normalization_notes"])
    lines.extend(
        [
            "",
            "Task submission constraint (from spec)",
            "  Submit modified images as 256 x 256 x 3 (npz, float32 in [0,1] typical for torch).",
            "",
            "Class layout in reference filenames",
            "  img_000..299: M (member)",
            "  img_300..599: N (non-member)",
            "  img_600..899: G (generated)",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    token = _resolve_token()
    local_dir = Path(os.environ.get("MGI_DATA_DIR", DEFAULT_LOCAL))
    local_paths = _load_from_local(local_dir)

    if local_paths is not None:
        arrays = _collect_arrays(local_paths, None, SAMPLE_SIZE)
        source = f"local directory {local_dir}"
    else:
        print(f"Loading {DATASET_ID} from Hugging Face (split=train)...", file=sys.stderr)
        hf_dataset = _load_from_hf(token)
        arrays = _collect_arrays(None, hf_dataset, SAMPLE_SIZE)
        source = f"Hugging Face datasets ({DATASET_ID}, split=train)"

    metrics = _analyze_arrays(arrays)
    report = _format_report(metrics, source)
    OUTPUT_PATH.write_text(report)
    print(report)
    print(f"Wrote metrics to {OUTPUT_PATH}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
