"""Load images, assemble submission arrays, save npz under API size limit."""

from __future__ import annotations

import io
import sys
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

from config import (
    BASE_IMAGES,
    DIRECTION_BLOCKS,
    IMAGE_SIZE,
    TOTAL_IMAGES,
    default_data_dir,
)

API_MAX_BYTES = 200 * 1024 * 1024


def load_reference_images(data_dir: Path | None = None) -> np.ndarray:
    data_dir = data_dir or default_data_dir()
    images = np.empty((BASE_IMAGES, IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
    for i in range(BASE_IMAGES):
        with Image.open(data_dir / f"img_{i:03d}.png") as img:
            images[i] = np.asarray(
                img.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR),
                dtype=np.uint8,
            )
    return images


def build_submission(
    originals: np.ndarray,
    attacked: dict[str, np.ndarray] | None = None,
) -> np.ndarray:
    attacked = attacked or {}
    submission = np.empty((TOTAL_IMAGES, IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
    for name, slot_start, source_start, count, mode in DIRECTION_BLOCKS:
        if mode == "original":
            block = originals[source_start : source_start + count]
        else:
            block = attacked.get(name)
            if block is None:
                print(f"WARNING: missing attack for {name}, using originals", file=sys.stderr)
                block = originals[source_start : source_start + count]
        submission[slot_start : slot_start + count] = block
    return submission


def jpeg_roundtrip_batch(images: np.ndarray, quality: int) -> np.ndarray:
    out = np.empty_like(images)
    for i in range(len(images)):
        im = Image.fromarray(images[i])
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=quality, optimize=True)
        buf.seek(0)
        out[i] = np.asarray(Image.open(buf).convert("RGB"), dtype=np.uint8)
    return out


def jpeg_roundtrip_uint8(image: np.ndarray, quality: int) -> np.ndarray:
    """Single-image JPEG roundtrip (same codec path as save_submission_npz)."""
    return jpeg_roundtrip_batch(image[np.newaxis, ...], quality)[0]


def _write_npz_lzma(path: Path, arrays: dict[str, np.ndarray]) -> None:
    def write_array(zf: zipfile.ZipFile, name: str, arr: np.ndarray) -> None:
        buf = io.BytesIO()
        np.save(buf, arr, allow_pickle=False)
        zf.writestr(name, buf.getvalue())

    with zipfile.ZipFile(
        path, mode="w", compression=zipfile.ZIP_LZMA, compresslevel=9
    ) as zf:
        for key, arr in arrays.items():
            write_array(zf, f"{key}.npy", arr)


def _npz_size(path: Path, images: np.ndarray, names: np.ndarray) -> int:
    _write_npz_lzma(path, {"images": images, "names": names})
    return path.stat().st_size


def save_submission_npz(
    images: np.ndarray,
    output_path: Path,
    *,
    jpeg_quality: int | None = None,
    enforce_api_limit: bool = True,
) -> int:
    """
    Save submission npz with LZMA; optionally JPEG-roundtrip to meet API size cap.

    Returns final file size in bytes.
    """
    if images.shape != (TOTAL_IMAGES, IMAGE_SIZE, IMAGE_SIZE, 3):
        raise ValueError(f"Bad submission shape {images.shape}")
    if images.dtype != np.uint8:
        raise ValueError(f"Expected uint8, got {images.dtype}")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    names = np.array([f"{i:04d}" for i in range(TOTAL_IMAGES)])

    if jpeg_quality is not None:
        payload = jpeg_roundtrip_batch(images, jpeg_quality)
        size = _npz_size(output_path, payload, names)
        print(f"Saved {output_path} ({size / 1e6:.2f} MB, jpeg q={jpeg_quality}, lzma)")
        if enforce_api_limit and size > API_MAX_BYTES:
            print(
                f"ERROR: {size} bytes exceeds API limit {API_MAX_BYTES}",
                file=sys.stderr,
            )
            raise SystemExit(1)
        return size

    size = _npz_size(output_path, images, names)
    if size <= API_MAX_BYTES:
        print(f"Saved {output_path} ({size / 1e6:.2f} MB, raw lzma)")
        return size

    print(
        f"Raw LZMA {size / 1e6:.2f} MB > {API_MAX_BYTES / 1e6:.2f} MB — "
        "JPEG roundtrip search...",
        file=sys.stderr,
    )

    best_q: int | None = None
    best_size = size
    lo, hi = 60, 95
    while lo <= hi:
        q = (lo + hi) // 2
        payload = jpeg_roundtrip_batch(images, q)
        trial_size = _npz_size(output_path, payload, names)
        if trial_size <= API_MAX_BYTES:
            best_q = q
            best_size = trial_size
            lo = q + 1
        else:
            hi = q - 1

    if best_q is None:
        print(
            f"ERROR: could not fit under {API_MAX_BYTES} bytes (smallest tried: {best_size})",
            file=sys.stderr,
        )
        if enforce_api_limit:
            raise SystemExit(1)
        return best_size

    # The trial loop keeps rewriting output_path while searching.
    # Rebuild once at best_q to guarantee the on-disk file matches the reported size.
    payload = jpeg_roundtrip_batch(images, best_q)
    final_size = _npz_size(output_path, payload, names)
    print(f"Saved {output_path} ({final_size / 1e6:.2f} MB, jpeg q={best_q}, lzma)")
    return final_size
