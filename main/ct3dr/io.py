from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import numpy as np


@dataclass(frozen=True)
class DRMeta:
    header_bytes: int = 128
    dtype: str = "<u2"  # little-endian uint16


def list_dr_files(folder: str | Path) -> list[Path]:
    folder = Path(folder)
    files = sorted(folder.glob("*.DR"))
    return files


def dr_shape_from_file(path: str | Path, header_bytes: int = 128, dtype: str = "<u2") -> tuple[int, int]:
    path = Path(path)
    size = path.stat().st_size - header_bytes
    if size <= 0 or size % np.dtype(dtype).itemsize != 0:
        raise ValueError(f"Unexpected DR file size for {path}")
    n = size // np.dtype(dtype).itemsize
    # Common layout in this dataset: 4096 x 4608
    # Infer a plausible pair by looking for factors near square.
    # If ambiguous, caller should pass explicit shape.
    h = int(round(n**0.5))
    if h > 0 and n % h == 0:
        w = int(n // h)
        return h, w
    # fallback: assume 4096 rows
    h = 4096
    if n % h != 0:
        raise ValueError(f"Cannot infer DR shape from file {path} (pixels={n})")
    w = int(n // h)
    return h, w


def read_dr_memmap(
    path: str | Path,
    shape_hw: tuple[int, int],
    header_bytes: int = 128,
    dtype: str = "<u2",
) -> np.memmap:
    path = Path(path)
    h, w = shape_hw
    return np.memmap(path, dtype=dtype, mode="r", offset=header_bytes, shape=(h, w))


def read_sinogram_u16(path: str | Path, shape: tuple[int, int]) -> np.ndarray:
    path = Path(path)
    arr = np.fromfile(path, dtype="<u2")
    expected = int(np.prod(shape))
    if arr.size != expected:
        raise ValueError(f"Sinogram size mismatch: got {arr.size} expected {expected} for {path}")
    return arr.reshape(shape)


def read_raw_f32(path: str | Path, shape: tuple[int, ...]) -> np.ndarray:
    path = Path(path)
    arr = np.fromfile(path, dtype="<f4")
    expected = int(np.prod(shape))
    if arr.size != expected:
        raise ValueError(f"RAW size mismatch: got {arr.size} expected {expected} for {path}")
    return arr.reshape(shape)


def save_preview_png_u16(img_u16: np.ndarray, out_png: str | Path, downsample: int = 8) -> None:
    from PIL import Image

    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    img = img_u16[::downsample, ::downsample].astype(np.float32)
    lo, hi = np.percentile(img, [0.5, 99.5])
    if hi <= lo:
        hi = lo + 1.0
    img_u8 = np.clip((img - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(img_u8).save(out_png)


def iter_chunks(n: int, chunk: int) -> Iterable[tuple[int, int]]:
    i = 0
    while i < n:
        j = min(n, i + chunk)
        yield i, j
        i = j

