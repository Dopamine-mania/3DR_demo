from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import json
import numpy as np
from scipy import ndimage as ndi

from .io import list_dr_files, read_dr_memmap, iter_chunks


@dataclass(frozen=True)
class CropBox:
    r0: int
    r1: int  # exclusive
    c0: int
    c1: int  # exclusive

    @property
    def h(self) -> int:
        return self.r1 - self.r0

    @property
    def w(self) -> int:
        return self.c1 - self.c0


def estimate_object_bbox(
    dr_folder: str | Path,
    shape_hw: tuple[int, int],
    sample_indices: list[int] | None = None,
    downsample: int = 8,
    thr: float = 0.75,
    pad_px: int = 64,
) -> CropBox:
    dr_folder = Path(dr_folder)
    files = list_dr_files(dr_folder)
    if not files:
        raise FileNotFoundError(f"No .DR files under {dr_folder}")

    n = len(files)
    if sample_indices is None:
        sample_indices = list(range(0, n, max(1, n // 10)))[:10]
    sample_indices = [i for i in sample_indices if 0 <= i < n]
    if not sample_indices:
        sample_indices = [0]

    imgs = []
    for idx in sample_indices:
        mm = read_dr_memmap(files[idx], shape_hw)
        imgs.append(mm[::downsample, ::downsample].astype(np.float32))
    med = np.median(np.stack(imgs, 0), 0)

    lo, hi = np.percentile(med, [1, 99])
    if hi <= lo:
        hi = lo + 1.0
    norm = np.clip((med - lo) / (hi - lo), 0.0, 1.0)

    # Object appears darker than background -> keep below threshold.
    mask = norm < thr
    mask = ndi.binary_opening(mask, structure=np.ones((3, 3), dtype=bool))
    lab, ncc = ndi.label(mask)
    if ncc <= 0:
        raise RuntimeError("Failed to find object region; try adjusting threshold.")
    sizes = ndi.sum(mask, lab, index=list(range(1, ncc + 1)))
    k = int(np.argmax(sizes)) + 1
    comp = lab == k
    rows = np.where(comp.any(1))[0]
    cols = np.where(comp.any(0))[0]

    r0_small, r1_small = int(rows[0]), int(rows[-1] + 1)
    c0_small, c1_small = int(cols[0]), int(cols[-1] + 1)

    r0 = max(0, r0_small * downsample - pad_px)
    c0 = max(0, c0_small * downsample - pad_px)
    r1 = min(shape_hw[0], r1_small * downsample + pad_px)
    c1 = min(shape_hw[1], c1_small * downsample + pad_px)

    return CropBox(r0=r0, r1=r1, c0=c0, c1=c1)


def crop_dr_dataset_to_npy(
    dr_folder: str | Path,
    out_dir: str | Path,
    shape_hw: tuple[int, int],
    crop: CropBox,
    downsample: int = 1,
    header_bytes: int = 128,
    chunk: int = 8,
) -> Path:
    dr_folder = Path(dr_folder)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = list_dr_files(dr_folder)
    n = len(files)
    if n == 0:
        raise FileNotFoundError(f"No .DR files under {dr_folder}")

    r0, r1, c0, c1 = crop.r0, crop.r1, crop.c0, crop.c1
    if downsample > 1:
        rr = slice(r0, r1, downsample)
        cc = slice(c0, c1, downsample)
        out_h = (r1 - r0 + downsample - 1) // downsample
        out_w = (c1 - c0 + downsample - 1) // downsample
    else:
        rr = slice(r0, r1)
        cc = slice(c0, c1)
        out_h = r1 - r0
        out_w = c1 - c0

    out_path = out_dir / "projections.npy"
    arr = np.lib.format.open_memmap(
        out_path, mode="w+", dtype=np.uint16, shape=(n, out_h, out_w)
    )

    for i0, i1 in iter_chunks(n, chunk):
        for i in range(i0, i1):
            mm = read_dr_memmap(files[i], shape_hw, header_bytes=header_bytes)
            arr[i, :, :] = mm[rr, cc]
        # Flushing frequently can stall on some filesystems; flush periodically.
        if (i1 % max(1, (chunk * 50))) == 0 or i1 == n:
            arr.flush()

    meta = {
        "source_folder": str(dr_folder),
        "num_projections": n,
        "shape_hw": list(shape_hw),
        "crop": asdict(crop),
        "downsample": downsample,
        "output_shape": [n, out_h, out_w],
        "header_bytes": header_bytes,
    }
    (out_dir / "projections_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return out_path
