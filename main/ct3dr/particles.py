from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any
import math

import numpy as np
from scipy import ndimage as ndi


def otsu_threshold(values: np.ndarray, nbins: int = 256) -> float:
    v = np.asarray(values, dtype=np.float32)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return 0.0
    vmin, vmax = float(v.min()), float(v.max())
    if vmax <= vmin:
        return vmin
    hist, bin_edges = np.histogram(v, bins=nbins, range=(vmin, vmax))
    hist = hist.astype(np.float64)
    prob = hist / (hist.sum() + 1e-12)
    omega = np.cumsum(prob)
    mu = np.cumsum(prob * (bin_edges[:-1] + bin_edges[1:]) / 2.0)
    mu_t = mu[-1]
    sigma_b2 = (mu_t * omega - mu) ** 2 / (omega * (1.0 - omega) + 1e-12)
    idx = int(np.argmax(sigma_b2))
    return float((bin_edges[idx] + bin_edges[idx + 1]) / 2.0)


@dataclass(frozen=True)
class Particle:
    label: int
    voxel_count: int
    center_zyx: tuple[float, float, float]
    diameter_mm: float

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["center_zyx"] = list(self.center_zyx)
        return d


def extract_particles(
    volume: np.ndarray,
    voxel_size_mm: float,
    threshold: float | None = None,
    min_voxels: int = 200,
    max_voxels: int = 200_000,
    keep_top_k: int | None = 10,
) -> list[Particle]:
    vol = np.asarray(volume, dtype=np.float32)
    if vol.ndim != 3:
        raise ValueError(f"Expected 3D volume (Z,Y,X); got {vol.shape}")
    if threshold is None:
        # High-density particles -> use Otsu on a robust subset.
        sample = vol[:: max(1, vol.shape[0] // 64), :, :]
        threshold = otsu_threshold(sample)

    mask = vol > float(threshold)
    lab, ncc = ndi.label(mask)
    if ncc == 0:
        return []

    sizes = ndi.sum(mask, lab, index=list(range(1, ncc + 1))).astype(np.int64)
    # Filter by size range to drop the container wall (huge component) and noise (tiny).
    valid = [i + 1 for i, s in enumerate(sizes) if min_voxels <= int(s) <= max_voxels]
    if not valid:
        return []

    # Sort by size desc.
    valid = sorted(valid, key=lambda k: int(sizes[k - 1]), reverse=True)
    if keep_top_k is not None:
        valid = valid[: int(keep_top_k)]

    particles: list[Particle] = []
    voxel_vol_mm3 = float(voxel_size_mm) ** 3
    for k in valid:
        vox = int(sizes[k - 1])
        com = ndi.center_of_mass(mask, labels=lab, index=k)  # z,y,x
        vol_mm3 = vox * voxel_vol_mm3
        # Equivalent sphere diameter.
        diameter = (6.0 * vol_mm3 / math.pi) ** (1.0 / 3.0)
        particles.append(
            Particle(
                label=int(k),
                voxel_count=vox,
                center_zyx=(float(com[0]), float(com[1]), float(com[2])),
                diameter_mm=float(diameter),
            )
        )
    return particles

