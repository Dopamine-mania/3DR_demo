from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal

import numpy as np


@dataclass(frozen=True)
class ViewSelection:
    mode: Literal["full", "sparse", "limited"]
    indices: np.ndarray  # (K,) int64
    angles_rad: np.ndarray  # (K,) float32


def full_angles(num_projections: int, start_angle_deg: float = 0.0, rotation_ccw: bool = True) -> np.ndarray:
    start = math.radians(float(start_angle_deg))
    step = 2.0 * math.pi / float(num_projections)
    idx = np.arange(num_projections, dtype=np.float32)
    if rotation_ccw:
        return (start + idx * step).astype(np.float32)
    return (start - idx * step).astype(np.float32)


def select_views(
    num_projections: int,
    start_angle_deg: float = 0.0,
    rotation_ccw: bool = True,
    mode: Literal["full", "sparse", "limited"] = "full",
    sparse_keep: int = 90,
    limited_center_deg: float = 180.0,
    limited_span_deg: float = 120.0,
) -> ViewSelection:
    angles = full_angles(num_projections, start_angle_deg=start_angle_deg, rotation_ccw=rotation_ccw)

    if mode == "full":
        idx = np.arange(num_projections, dtype=np.int64)
        return ViewSelection(mode=mode, indices=idx, angles_rad=angles)

    if mode == "sparse":
        keep = int(sparse_keep)
        if keep <= 0 or keep > num_projections:
            raise ValueError(f"sparse_keep must be in [1,{num_projections}], got {keep}")
        step = max(1, int(round(num_projections / keep)))
        idx = np.arange(0, num_projections, step, dtype=np.int64)[:keep]
        return ViewSelection(mode=mode, indices=idx, angles_rad=angles[idx])

    if mode == "limited":
        span = float(limited_span_deg)
        if span <= 0 or span > 360:
            raise ValueError(f"limited_span_deg must be in (0,360], got {span}")
        center = float(limited_center_deg) % 360.0
        lo = (center - span / 2.0) % 360.0
        hi = (center + span / 2.0) % 360.0

        deg = (np.degrees(angles) % 360.0).astype(np.float32)
        if lo <= hi:
            m = (deg >= lo) & (deg <= hi)
        else:
            # wrap-around interval
            m = (deg >= lo) | (deg <= hi)
        idx = np.where(m)[0].astype(np.int64)
        if idx.size == 0:
            raise RuntimeError("Limited-angle selection produced 0 views; check center/span.")
        return ViewSelection(mode=mode, indices=idx, angles_rad=angles[idx])

    raise ValueError(f"Unknown mode: {mode}")

