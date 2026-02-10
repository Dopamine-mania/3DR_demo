from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter


def mse(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")
    return float(np.mean((a - b) ** 2))


def psnr(a: np.ndarray, b: np.ndarray, data_range: float = 1.0) -> float:
    m = mse(a, b)
    if m <= 0:
        return float("inf")
    return float(20.0 * math.log10(float(data_range) / math.sqrt(m)))


def dice(a: np.ndarray, b: np.ndarray, threshold: float) -> float:
    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")
    aa = a > threshold
    bb = b > threshold
    inter = float(np.logical_and(aa, bb).sum())
    denom = float(aa.sum() + bb.sum())
    if denom == 0:
        return 1.0
    return 2.0 * inter / denom


def _ssim_2d(x: np.ndarray, y: np.ndarray, data_range: float, sigma: float = 1.5) -> float:
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    if x.shape != y.shape:
        raise ValueError(f"Shape mismatch: {x.shape} vs {y.shape}")

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2

    mu_x = gaussian_filter(x, sigma=sigma)
    mu_y = gaussian_filter(y, sigma=sigma)
    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x2 = gaussian_filter(x * x, sigma=sigma) - mu_x2
    sigma_y2 = gaussian_filter(y * y, sigma=sigma) - mu_y2
    sigma_xy = gaussian_filter(x * y, sigma=sigma) - mu_xy

    num = (2.0 * mu_xy + c1) * (2.0 * sigma_xy + c2)
    den = (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2)
    s = num / (den + 1e-12)
    return float(np.mean(s))


def ssim_slicewise(a: np.ndarray, b: np.ndarray, data_range: float = 1.0, sigma: float = 1.5) -> float:
    """
    SSIM averaged over slices for 3D arrays (Z,Y,X), or direct for 2D arrays.
    """
    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")
    if a.ndim == 2:
        return _ssim_2d(a, b, data_range=data_range, sigma=sigma)
    if a.ndim == 3:
        vals = [_ssim_2d(a[i], b[i], data_range=data_range, sigma=sigma) for i in range(a.shape[0])]
        return float(np.mean(vals))
    raise ValueError(f"Unsupported ndim for SSIM: {a.ndim}")


@dataclass(frozen=True)
class MetricResult:
    mse: float
    ssim: float
    psnr: float
    dice: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compute_all(
    a: np.ndarray,
    b: np.ndarray,
    data_range: float = 1.0,
    dice_threshold: float | None = None,
) -> MetricResult:
    m = mse(a, b)
    s = ssim_slicewise(a, b, data_range=data_range)
    p = psnr(a, b, data_range=data_range)
    d = None if dice_threshold is None else dice(a, b, threshold=float(dice_threshold))
    return MetricResult(mse=m, ssim=s, psnr=p, dice=d)

