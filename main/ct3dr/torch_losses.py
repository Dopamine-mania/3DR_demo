from __future__ import annotations

import math
import torch
import torch.nn.functional as F


def _gaussian_kernel_1d(window_size: int, sigma: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    half = window_size // 2
    x = torch.arange(-half, half + 1, device=device, dtype=dtype)
    g = torch.exp(-(x * x) / (2.0 * (sigma * sigma)))
    g = g / torch.sum(g)
    return g


def _gaussian_window_2d(
    window_size: int, sigma: float, channels: int, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    g1 = _gaussian_kernel_1d(window_size, sigma, device=device, dtype=dtype)
    g2 = torch.outer(g1, g1)
    w = g2.view(1, 1, window_size, window_size).repeat(channels, 1, 1, 1)
    return w


def ssim(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    data_range: float = 1.0,
    window_size: int = 11,
    sigma: float = 1.5,
    k1: float = 0.01,
    k2: float = 0.03,
) -> torch.Tensor:
    """
    Differentiable SSIM for tensors shaped (B,1,H,W) in [0,1].
    Returns mean SSIM over batch.
    """
    if x.shape != y.shape:
        raise ValueError(f"SSIM shape mismatch: {tuple(x.shape)} vs {tuple(y.shape)}")
    if x.ndim != 4 or x.shape[1] != 1:
        raise ValueError("SSIM expects (B,1,H,W)")

    B, C, H, W = x.shape
    dev = x.device
    dtype = x.dtype
    win = _gaussian_window_2d(window_size, sigma, channels=C, device=dev, dtype=dtype)
    pad = window_size // 2

    mu_x = F.conv2d(x, win, padding=pad, groups=C)
    mu_y = F.conv2d(y, win, padding=pad, groups=C)
    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x2 = F.conv2d(x * x, win, padding=pad, groups=C) - mu_x2
    sigma_y2 = F.conv2d(y * y, win, padding=pad, groups=C) - mu_y2
    sigma_xy = F.conv2d(x * y, win, padding=pad, groups=C) - mu_xy

    c1 = (k1 * float(data_range)) ** 2
    c2 = (k2 * float(data_range)) ** 2

    num = (2.0 * mu_xy + c1) * (2.0 * sigma_xy + c2)
    den = (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2)
    s = num / (den + 1e-12)
    return s.mean()


def ssim_loss(x: torch.Tensor, y: torch.Tensor, **kwargs) -> torch.Tensor:
    return 1.0 - ssim(x, y, **kwargs)


def dice_coeff(
    p: torch.Tensor,
    t: torch.Tensor,
    *,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Soft Dice coefficient for tensors shaped (B,1,H,W) in [0,1].
    """
    if p.shape != t.shape:
        raise ValueError(f"Dice shape mismatch: {tuple(p.shape)} vs {tuple(t.shape)}")
    if p.ndim != 4 or p.shape[1] != 1:
        raise ValueError("Dice expects (B,1,H,W)")
    p = p.clamp(0.0, 1.0)
    t = t.clamp(0.0, 1.0)
    inter = torch.sum(p * t, dim=(1, 2, 3))
    denom = torch.sum(p, dim=(1, 2, 3)) + torch.sum(t, dim=(1, 2, 3))
    d = (2.0 * inter + eps) / (denom + eps)
    return d.mean()


def dice_loss(p: torch.Tensor, t: torch.Tensor, **kwargs) -> torch.Tensor:
    return 1.0 - dice_coeff(p, t, **kwargs)

