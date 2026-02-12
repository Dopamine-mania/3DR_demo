from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal

import numpy as np
import torch


@dataclass(frozen=True)
class FanbeamGeom:
    num_views: int
    num_det: int
    det_spacing_mm: float
    dso_mm: float
    dsd_mm: float
    det_center_offset_px: float = 0.0
    rotation_ccw: bool = True
    start_angle_rad: float = 0.0
    angles_rad_override: np.ndarray | None = None

    def angles(self, device: torch.device) -> torch.Tensor:
        if self.angles_rad_override is not None:
            ang = torch.as_tensor(self.angles_rad_override, device=device, dtype=torch.float32)
            if ang.ndim != 1 or ang.numel() != self.num_views:
                raise ValueError("angles_rad_override must be 1D with length num_views")
            return ang
        step = 2.0 * math.pi / float(self.num_views)
        idx = torch.arange(self.num_views, device=device, dtype=torch.float32)
        if self.rotation_ccw:
            return self.start_angle_rad + idx * step
        return self.start_angle_rad - idx * step

    def det_u_mm(self, device: torch.device) -> torch.Tensor:
        # detector index i -> u (mm), where i=0 is leftmost.
        i = torch.arange(self.num_det, device=device, dtype=torch.float32)
        center = (self.num_det - 1) / 2.0 + float(self.det_center_offset_px)
        return (i - center) * float(self.det_spacing_mm)


def transmission_to_line_integral(
    proj_u16: np.ndarray | torch.Tensor,
    i0: float | None = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    if not isinstance(proj_u16, torch.Tensor):
        proj = torch.from_numpy(proj_u16.astype(np.float32))
    else:
        proj = proj_u16.float()
    if i0 is None:
        i0 = float(torch.quantile(proj.flatten(), 0.999).item())
    proj = torch.clamp(proj, min=eps, max=i0)
    trans = proj / i0
    return (-torch.log(trans)).contiguous()


def _ramp_filter_1d(n: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    # Frequency response |f| for rFFT bins in cycles/sample.
    freqs = torch.fft.rfftfreq(n, d=1.0, device=device)
    return freqs.abs().to(dtype)


def fbp_fanbeam_flat_detector_line_integral(
    proj_line: np.ndarray,
    geom: FanbeamGeom,
    out_size: int,
    out_pixel_mm: float,
    device: Literal["cuda", "cpu"] = "cuda",
    batch_views: int = 16,
    angle_step_rad: float | None = None,
) -> np.ndarray:
    """
    2D fan-beam FBP for a flat detector from line integrals.
    Input shape: (num_views, num_det) float32.
    """
    dev = torch.device(device if (device == "cpu" or torch.cuda.is_available()) else "cpu")
    proj = torch.from_numpy(np.asarray(proj_line, dtype=np.float32)).to(dev)

    assert proj.shape == (geom.num_views, geom.num_det)

    # Pre-weight for flat detector: DSD / sqrt(DSD^2 + u^2)
    u = geom.det_u_mm(dev)
    w_u = float(geom.dsd_mm) / torch.sqrt(float(geom.dsd_mm) ** 2 + u**2)
    proj_w = proj * w_u[None, :]

    # Ramp filter along detector axis.
    n_det = geom.num_det
    H = _ramp_filter_1d(n_det, dev, proj_w.dtype)
    fft = torch.fft.rfft(proj_w, dim=1)
    proj_f = torch.fft.irfft(fft * H[None, :], n=n_det, dim=1)

    # Reconstruction grid in mm, centered at origin.
    coords = (torch.arange(out_size, device=dev, dtype=torch.float32) - (out_size - 1) / 2.0) * float(
        out_pixel_mm
    )
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    x = xx.reshape(-1)  # (P,)
    y = yy.reshape(-1)
    P = x.numel()

    betas = geom.angles(dev)  # (V,)
    if angle_step_rad is None:
        db = torch.diff(betas)
        d_beta = float(torch.median(torch.abs(db)).item()) if db.numel() > 0 else 0.0
    else:
        d_beta = float(angle_step_rad)

    out = torch.zeros((P,), device=dev, dtype=torch.float32)
    det_center = (geom.num_det - 1) / 2.0 + float(geom.det_center_offset_px)

    for v0 in range(0, geom.num_views, batch_views):
        v1 = min(geom.num_views, v0 + batch_views)
        beta = betas[v0:v1]  # (B,)
        cosb = torch.cos(beta)[:, None]
        sinb = torch.sin(beta)[:, None]

        x_prime = x[None, :] * cosb + y[None, :] * sinb  # (B,P)
        y_prime = -x[None, :] * sinb + y[None, :] * cosb  # (B,P)

        denom = float(geom.dso_mm) - x_prime
        denom = torch.clamp(denom, min=1e-3)
        u_mm = float(geom.dsd_mm) * y_prime / denom  # (B,P)

        u_idx = u_mm / float(geom.det_spacing_mm) + det_center
        idx0 = torch.floor(u_idx).to(torch.int64)
        w = (u_idx - idx0.float()).clamp(0.0, 1.0)
        idx0 = idx0.clamp(0, geom.num_det - 1)
        idx1 = (idx0 + 1).clamp(0, geom.num_det - 1)

        pv = proj_f[v0:v1, :]  # (B,nu)
        val0 = torch.gather(pv, 1, idx0)
        val1 = torch.gather(pv, 1, idx1)
        samp = val0 * (1.0 - w) + val1 * w  # (B,P)

        weight = (float(geom.dso_mm) / denom) ** 2
        out += (samp * weight).sum(dim=0)

    out = out * float(d_beta)
    img = out.reshape(out_size, out_size).detach().float().cpu().numpy()
    return img.astype(np.float32)


def fbp_fanbeam_flat_detector(
    sino_u16: np.ndarray,
    geom: FanbeamGeom,
    out_size: int,
    out_pixel_mm: float,
    device: Literal["cuda", "cpu"] = "cuda",
    batch_views: int = 16,
    i0: float | None = None,
) -> np.ndarray:
    """
    2D fan-beam FBP for a flat detector (good enough baseline).
    Input sinogram shape: (num_views, num_det) uint16.
    Output: (out_size, out_size) float32.
    """
    dev = torch.device(device if (device == "cpu" or torch.cuda.is_available()) else "cpu")

    assert sino_u16.shape == (geom.num_views, geom.num_det)
    proj = transmission_to_line_integral(sino_u16, i0=i0).to(dev)

    # Pre-weight for flat detector: DSD / sqrt(DSD^2 + u^2)
    u = geom.det_u_mm(dev)
    w_u = float(geom.dsd_mm) / torch.sqrt(float(geom.dsd_mm) ** 2 + u**2)
    proj_w = proj * w_u[None, :]

    # Ramp filter along detector axis.
    n_det = geom.num_det
    H = _ramp_filter_1d(n_det, dev, proj_w.dtype)
    fft = torch.fft.rfft(proj_w, dim=1)
    proj_f = torch.fft.irfft(fft * H[None, :], n=n_det, dim=1)

    # Reconstruction grid in mm, centered at origin.
    coords = (torch.arange(out_size, device=dev, dtype=torch.float32) - (out_size - 1) / 2.0) * float(
        out_pixel_mm
    )
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    x = xx.reshape(-1)  # (P,)
    y = yy.reshape(-1)
    P = x.numel()

    betas = geom.angles(dev)  # (V,)
    d_beta = 2.0 * math.pi / float(geom.num_views)

    # Accumulate in batches of views to limit memory.
    out = torch.zeros((P,), device=dev, dtype=torch.float32)
    det_center = (geom.num_det - 1) / 2.0 + float(geom.det_center_offset_px)

    for v0 in range(0, geom.num_views, batch_views):
        v1 = min(geom.num_views, v0 + batch_views)
        beta = betas[v0:v1]  # (B,)
        cosb = torch.cos(beta)[:, None]
        sinb = torch.sin(beta)[:, None]

        # Rotate points into view frame: x' along source->origin axis.
        x_prime = x[None, :] * cosb + y[None, :] * sinb  # (B,P)
        y_prime = -x[None, :] * sinb + y[None, :] * cosb  # (B,P)

        denom = float(geom.dso_mm) - x_prime
        denom = torch.clamp(denom, min=1e-3)
        u_mm = float(geom.dsd_mm) * y_prime / denom  # (B,P)

        u_idx = u_mm / float(geom.det_spacing_mm) + det_center
        idx0 = torch.floor(u_idx).to(torch.int64)
        w = (u_idx - idx0.float()).clamp(0.0, 1.0)
        idx0 = idx0.clamp(0, geom.num_det - 1)
        idx1 = (idx0 + 1).clamp(0, geom.num_det - 1)

        pv = proj_f[v0:v1, :]  # (B,nu)
        val0 = torch.gather(pv, 1, idx0)
        val1 = torch.gather(pv, 1, idx1)
        samp = val0 * (1.0 - w) + val1 * w  # (B,P)

        # Distance weighting for backprojection.
        weight = (float(geom.dso_mm) / denom) ** 2
        contrib = (samp * weight).sum(dim=0)  # (P,)
        out += contrib

    out = out * float(d_beta)
    img = out.reshape(out_size, out_size).detach().float().cpu().numpy()
    return img.astype(np.float32)
