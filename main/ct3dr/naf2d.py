from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal

import numpy as np
import torch
import torch.nn as nn

from .fanbeam_fbp import FanbeamGeom, transmission_to_line_integral


class PosEnc(nn.Module):
    def __init__(self, num_freqs: int = 6):
        super().__init__()
        self.num_freqs = int(num_freqs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (...,2)
        outs = [x]
        for i in range(self.num_freqs):
            f = 2.0**i * math.pi
            outs.append(torch.sin(f * x))
            outs.append(torch.cos(f * x))
        return torch.cat(outs, dim=-1)


class MLP(nn.Module):
    def __init__(self, in_dim: int, width: int = 128, depth: int = 4):
        super().__init__()
        layers: list[nn.Module] = []
        d = in_dim
        for _ in range(depth):
            layers.append(nn.Linear(d, width))
            layers.append(nn.ReLU(inplace=True))
            d = width
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass(frozen=True)
class NAF2DConfig:
    fov_mm: float
    n_samples: int = 128
    lr: float = 2e-3
    steps: int = 800
    batch_views: int = 16
    batch_det: int = 256
    posenc_freqs: int = 6
    device: Literal["cuda", "cpu"] = "cuda"


@torch.no_grad()
def _ray_points(
    geom: FanbeamGeom,
    beta: torch.Tensor,  # (B,)
    u_mm: torch.Tensor,  # (B,D)
    t: torch.Tensor,  # (S,)
) -> torch.Tensor:
    """
    Return points along each ray inside the object plane.
    Coordinate system: object at origin, source rotates at radius R=DSO.
    For each (beta,u), compute source and detector point, then sample along segment.
    """
    B, D = u_mm.shape
    S = t.numel()
    cosb = torch.cos(beta)[:, None]
    sinb = torch.sin(beta)[:, None]

    R = float(geom.dso_mm)
    DSD = float(geom.dsd_mm)

    # Source position.
    sx = R * cosb
    sy = R * sinb

    # Detector center position.
    dcx = (R - DSD) * cosb
    dcy = (R - DSD) * sinb

    # Detector u-axis (tangent).
    ux = -sinb
    uy = cosb

    # Detector point for each u.
    px = dcx + u_mm * ux
    py = dcy + u_mm * uy

    # Ray: s + t*(p - s), t in [0,1]
    dx = px - sx
    dy = py - sy
    tt = t.view(1, 1, S)

    x = sx[:, :, None] + dx[:, :, None] * tt
    y = sy[:, :, None] + dy[:, :, None] * tt
    pts = torch.stack([x, y], dim=-1)  # (B,D,S,2)
    return pts


def train_naf2d(
    sino_u16: np.ndarray,
    geom: FanbeamGeom,
    cfg: NAF2DConfig,
    seed: int = 0,
) -> tuple[np.ndarray, dict]:
    """
    Prototype 2D neural attenuation field (implicit MLP) fitted to a sinogram.
    Returns a reconstructed 2D slice on a square grid plus training summary.
    """
    torch.manual_seed(int(seed))
    dev = torch.device(cfg.device if (cfg.device == "cpu" or torch.cuda.is_available()) else "cpu")

    assert sino_u16.shape == (geom.num_views, geom.num_det)
    proj = transmission_to_line_integral(sino_u16).to(dev)  # (V,U)

    # Normalize target to stabilize optimization.
    tgt_scale = float(torch.quantile(proj.flatten(), 0.99).item())
    proj_n = proj / max(tgt_scale, 1e-6)

    pe = PosEnc(cfg.posenc_freqs).to(dev)
    in_dim = 2 + 4 * cfg.posenc_freqs
    mlp = MLP(in_dim=in_dim, width=128, depth=4).to(dev)
    opt = torch.optim.Adam(mlp.parameters(), lr=float(cfg.lr))

    betas = geom.angles(dev)  # (V,)
    u_full = geom.det_u_mm(dev)  # (U,)

    # Sample points along ray between source and detector point; restrict to central part.
    # Using [t0,t1] rather than [0,1] reduces wasted samples outside FOV.
    t = torch.linspace(0.1, 0.9, int(cfg.n_samples), device=dev, dtype=torch.float32)

    losses: list[float] = []
    for step in range(int(cfg.steps)):
        views = torch.randint(0, geom.num_views, (int(cfg.batch_views),), device=dev)
        det_idx = torch.randint(0, geom.num_det, (int(cfg.batch_det),), device=dev)

        beta = betas[views]  # (B,)
        u_mm = u_full[det_idx][None, :].expand(beta.shape[0], -1)  # (B,D)

        pts = _ray_points(geom, beta, u_mm, t)  # (B,D,S,2)
        # Convert (x,y) to normalized coords within FOV.
        pts_n = pts / (float(cfg.fov_mm) / 2.0)
        pts_n = torch.clamp(pts_n, -1.2, 1.2)

        feat = pe(pts_n.reshape(-1, 2))
        mu = mlp(feat).reshape(pts.shape[0], pts.shape[1], pts.shape[2])  # (B,D,S)
        mu = torch.nn.functional.softplus(mu)  # non-negative

        # Approx line integral along the sampled segment (scaled).
        # We ignore exact physical path length here; prototype only.
        pred = mu.mean(dim=2)  # (B,D)
        tgt = proj_n[views[:, None], det_idx[None, :]]

        loss = torch.mean((pred - tgt) ** 2)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step % 20 == 0 or step == int(cfg.steps) - 1:
            losses.append(float(loss.item()))

    # Render a slice on a grid.
    out_size = 256
    coords = (torch.arange(out_size, device=dev, dtype=torch.float32) - (out_size - 1) / 2.0) / (
        out_size / 2.0
    )
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    grid = torch.stack([xx, yy], dim=-1).reshape(-1, 2)
    mu = torch.nn.functional.softplus(mlp(pe(grid))).reshape(out_size, out_size)
    img = mu.detach().cpu().numpy().astype(np.float32) * tgt_scale

    summary = {
        "steps": int(cfg.steps),
        "batch_views": int(cfg.batch_views),
        "batch_det": int(cfg.batch_det),
        "n_samples": int(cfg.n_samples),
        "loss_samples": losses,
        "tgt_scale": tgt_scale,
        "out_size": out_size,
    }
    return img, summary

