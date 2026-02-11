from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np
import torch
import torch.nn as nn


class PosEnc3D(nn.Module):
    def __init__(self, num_freqs: int = 6):
        super().__init__()
        self.num_freqs = int(num_freqs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (...,3)
        outs = [x]
        for i in range(self.num_freqs):
            f = (2.0**i) * math.pi
            outs.append(torch.sin(f * x))
            outs.append(torch.cos(f * x))
        return torch.cat(outs, dim=-1)


class MLP3D(nn.Module):
    def __init__(self, in_dim: int, width: int = 128, depth: int = 6):
        super().__init__()
        layers: list[nn.Module] = []
        d = in_dim
        for _ in range(int(depth)):
            layers.append(nn.Linear(d, int(width)))
            layers.append(nn.ReLU(inplace=True))
            d = int(width)
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass(frozen=True)
class ConeBeamRays:
    source_xyz: torch.Tensor  # (V,3)
    det_center_xyz: torch.Tensor  # (V,3)
    det_u_axis_xyz: torch.Tensor  # (V,3) mm per pixel unit along cols
    det_v_axis_xyz: torch.Tensor  # (V,3) mm per pixel unit along rows
    det_spacing_x_mm: float
    det_spacing_y_mm: float
    det_center_offset_x_px: float
    det_center_offset_y_px: float
    det_rows: int
    det_cols: int


def build_cone_beam_rays(
    angles_rad: np.ndarray,
    dso_mm: float,
    dsd_mm: float,
    det_rows: int,
    det_cols: int,
    det_spacing_x_mm: float,
    det_spacing_y_mm: float,
    det_center_offset_x_px: float,
    det_center_offset_y_px: float,
    device: torch.device,
) -> ConeBeamRays:
    """
    Geometry conventions follow ct3dr.astra_fdk.build_cone_vecs.
    """
    ang = torch.as_tensor(np.asarray(angles_rad, dtype=np.float32), device=device)
    cosb = torch.cos(ang)
    sinb = torch.sin(ang)

    dso = float(dso_mm)
    dod = float(dsd_mm) - float(dso_mm)

    # Source position.
    sx = dso * cosb
    sy = dso * sinb
    sz = torch.zeros_like(sx)
    source = torch.stack([sx, sy, sz], dim=-1)  # (V,3)

    # Detector center (before offset).
    dcx = -dod * cosb
    dcy = -dod * sinb
    dcz = torch.zeros_like(dcx)
    det_center = torch.stack([dcx, dcy, dcz], dim=-1)

    # Detector axes (unit vectors).
    u_axis = torch.stack([-sinb, cosb, torch.zeros_like(cosb)], dim=-1)  # tangent
    v_axis = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=torch.float32).view(1, 3).expand_as(u_axis)

    # Apply center offsets in pixel units, converting to mm along axes.
    det_center = det_center + (float(det_center_offset_x_px) * float(det_spacing_x_mm)) * u_axis
    det_center = det_center + (float(det_center_offset_y_px) * float(det_spacing_y_mm)) * v_axis

    return ConeBeamRays(
        source_xyz=source,
        det_center_xyz=det_center,
        det_u_axis_xyz=u_axis,
        det_v_axis_xyz=v_axis,
        det_spacing_x_mm=float(det_spacing_x_mm),
        det_spacing_y_mm=float(det_spacing_y_mm),
        det_center_offset_x_px=float(det_center_offset_x_px),
        det_center_offset_y_px=float(det_center_offset_y_px),
        det_rows=int(det_rows),
        det_cols=int(det_cols),
    )


def _ray_box_intersect_unit(
    o: torch.Tensor,  # (...,3)
    d: torch.Tensor,  # (...,3)
    half_size_xyz: torch.Tensor,  # (3,)
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Intersect parametric ray p(t)=o+t*d with axis-aligned box [-h,+h] for each axis.
    Returns (tmin, tmax, hit_mask). t is in the same units as d.
    """
    h = half_size_xyz.view(*([1] * (o.ndim - 1)), 3)
    # Avoid division by (near) zero while keeping correct scale for typical d.
    d_safe = torch.where(d.abs() < eps, torch.sign(d) * eps, d)
    d_safe = torch.where(d_safe == 0, torch.full_like(d_safe, eps), d_safe)
    inv = 1.0 / d_safe
    t1 = (-h - o) * inv
    t2 = (h - o) * inv
    tmin = torch.maximum(torch.minimum(t1, t2).amax(dim=-1), torch.tensor(0.0, device=o.device))
    tmax = torch.minimum(torch.maximum(t1, t2).amin(dim=-1), torch.tensor(1.0, device=o.device))
    hit = tmax > tmin
    return tmin, tmax, hit


def forward_project_grid(
    model: nn.Module,
    pe: nn.Module,
    rays: ConeBeamRays,
    view_ids: torch.Tensor,  # (B,)
    row_ids: torch.Tensor,  # (R,)
    col_ids: torch.Tensor,  # (C,)
    half_size_mm: tuple[float, float, float],
    n_samples: int = 64,
    normalize_coords: bool = True,
) -> torch.Tensor:
    """
    Compute predicted line integrals for a batch of rays.
    Returns shape (B,R,C).
    """
    dev = view_ids.device
    B = int(view_ids.numel())
    R = int(row_ids.numel())
    C = int(col_ids.numel())

    src = rays.source_xyz[view_ids]  # (B,3)
    dc = rays.det_center_xyz[view_ids]  # (B,3)
    u_axis = rays.det_u_axis_xyz[view_ids]  # (B,3)
    v_axis = rays.det_v_axis_xyz[view_ids]  # (B,3)

    # Detector mm coordinates for selected pixels.
    # row/col index -> offset from detector center in mm.
    col = col_ids.to(torch.float32)
    row = row_ids.to(torch.float32)
    cu = (rays.det_cols - 1) / 2.0
    cv = (rays.det_rows - 1) / 2.0
    u_mm = (col[None, None, :] - cu) * float(rays.det_spacing_x_mm)  # (1,1,C)
    v_mm = (row[None, :, None] - cv) * float(rays.det_spacing_y_mm)  # (1,R,1)

    det = (
        dc[:, None, None, :]
        + u_mm[..., None] * u_axis[:, None, None, :]
        + v_mm[..., None] * v_axis[:, None, None, :]
    )  # (B,R,C,3)

    o = src[:, None, None, :]  # (B,1,1,3)
    d = det - o  # (B,R,C,3)

    # Intersect with AABB in normalized segment parameter t in [0,1].
    half = torch.tensor(list(half_size_mm), device=dev, dtype=torch.float32)
    tmin, tmax, hit = _ray_box_intersect_unit(o, d, half)

    # Sample points along the hit segment.
    S = int(n_samples)
    kk = (torch.arange(S, device=dev, dtype=torch.float32) + 0.5) / float(S)  # (S,)
    seg = (tmax - tmin).clamp(min=0.0)
    t = tmin[..., None] + seg[..., None] * kk.view(1, 1, 1, S)  # (B,R,C,S)
    pts = o[..., None, :] + d[..., None, :] * t[..., None]  # (B,R,C,S,3)

    pts_flat = pts.reshape(-1, 3)
    if normalize_coords:
        pts_flat = pts_flat / torch.tensor(list(half_size_mm), device=dev, dtype=torch.float32)
    feat = pe(pts_flat)
    dens = torch.nn.functional.softplus(model(feat)).reshape(B, R, C, S)  # (B,R,C,S)

    # Mask out rays that miss the box.
    dens = dens * hit[..., None].to(dens.dtype)

    # Step length in mm per sample.
    ray_len = torch.linalg.norm(d, dim=-1)  # (B,R,C)
    step = (ray_len * seg) / float(S)  # (B,R,C)
    pred = (dens.sum(dim=-1) * step).contiguous()
    return pred


def forward_project_pixels(
    model: nn.Module,
    pe: nn.Module,
    rays: ConeBeamRays,
    view_ids: torch.Tensor,  # (B,)
    rows: torch.Tensor,  # (B,K) int64
    cols: torch.Tensor,  # (B,K) int64
    half_size_mm: tuple[float, float, float],
    n_samples: int = 64,
    normalize_coords: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute predicted line integrals for an arbitrary set of detector pixels.
    Returns (pred_integral, length_mm, hit_mask), each shape (B,K).
    """
    dev = view_ids.device
    B = int(view_ids.numel())
    if rows.shape != cols.shape or rows.ndim != 2 or rows.shape[0] != B:
        raise ValueError("rows/cols must be shaped (B,K) and match view_ids")
    K = int(rows.shape[1])

    src = rays.source_xyz[view_ids]  # (B,3)
    dc = rays.det_center_xyz[view_ids]  # (B,3)
    u_axis = rays.det_u_axis_xyz[view_ids]  # (B,3)
    v_axis = rays.det_v_axis_xyz[view_ids]  # (B,3)

    cu = (rays.det_cols - 1) / 2.0
    cv = (rays.det_rows - 1) / 2.0
    u_mm = (cols.to(torch.float32) - cu) * float(rays.det_spacing_x_mm)  # (B,K)
    v_mm = (rows.to(torch.float32) - cv) * float(rays.det_spacing_y_mm)  # (B,K)

    det = dc[:, None, :] + u_mm[:, :, None] * u_axis[:, None, :] + v_mm[:, :, None] * v_axis[:, None, :]  # (B,K,3)
    o = src[:, None, :]  # (B,1,3)
    d = det - o  # (B,K,3)

    half = torch.tensor(list(half_size_mm), device=dev, dtype=torch.float32)
    tmin, tmax, hit = _ray_box_intersect_unit(o, d, half)

    S = int(n_samples)
    kk = (torch.arange(S, device=dev, dtype=torch.float32) + 0.5) / float(S)
    seg = (tmax - tmin).clamp(min=0.0)  # (B,K)
    t = tmin[..., None] + seg[..., None] * kk.view(1, 1, S)  # (B,K,S)
    pts = o[..., None, :] + d[..., None, :] * t[..., None]  # (B,K,S,3)

    pts_flat = pts.reshape(-1, 3)
    if normalize_coords:
        pts_flat = pts_flat / half.view(1, 3)
    feat = pe(pts_flat)
    dens = torch.nn.functional.softplus(model(feat)).reshape(B, K, S)
    dens = dens * hit[..., None].to(dens.dtype)

    ray_len = torch.linalg.norm(d, dim=-1)  # (B,K)
    length_mm = ray_len * seg  # (B,K)
    step_len = length_mm / float(S)
    pred = dens.sum(dim=-1) * step_len
    return pred.contiguous(), length_mm.contiguous(), hit.contiguous()


def tv_loss_grad(
    model: nn.Module,
    pe: nn.Module,
    pts_xyz: torch.Tensor,  # (N,3)
    half_size_mm: tuple[float, float, float],
) -> torch.Tensor:
    pts_xyz = pts_xyz.detach().requires_grad_(True)
    feat = pe(pts_xyz / torch.tensor(list(half_size_mm), device=pts_xyz.device, dtype=torch.float32))
    y = torch.nn.functional.softplus(model(feat)).sum()
    grad = torch.autograd.grad(y, pts_xyz, create_graph=True)[0]
    return torch.mean(torch.abs(grad))


@dataclass(frozen=True)
class TrainNAF3DConfig:
    half_size_mm: tuple[float, float, float]
    n_samples: int = 64
    lr: float = 2e-3
    steps: int = 4000
    batch_views: int = 2
    batch_pixels: int = 2048
    posenc_freqs: int = 6
    mlp_width: int = 128
    mlp_depth: int = 6
    tv_weight: float = 1e-4
    tv_points: int = 2048
    hot_quantile: float = 0.98
    hot_fraction: float = 0.8
    device: str = "cuda"


def train_naf3d(
    proj_line: np.ndarray,  # (V,H,W) float32 line integrals
    angles_rad: np.ndarray,  # (V,)
    *,
    dso_mm: float,
    dsd_mm: float,
    det_spacing_x_mm: float,
    det_spacing_y_mm: float,
    det_center_offset_x_px: float,
    det_center_offset_y_px: float,
    cfg: TrainNAF3DConfig,
    seed: int = 0,
) -> tuple[dict[str, Any], nn.Module, nn.Module, ConeBeamRays]:
    torch.manual_seed(int(seed))
    dev = torch.device(cfg.device if (cfg.device == "cpu" or torch.cuda.is_available()) else "cpu")

    proj = torch.as_tensor(np.asarray(proj_line, dtype=np.float32), device=dev)
    V, H, W = int(proj.shape[0]), int(proj.shape[1]), int(proj.shape[2])

    # Normalize targets to stabilize optimization.
    tgt_scale = float(torch.quantile(proj.flatten(), 0.995).item())
    if not math.isfinite(tgt_scale) or tgt_scale <= 0:
        tgt_scale = 1.0
    proj_n = proj / float(tgt_scale)

    rays = build_cone_beam_rays(
        angles_rad=angles_rad,
        dso_mm=float(dso_mm),
        dsd_mm=float(dsd_mm),
        det_rows=H,
        det_cols=W,
        det_spacing_x_mm=float(det_spacing_x_mm),
        det_spacing_y_mm=float(det_spacing_y_mm),
        det_center_offset_x_px=float(det_center_offset_x_px),
        det_center_offset_y_px=float(det_center_offset_y_px),
        device=dev,
    )

    pe = PosEnc3D(cfg.posenc_freqs).to(dev)
    in_dim = 3 + 6 * cfg.posenc_freqs
    mlp = MLP3D(in_dim=in_dim, width=cfg.mlp_width, depth=cfg.mlp_depth).to(dev)
    opt = torch.optim.Adam(mlp.parameters(), lr=float(cfg.lr))

    half = cfg.half_size_mm
    losses: list[dict[str, float]] = []

    # Precompute "hot" pixels per view for importance sampling.
    hot_thr = float(np.quantile(np.asarray(proj_line, dtype=np.float32).reshape(V, -1), float(cfg.hot_quantile)))
    hot_coords: list[np.ndarray] = []
    for v in range(V):
        coords = np.argwhere(np.asarray(proj_line[v], dtype=np.float32) > hot_thr)
        # coords is (N,2) in (row,col)
        hot_coords.append(coords.astype(np.int64, copy=False))

    for step in range(int(cfg.steps)):
        view_ids = torch.randint(0, V, (int(cfg.batch_views),), device=dev)
        K = int(cfg.batch_pixels)
        k_hot = int(round(K * float(cfg.hot_fraction)))
        k_uni = K - k_hot

        rows_np = np.empty((int(view_ids.numel()), K), dtype=np.int64)
        cols_np = np.empty((int(view_ids.numel()), K), dtype=np.int64)
        view_ids_cpu = view_ids.detach().cpu().numpy().astype(np.int64)
        for bi, v in enumerate(view_ids_cpu):
            hc = hot_coords[int(v)]
            if hc.size == 0 or k_hot <= 0:
                rr = np.random.randint(0, H, size=(K,), dtype=np.int64)
                cc = np.random.randint(0, W, size=(K,), dtype=np.int64)
            else:
                sel_hot = hc[np.random.randint(0, hc.shape[0], size=(k_hot,))]
                rr_hot = sel_hot[:, 0]
                cc_hot = sel_hot[:, 1]
                rr_uni = np.random.randint(0, H, size=(k_uni,), dtype=np.int64) if k_uni > 0 else np.empty((0,), dtype=np.int64)
                cc_uni = np.random.randint(0, W, size=(k_uni,), dtype=np.int64) if k_uni > 0 else np.empty((0,), dtype=np.int64)
                rr = np.concatenate([rr_hot, rr_uni], axis=0)
                cc = np.concatenate([cc_hot, cc_uni], axis=0)
            rows_np[bi] = rr
            cols_np[bi] = cc

        rows = torch.as_tensor(rows_np, device=dev, dtype=torch.int64)
        cols = torch.as_tensor(cols_np, device=dev, dtype=torch.int64)

        pred_int, length_mm, hit = forward_project_pixels(
            mlp,
            pe,
            rays,
            view_ids=view_ids,
            rows=rows,
            cols=cols,
            half_size_mm=half,
            n_samples=int(cfg.n_samples),
            normalize_coords=True,
        )
        tgt_int = proj_n[view_ids, rows, cols] * float(tgt_scale)  # back to original scale

        eps = 1e-6
        pred_avg = pred_int / (length_mm + eps)
        tgt_avg = tgt_int / (length_mm + eps)
        # Only enforce loss on rays intersecting the modeled volume; otherwise target includes
        # contributions from outside the AABB, which the model cannot represent.
        hit_f = hit.to(pred_avg.dtype)
        diff2 = ((pred_avg - tgt_avg) ** 2) * hit_f
        denom = torch.clamp(hit_f.mean(), min=1e-6)
        data_loss = diff2.mean() / denom

        tv = torch.tensor(0.0, device=dev)
        if cfg.tv_weight > 0:
            pts = (torch.rand(int(cfg.tv_points), 3, device=dev) * 2.0 - 1.0) * torch.tensor(
                list(half), device=dev, dtype=torch.float32
            )
            tv = tv_loss_grad(mlp, pe, pts, half_size_mm=half)

        loss = data_loss + float(cfg.tv_weight) * tv

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step % 50 == 0 or step == int(cfg.steps) - 1:
            losses.append(
                {
                    "step": float(step),
                    "data_loss": float(data_loss.item()),
                    "tv": float(tv.item()),
                    "total": float(loss.item()),
                }
            )

    summary: dict[str, Any] = {
        "steps": int(cfg.steps),
        "n_samples": int(cfg.n_samples),
        "batch_views": int(cfg.batch_views),
        "batch_pixels": int(cfg.batch_pixels),
        "tv_weight": float(cfg.tv_weight),
        "half_size_mm": list(cfg.half_size_mm),
        "tgt_scale": float(tgt_scale),
        "hot_quantile": float(cfg.hot_quantile),
        "hot_fraction": float(cfg.hot_fraction),
        "loss_log": losses,
    }
    return summary, mlp, pe, rays


@torch.no_grad()
def render_volume(
    model: nn.Module,
    pe: nn.Module,
    shape_zyx: tuple[int, int, int],
    half_size_mm: tuple[float, float, float],
    device: str = "cuda",
    chunk: int = 131072,
) -> np.ndarray:
    dev = torch.device(device if (device == "cpu" or torch.cuda.is_available()) else "cpu")
    nz, ny, nx = (int(shape_zyx[0]), int(shape_zyx[1]), int(shape_zyx[2]))
    hx, hy, hz = float(half_size_mm[0]), float(half_size_mm[1]), float(half_size_mm[2])

    xs = torch.linspace(-hx, hx, nx, device=dev, dtype=torch.float32)
    ys = torch.linspace(-hy, hy, ny, device=dev, dtype=torch.float32)
    zs = torch.linspace(-hz, hz, nz, device=dev, dtype=torch.float32)
    zz, yy, xx = torch.meshgrid(zs, ys, xs, indexing="ij")
    pts = torch.stack([xx, yy, zz], dim=-1).reshape(-1, 3)

    out = torch.empty((pts.shape[0],), device="cpu", dtype=torch.float32)
    model = model.to(dev)
    pe = pe.to(dev)
    for i in range(0, pts.shape[0], int(chunk)):
        j = min(pts.shape[0], i + int(chunk))
        feat = pe(pts[i:j] / torch.tensor(list(half_size_mm), device=dev, dtype=torch.float32))
        dens = torch.nn.functional.softplus(model(feat)).squeeze(-1)
        out[i:j] = dens.detach().cpu()

    vol = out.reshape(nz, ny, nx).numpy().astype(np.float32)
    return vol
