from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import math

import numpy as np


@dataclass(frozen=True)
class ConeVecGeom:
    det_rows: int
    det_cols: int
    det_spacing_y_mm: float
    det_spacing_x_mm: float
    dso_mm: float
    dsd_mm: float
    det_center_offset_x_px: float = 0.0
    det_center_offset_y_px: float = 0.0
    rotation_ccw: bool = True
    start_angle_deg: float = 0.0
    num_projections: int = 0
    angles_rad_override: np.ndarray | None = None

    @property
    def dod_mm(self) -> float:
        return float(self.dsd_mm) - float(self.dso_mm)

    def angles_rad(self) -> np.ndarray:
        if self.angles_rad_override is not None:
            ang = np.asarray(self.angles_rad_override, dtype=np.float32)
            if ang.ndim != 1 or ang.size == 0:
                raise ValueError("angles_rad_override must be a non-empty 1D array")
            return ang
        if self.num_projections <= 0:
            raise ValueError("num_projections must be > 0 when angles_rad_override is None")
        start = math.radians(float(self.start_angle_deg))
        step = 2.0 * math.pi / float(self.num_projections)
        idx = np.arange(self.num_projections, dtype=np.float32)
        if self.rotation_ccw:
            return start + idx * step
        return start - idx * step


def build_cone_vecs(geom: ConeVecGeom) -> np.ndarray:
    """
    Return ASTRA cone_vec vectors of shape (V,12).
    """
    betas = geom.angles_rad()
    vecs = np.zeros((betas.size, 12), dtype=np.float32)

    dso = float(geom.dso_mm)
    dod = float(geom.dod_mm)
    sx = dso * np.cos(betas)
    sy = dso * np.sin(betas)
    sz = np.zeros_like(sx)

    dcx = -dod * np.cos(betas)
    dcy = -dod * np.sin(betas)
    dcz = np.zeros_like(dcx)

    # Detector axes (mm per pixel).
    ux = -np.sin(betas) * float(geom.det_spacing_x_mm)
    uy = np.cos(betas) * float(geom.det_spacing_x_mm)
    uz = np.zeros_like(ux)

    vx = np.zeros_like(ux)
    vy = np.zeros_like(ux)
    vz = np.ones_like(ux) * float(geom.det_spacing_y_mm)

    # Apply detector center offsets (in pixels) by shifting detector center.
    dcx = dcx + float(geom.det_center_offset_x_px) * ux + float(geom.det_center_offset_y_px) * vx
    dcy = dcy + float(geom.det_center_offset_x_px) * uy + float(geom.det_center_offset_y_px) * vy
    dcz = dcz + float(geom.det_center_offset_x_px) * uz + float(geom.det_center_offset_y_px) * vz

    vecs[:, 0] = sx
    vecs[:, 1] = sy
    vecs[:, 2] = sz
    vecs[:, 3] = dcx
    vecs[:, 4] = dcy
    vecs[:, 5] = dcz
    vecs[:, 6] = ux
    vecs[:, 7] = uy
    vecs[:, 8] = uz
    vecs[:, 9] = vx
    vecs[:, 10] = vy
    vecs[:, 11] = vz
    return vecs


def u16_to_line_integral(proj_u16: np.ndarray, i0: float | None = None, eps: float = 1.0) -> np.ndarray:
    """
    Convert raw uint16 intensity to line integral -log(I/I0).
    Assumes higher is brighter (more transmission).
    """
    x = proj_u16.astype(np.float32)
    if i0 is None:
        i0 = float(np.quantile(x, 0.999))
    i0 = max(float(i0), eps)
    x = np.clip(x, eps, i0)
    return (-np.log(x / i0)).astype(np.float32)


def run_fdk_cuda(
    projections_line_integral: np.ndarray,
    vec_geom: ConeVecGeom,
    vol_shape_zyx: tuple[int, int, int],
    voxel_mm: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    import astra

    if not astra.astra.has_feature("cuda"):
        raise RuntimeError("ASTRA CUDA feature is not available in this environment.")

    det_rows, det_cols = int(vec_geom.det_rows), int(vec_geom.det_cols)
    V = int(vec_geom.num_projections)

    # ASTRA expects (rows, angles, cols).
    if projections_line_integral.shape != (V, det_rows, det_cols):
        raise ValueError(
            f"Expected projections shape (V,rows,cols)=({V},{det_rows},{det_cols}); got {projections_line_integral.shape}"
        )
    proj = np.transpose(projections_line_integral, (1, 0, 2)).astype(np.float32, copy=False)

    vecs = build_cone_vecs(vec_geom)
    proj_geom = astra.create_proj_geom("cone_vec", det_rows, det_cols, vecs)

    nz, ny, nx = (int(vol_shape_zyx[0]), int(vol_shape_zyx[1]), int(vol_shape_zyx[2]))
    half_x = (nx * float(voxel_mm)) / 2.0
    half_y = (ny * float(voxel_mm)) / 2.0
    half_z = (nz * float(voxel_mm)) / 2.0
    vol_geom = astra.create_vol_geom(nx, ny, nz, -half_x, half_x, -half_y, half_y, -half_z, half_z)

    proj_id = astra.data3d.create("-sino", proj_geom, proj)
    vol_id = astra.data3d.create("-vol", vol_geom)

    cfg = astra.astra_dict("FDK_CUDA")
    cfg["ProjectionDataId"] = proj_id
    cfg["ReconstructionDataId"] = vol_id

    alg_id = astra.algorithm.create(cfg)
    astra.algorithm.run(alg_id, 1)
    vol = astra.data3d.get(vol_id).astype(np.float32)

    astra.algorithm.delete(alg_id)
    astra.data3d.delete(proj_id)
    astra.data3d.delete(vol_id)

    report = {
        "voxel_mm": float(voxel_mm),
        "vol_shape_zyx": [nz, ny, nx],
        "det_shape": [det_rows, det_cols],
        "num_projections": V,
    }
    return vol, report
