from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np

from ct3dr.astra_fdk import u16_to_line_integral
from ct3dr.fanbeam_fbp import FanbeamGeom, fbp_fanbeam_flat_detector_line_integral
from ct3dr.view_sampling import full_angles, select_views


def save_png(img: np.ndarray, out_png: Path) -> None:
    from PIL import Image

    out_png.parent.mkdir(parents=True, exist_ok=True)
    x = img.astype(np.float32)
    lo, hi = np.percentile(x, [0.5, 99.5])
    if hi <= lo:
        hi = lo + 1.0
    u8 = np.clip((x - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(u8).save(out_png)


def main() -> int:
    ap = argparse.ArgumentParser(description="3D FDK-lite: per-detector-row 2D fanbeam FBP stack (no ASTRA).")
    ap.add_argument("--processed_dir", type=Path, default=Path("main/processed/6_ds4_run2"))
    ap.add_argument("--out_dir", type=Path, default=Path("main/output/fdk_lite_full"))
    ap.add_argument("--nx", type=int, default=160, help="X/Y size of reconstructed slices")
    ap.add_argument("--nz", type=int, default=160, help="Number of Z slices (sampled from detector rows)")
    ap.add_argument("--pixel_mm", type=float, default=0.25, help="Pixel size in mm for reconstructed slices")
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")

    ap.add_argument("--view_mode", choices=["full", "sparse", "limited"], default="full")
    ap.add_argument("--sparse_keep", type=int, default=90)
    ap.add_argument("--limited_center_deg", type=float, default=180.0)
    ap.add_argument("--limited_span_deg", type=float, default=120.0)

    ap.add_argument("--target_mode", choices=["raw", "residual"], default="raw")
    args = ap.parse_args()

    processed_dir: Path = args.processed_dir
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    proj_path = processed_dir / "projections.npy"
    meta_proj_path = processed_dir / "projections_meta.json"
    meta_scan_path = processed_dir / "scan_recon_meta.json"
    proj_meta = json.loads(meta_proj_path.read_text(encoding="utf-8"))
    scan_meta = json.loads(meta_scan_path.read_text(encoding="utf-8"))
    scan = scan_meta["scan"]
    recon = scan_meta["recon"]
    crop = proj_meta["crop"]
    ds = int(proj_meta["downsample"])

    det_rows = int(proj_meta["output_shape"][1])
    det_cols = int(proj_meta["output_shape"][2])
    num_full = int(proj_meta["num_projections"])

    sel = select_views(
        num_full,
        start_angle_deg=float(scan["start_angle_deg"]),
        rotation_ccw=bool(scan["rotation_ccw"]),
        mode=str(args.view_mode),
        sparse_keep=int(args.sparse_keep),
        limited_center_deg=float(args.limited_center_deg),
        limited_span_deg=float(args.limited_span_deg),
    )

    full_cols = int(scan["detector_cols"])
    full_rows = int(scan["detector_rows"])
    full_center_u = (full_cols - 1) / 2.0 + float(scan["det_h_center_offset_px"])
    full_center_v = (full_rows - 1) / 2.0 + float(scan["det_v_center_offset_px"])
    center_u_crop = (full_center_u - float(crop["c0"])) / float(ds)
    center_v_crop = (full_center_v - float(crop["r0"])) / float(ds)
    det_offset_x_px = center_u_crop - (det_cols - 1) / 2.0
    det_offset_y_px = center_v_crop - (det_rows - 1) / 2.0

    det_spacing_x = float(recon["det_h_spacing_mm"]) * float(ds)
    # det_spacing_y = float(recon["det_v_spacing_mm"]) * float(ds)  # unused in lite baseline

    proj_u16 = np.load(proj_path, mmap_mode="r")  # (V,H,W)
    proj_u16 = np.asarray(proj_u16[sel.indices, :, :], dtype=np.uint16)

    i0 = float(np.quantile(proj_u16.astype(np.float32), 0.999))
    proj_line = u16_to_line_integral(proj_u16, i0=i0)  # (V,H,W) float32
    if args.target_mode == "residual":
        med = np.median(proj_line.reshape(proj_line.shape[0], -1), axis=1).astype(np.float32)
        proj_line = np.clip(proj_line - med[:, None, None], 0.0, None)

    # Angle step should match the original acquisition spacing for correct scaling.
    angles_full = full_angles(num_full, start_angle_deg=float(scan["start_angle_deg"]), rotation_ccw=bool(scan["rotation_ccw"]))
    if sel.indices.size >= 2:
        angle_step = float(np.median(np.abs(np.diff(angles_full[sel.indices]))))
    else:
        angle_step = 0.0

    geom = FanbeamGeom(
        num_views=int(sel.indices.size),
        num_det=det_cols,
        det_spacing_mm=float(det_spacing_x),
        dso_mm=float(scan["dso_mm"]),
        dsd_mm=float(scan["dsd_mm"]),
        det_center_offset_px=float(det_offset_x_px),
        rotation_ccw=bool(scan["rotation_ccw"]),
        start_angle_rad=float(np.deg2rad(scan["start_angle_deg"])),
        angles_rad_override=sel.angles_rad,
    )

    nx = int(args.nx)
    nz = int(args.nz)
    # Select nz detector rows evenly to form a volume.
    z_rows = np.linspace(0, det_rows - 1, nz).round().astype(np.int64)
    vol = np.zeros((nz, nx, nx), dtype=np.float32)

    for zi, rr in enumerate(z_rows):
        sino = proj_line[:, int(rr), :]  # (V,U)
        img = fbp_fanbeam_flat_detector_line_integral(
            sino,
            geom,
            out_size=nx,
            out_pixel_mm=float(args.pixel_mm),
            device=str(args.device),
            batch_views=16,
            angle_step_rad=angle_step,
        )
        vol[zi] = img

    np.save(out_dir / "vol.npy", vol.astype(np.float32))
    zc, yc, xc = nz // 2, nx // 2, nx // 2
    save_png(vol[zc], out_dir / "slice_zc.png")
    save_png(vol[:, yc, :], out_dir / "slice_yc.png")
    save_png(vol[:, :, xc], out_dir / "slice_xc.png")

    rep = {
        "i0": float(i0),
        "view_mode": str(args.view_mode),
        "view_count": int(sel.indices.size),
        "target_mode": str(args.target_mode),
        "nx": nx,
        "nz": nz,
        "pixel_mm": float(args.pixel_mm),
        "det_cols": det_cols,
        "det_rows": det_rows,
        "det_offset_x_px": float(det_offset_x_px),
        "det_offset_y_px": float(det_offset_y_px),
    }
    (out_dir / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {out_dir/'vol.npy'} and report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

