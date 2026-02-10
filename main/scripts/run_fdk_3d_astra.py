from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np

from ct3dr.astra_fdk import ConeVecGeom, run_fdk_cuda, u16_to_line_integral


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
    ap = argparse.ArgumentParser(description="Run 3D FDK (ASTRA CUDA) from cropped projections.npy")
    ap.add_argument(
        "--processed_dir",
        type=Path,
        default=Path("main/processed/6_ds4_run2"),
        help="Output folder produced by stage1_prepare_6.py",
    )
    ap.add_argument("--out_dir", type=Path, default=Path("main/output/fdk_3d"), help="Output folder")
    ap.add_argument("--voxel_mm", type=float, default=0.2, help="Voxel size in mm")
    ap.add_argument("--nx", type=int, default=192)
    ap.add_argument("--ny", type=int, default=192)
    ap.add_argument("--nz", type=int, default=192)
    ap.add_argument("--i0", type=float, default=None, help="Intensity I0 for -log(I/I0); default=auto")
    args = ap.parse_args()

    processed_dir: Path = args.processed_dir
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    proj_path = processed_dir / "projections.npy"
    meta_proj_path = processed_dir / "projections_meta.json"
    meta_scan_path = processed_dir / "scan_recon_meta.json"
    if not proj_path.exists():
        raise FileNotFoundError(f"Missing {proj_path}")
    if not meta_proj_path.exists():
        raise FileNotFoundError(f"Missing {meta_proj_path}")
    if not meta_scan_path.exists():
        raise FileNotFoundError(f"Missing {meta_scan_path}")

    proj_meta = json.loads(meta_proj_path.read_text(encoding="utf-8"))
    scan_meta = json.loads(meta_scan_path.read_text(encoding="utf-8"))

    scan = scan_meta["scan"]
    recon = scan_meta["recon"]

    crop = proj_meta["crop"]
    ds = int(proj_meta["downsample"])
    det_rows = int(proj_meta["output_shape"][1])
    det_cols = int(proj_meta["output_shape"][2])
    num_projections = int(proj_meta["num_projections"])

    # Effective center offsets in the cropped detector coordinates.
    full_cols = int(scan["detector_cols"])
    full_rows = int(scan["detector_rows"])
    full_center_u = (full_cols - 1) / 2.0 + float(scan["det_h_center_offset_px"])
    full_center_v = (full_rows - 1) / 2.0 + float(scan["det_v_center_offset_px"])

    center_u_crop = (full_center_u - float(crop["c0"])) / float(ds)
    center_v_crop = (full_center_v - float(crop["r0"])) / float(ds)
    det_offset_x_px = center_u_crop - (det_cols - 1) / 2.0
    det_offset_y_px = center_v_crop - (det_rows - 1) / 2.0

    det_spacing_x = float(recon["det_h_spacing_mm"]) * float(ds)
    det_spacing_y = float(recon["det_v_spacing_mm"]) * float(ds)

    vec_geom = ConeVecGeom(
        det_rows=det_rows,
        det_cols=det_cols,
        det_spacing_y_mm=det_spacing_y,
        det_spacing_x_mm=det_spacing_x,
        dso_mm=float(scan["dso_mm"]),
        dsd_mm=float(scan["dsd_mm"]),
        det_center_offset_x_px=float(det_offset_x_px),
        det_center_offset_y_px=float(det_offset_y_px),
        rotation_ccw=bool(scan["rotation_ccw"]),
        start_angle_deg=float(scan["start_angle_deg"]),
        num_projections=num_projections,
    )

    # Load uint16 projections and convert to line integrals.
    proj_u16 = np.load(proj_path, mmap_mode="r")  # (V,H,W)
    if proj_u16.shape != (num_projections, det_rows, det_cols):
        raise ValueError(f"projections.npy shape mismatch: {proj_u16.shape}")

    # Estimate I0 from a subset for speed.
    if args.i0 is None:
        sample = np.asarray(proj_u16[:: max(1, num_projections // 30), :, :], dtype=np.float32)
        i0 = float(np.quantile(sample, 0.999))
    else:
        i0 = float(args.i0)

    proj_line = u16_to_line_integral(np.asarray(proj_u16, dtype=np.uint16), i0=i0)

    vol, rep = run_fdk_cuda(
        projections_line_integral=proj_line,
        vec_geom=vec_geom,
        vol_shape_zyx=(int(args.nz), int(args.ny), int(args.nx)),
        voxel_mm=float(args.voxel_mm),
    )

    np.save(out_dir / "vol.npy", vol)

    zc, yc, xc = vol.shape[0] // 2, vol.shape[1] // 2, vol.shape[2] // 2
    save_png(vol[zc], out_dir / "slice_zc.png")
    save_png(vol[:, yc, :], out_dir / "slice_yc.png")
    save_png(vol[:, :, xc], out_dir / "slice_xc.png")

    report = {
        "processed_dir": str(processed_dir),
        "i0": i0,
        "vec_geom": vec_geom.__dict__,
        "astra_report": rep,
        "out_slices": ["slice_zc.png", "slice_yc.png", "slice_xc.png"],
    }
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {out_dir/'vol.npy'} and report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

