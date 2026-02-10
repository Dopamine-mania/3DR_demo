from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np

from ct3dr.fanbeam_fbp import FanbeamGeom, fbp_fanbeam_flat_detector
from ct3dr.geom import load_recon_params, load_scan_params
from ct3dr.io import read_raw_f32, read_sinogram_u16
from ct3dr.metrics import compute_all


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
    ap = argparse.ArgumentParser(description="Run 2D FDK/FBP baseline on Sino.sin and save slice.")
    ap.add_argument(
        "--data_dir",
        type=Path,
        default=Path("main/data/6-ScanTask-仅4个微粒"),
        help="Folder containing sin/Sino.sin and params",
    )
    ap.add_argument("--out_dir", type=Path, default=Path("main/output/fdk_2d"), help="Output folder")
    ap.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    ap.add_argument("--det_offset_px", type=float, default=None, help="Override detector center offset (px)")
    args = ap.parse_args()

    data_dir: Path = args.data_dir
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    scan_xml = data_dir / "ScanTask.xml"
    recon_txt = data_dir / "Volume" / "ReconPara.txt"
    sino_path = data_dir / "sin" / "Sino.sin"
    slice_raw = data_dir / "Volume" / "0位置切片.raw"

    recon = load_recon_params(recon_txt)
    scan = load_scan_params(scan_xml, recon_txt=recon_txt)

    sino = read_sinogram_u16(sino_path, shape=(scan.num_projections, scan.detector_cols))

    det_spacing = recon.det_h_spacing_mm if recon.det_h_spacing_mm > 0 else scan.pixel_size_mm
    det_offset = scan.det_h_center_offset_px
    if args.det_offset_px is not None:
        det_offset = float(args.det_offset_px)

    geom = FanbeamGeom(
        num_views=scan.num_projections,
        num_det=scan.detector_cols,
        det_spacing_mm=float(det_spacing),
        dso_mm=float(scan.dso_mm),
        dsd_mm=float(scan.dsd_mm),
        det_center_offset_px=float(det_offset),
        rotation_ccw=bool(scan.rotation_ccw),
        start_angle_rad=float(np.deg2rad(scan.start_angle_deg)),
    )

    img = fbp_fanbeam_flat_detector(
        sino_u16=sino,
        geom=geom,
        out_size=int(recon.recon_nx),
        out_pixel_mm=float(recon.voxel_size_mm),
        device=args.device,
        batch_views=16,
    )

    np.save(out_dir / "fbp_slice.npy", img)
    save_png(img, out_dir / "fbp_slice.png")

    report = {
        "geom": geom.__dict__,
        "recon_nx": int(recon.recon_nx),
        "voxel_size_mm": float(recon.voxel_size_mm),
    }

    if slice_raw.exists():
        gt = read_raw_f32(slice_raw, shape=(recon.recon_ny, recon.recon_nx))
        # Normalize both to [0,1] for metric convention.
        def norm01(x: np.ndarray) -> np.ndarray:
            lo, hi = np.percentile(x.astype(np.float32), [1, 99])
            if hi <= lo:
                hi = lo + 1.0
            return np.clip((x - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)

        m = compute_all(norm01(img), norm01(gt), data_range=1.0, dice_threshold=0.5)
        report["metrics_vs_volume0slice"] = m.to_dict()
        (out_dir / "gt_slice.npy").write_bytes(slice_raw.read_bytes())
        save_png(gt, out_dir / "gt_slice.png")

    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {out_dir/'fbp_slice.npy'} and report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
