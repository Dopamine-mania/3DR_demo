from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np

from ct3dr.geom import load_recon_params, load_scan_params
from ct3dr.io import read_sinogram_u16
from ct3dr.fanbeam_fbp import FanbeamGeom
from ct3dr.naf2d import NAF2DConfig, train_naf2d


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
    ap = argparse.ArgumentParser(description="Prototype NAF-2D training on Sino.sin (outputs a 2D slice).")
    ap.add_argument(
        "--data_dir",
        type=Path,
        default=Path("main/data/6-ScanTask-仅4个微粒"),
        help="Folder containing sin/Sino.sin and params",
    )
    ap.add_argument("--out_dir", type=Path, default=Path("main/output/naf2d"), help="Output folder")
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    ap.add_argument("--fov_mm", type=float, default=81.3, help="FOV size in mm for normalization")
    args = ap.parse_args()

    data_dir: Path = args.data_dir
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    scan_xml = data_dir / "ScanTask.xml"
    recon_txt = data_dir / "Volume" / "ReconPara.txt"
    sino_path = data_dir / "sin" / "Sino.sin"

    recon = load_recon_params(recon_txt)
    scan = load_scan_params(scan_xml, recon_txt=recon_txt)
    sino = read_sinogram_u16(sino_path, shape=(scan.num_projections, scan.detector_cols))

    det_spacing = recon.det_h_spacing_mm if recon.det_h_spacing_mm > 0 else scan.pixel_size_mm
    geom = FanbeamGeom(
        num_views=scan.num_projections,
        num_det=scan.detector_cols,
        det_spacing_mm=float(det_spacing),
        dso_mm=float(scan.dso_mm),
        dsd_mm=float(scan.dsd_mm),
        det_center_offset_px=float(scan.det_h_center_offset_px),
        rotation_ccw=bool(scan.rotation_ccw),
        start_angle_rad=float(np.deg2rad(scan.start_angle_deg)),
    )

    cfg = NAF2DConfig(
        fov_mm=float(args.fov_mm),
        steps=int(args.steps),
        device=args.device,
    )
    img, summary = train_naf2d(sino, geom=geom, cfg=cfg)
    np.save(out_dir / "naf2d_slice.npy", img)
    save_png(img, out_dir / "naf2d_slice.png")
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {out_dir/'naf2d_slice.npy'} and summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
