from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np

from ct3dr.astra_fdk import u16_to_line_integral
from ct3dr.naf3d import TrainNAF3DConfig, render_volume, train_naf3d
from ct3dr.view_sampling import select_views


def save_png(img: np.ndarray, out_png: Path, *, p_lo: float = 0.5, p_hi: float = 99.5) -> None:
    from PIL import Image

    out_png.parent.mkdir(parents=True, exist_ok=True)
    x = img.astype(np.float32)
    lo, hi = np.percentile(x, [float(p_lo), float(p_hi)])
    if hi <= lo:
        hi = lo + 1.0
    u8 = np.clip((x - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(u8).save(out_png)


def main() -> int:
    ap = argparse.ArgumentParser(description="Train 3D NAF + TV(CS) on cropped projections.npy")
    ap.add_argument("--processed_dir", type=Path, default=Path("main/processed/6_ds4_run2"))
    ap.add_argument("--out_dir", type=Path, default=Path("main/output/naf3d_cs_sparse"))

    ap.add_argument("--view_mode", choices=["full", "sparse", "limited"], default="sparse")
    ap.add_argument("--sparse_keep", type=int, default=90)
    ap.add_argument("--limited_center_deg", type=float, default=180.0)
    ap.add_argument("--limited_span_deg", type=float, default=120.0)

    ap.add_argument("--half_mm", type=float, default=24.0, help="Half size of training cube (mm)")
    ap.add_argument("--n_samples", type=int, default=64)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--batch_views", type=int, default=2)
    ap.add_argument("--batch_pixels", type=int, default=2048)
    ap.add_argument("--tv_weight", type=float, default=1e-4)
    ap.add_argument("--tv_points", type=int, default=2048)
    ap.add_argument("--hot_q", type=float, default=0.98, help="Quantile for hot-pixel sampling")
    ap.add_argument("--hot_frac", type=float, default=0.8, help="Fraction of hot pixels in a batch")
    ap.add_argument(
        "--target_mode",
        choices=["raw", "residual"],
        default="residual",
        help="raw: use -log(I/I0); residual: subtract per-view median and clamp >=0 (focus particles)",
    )

    ap.add_argument("--render_n", type=int, default=160, help="Render volume size (cube)")
    ap.add_argument("--render_chunk", type=int, default=131072)
    ap.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    processed_dir: Path = args.processed_dir
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    proj_path = processed_dir / "projections.npy"
    meta_proj_path = processed_dir / "projections_meta.json"
    meta_scan_path = processed_dir / "scan_recon_meta.json"
    if not proj_path.exists():
        raise FileNotFoundError(f"Missing {proj_path}")
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

    # Compute effective detector offsets for the cropped & downsampled detector.
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

    proj_u16 = np.load(proj_path, mmap_mode="r")  # (V,H,W)
    proj_u16 = np.asarray(proj_u16[sel.indices, :, :], dtype=np.uint16)

    # Estimate I0 robustly from selected views.
    i0 = float(np.quantile(proj_u16.astype(np.float32), 0.999))
    proj_line = u16_to_line_integral(proj_u16, i0=i0)  # float32
    if args.target_mode == "residual":
        # Remove dominant smooth/background component (e.g., container) and keep positive residuals.
        med = np.median(proj_line.reshape(proj_line.shape[0], -1), axis=1).astype(np.float32)  # (V,)
        proj_line = proj_line - med[:, None, None]
        proj_line = np.clip(proj_line, 0.0, None)

    half = float(args.half_mm)
    cfg = TrainNAF3DConfig(
        half_size_mm=(half, half, half),
        n_samples=int(args.n_samples),
        lr=float(args.lr),
        steps=int(args.steps),
        batch_views=int(args.batch_views),
        batch_pixels=int(args.batch_pixels),
        tv_weight=float(args.tv_weight),
        tv_points=int(args.tv_points),
        hot_quantile=float(args.hot_q),
        hot_fraction=float(args.hot_frac),
        device=str(args.device),
    )

    t0 = time.time()
    summary, mlp, pe, _ = train_naf3d(
        proj_line,
        sel.angles_rad,
        dso_mm=float(scan["dso_mm"]),
        dsd_mm=float(scan["dsd_mm"]),
        det_spacing_x_mm=float(det_spacing_x),
        det_spacing_y_mm=float(det_spacing_y),
        det_center_offset_x_px=float(det_offset_x_px),
        det_center_offset_y_px=float(det_offset_y_px),
        cfg=cfg,
        seed=int(args.seed),
    )
    summary["elapsed_sec"] = float(time.time() - t0)
    summary["i0"] = float(i0)
    summary["view_mode"] = str(args.view_mode)
    summary["view_count"] = int(sel.indices.size)
    summary["target_mode"] = str(args.target_mode)
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    # Render a cubic volume for visualization.
    n = int(args.render_n)
    vol = render_volume(
        mlp,
        pe,
        shape_zyx=(n, n, n),
        half_size_mm=(half, half, half),
        device=str(args.device),
        chunk=int(args.render_chunk),
    )
    np.save(out_dir / "vol.npy", vol.astype(np.float32))

    zc, yc, xc = n // 2, n // 2, n // 2
    # Use a wider dynamic range window to make tiny particles visible early.
    save_png(vol[zc], out_dir / "slice_zc.png", p_lo=0.1, p_hi=99.9)
    save_png(vol[:, yc, :], out_dir / "slice_yc.png", p_lo=0.1, p_hi=99.9)
    save_png(vol[:, :, xc], out_dir / "slice_xc.png", p_lo=0.1, p_hi=99.9)

    print(f"Wrote {out_dir/'vol.npy'} and summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
