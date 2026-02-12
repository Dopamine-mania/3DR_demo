from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np

from ct3dr.astra_fdk import u16_to_line_integral
from ct3dr.naf3d import MLP3D, PosEnc3D, build_cone_beam_rays, forward_project_grid, render_volume, tv_loss_grad
from ct3dr.torch_losses import dice_loss, ssim_loss
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
    ap.add_argument("--batch_views", type=int, default=1)
    ap.add_argument("--patch", type=int, default=96, help="Detector patch size for composite loss (square)")
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

    ap.add_argument("--posenc_freqs", type=int, default=6)
    ap.add_argument("--mlp_width", type=int, default=128)
    ap.add_argument("--mlp_depth", type=int, default=6)

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

    # Build "hot" coordinates per view for patch sampling.
    V, H, W = proj_line.shape
    hot_thr = float(np.quantile(proj_line.reshape(V, -1), float(args.hot_q)))
    hot_coords: list[np.ndarray] = []
    for v in range(V):
        coords = np.argwhere(proj_line[v] > hot_thr)  # (N,2) row,col
        hot_coords.append(coords.astype(np.int64, copy=False))

    import torch

    torch.manual_seed(int(args.seed))
    dev = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")

    rays = build_cone_beam_rays(
        sel.angles_rad,
        dso_mm=float(scan["dso_mm"]),
        dsd_mm=float(scan["dsd_mm"]),
        det_rows=int(det_rows),
        det_cols=int(det_cols),
        det_spacing_x_mm=float(det_spacing_x),
        det_spacing_y_mm=float(det_spacing_y),
        det_center_offset_x_px=float(det_offset_x_px),
        det_center_offset_y_px=float(det_offset_y_px),
        device=dev,
    )

    pe = PosEnc3D(int(args.posenc_freqs)).to(dev)
    in_dim = 3 + 6 * int(args.posenc_freqs)
    mlp = MLP3D(in_dim=in_dim, width=int(args.mlp_width), depth=int(args.mlp_depth)).to(dev)
    opt = torch.optim.Adam(mlp.parameters(), lr=float(args.lr))

    proj_t = torch.as_tensor(proj_line, device=dev, dtype=torch.float32)  # (V,H,W)

    patch = int(args.patch)
    if patch <= 8 or patch > min(H, W):
        raise ValueError(f"--patch must be in [9, {min(H,W)}], got {patch}")

    half = float(args.half_mm)
    half_box = (half, half, half)

    log: list[dict[str, float]] = []
    t0 = time.time()
    for step in range(int(args.steps)):
        view_ids = torch.randint(0, V, (int(args.batch_views),), device=dev)

        total = torch.tensor(0.0, device=dev)
        mse_v = torch.tensor(0.0, device=dev)
        l1_v = torch.tensor(0.0, device=dev)
        ssim_v = torch.tensor(0.0, device=dev)
        dice_v = torch.tensor(0.0, device=dev)

        for vv in view_ids.tolist():
            hc = hot_coords[int(vv)]
            if hc.size > 0:
                rr, cc = hc[np.random.randint(0, hc.shape[0])]
            else:
                rr = np.random.randint(0, H)
                cc = np.random.randint(0, W)
            r0 = int(np.clip(rr - patch // 2, 0, H - patch))
            c0 = int(np.clip(cc - patch // 2, 0, W - patch))

            row_ids = torch.arange(r0, r0 + patch, device=dev, dtype=torch.int64)
            col_ids = torch.arange(c0, c0 + patch, device=dev, dtype=torch.int64)

            pred = forward_project_grid(
                mlp,
                pe,
                rays,
                view_ids=torch.tensor([vv], device=dev, dtype=torch.int64),
                row_ids=row_ids,
                col_ids=col_ids,
                half_size_mm=half_box,
                n_samples=int(args.n_samples),
                normalize_coords=True,
            )[0]  # (patch,patch)
            tgt = proj_t[vv, r0 : r0 + patch, c0 : c0 + patch]

            # Normalize to [0,1] for composite loss stability.
            scale = torch.quantile(tgt.flatten(), 0.99).clamp(min=1e-6)
            pred_n = (pred / scale).clamp(0.0, 1.0)
            tgt_n = (tgt / scale).clamp(0.0, 1.0)

            diff = pred_n - tgt_n
            mse = torch.mean(diff * diff)
            l1 = torch.mean(torch.abs(diff))
            ssim_l = ssim_loss(pred_n[None, None, :, :], tgt_n[None, None, :, :], data_range=1.0)
            dice_l = dice_loss(pred_n[None, None, :, :], tgt_n[None, None, :, :])
            loss = mse + l1 + ssim_l + dice_l

            mse_v = mse_v + mse
            l1_v = l1_v + l1
            ssim_v = ssim_v + ssim_l
            dice_v = dice_v + dice_l
            total = total + loss

        # Average over views.
        bv = max(1, int(args.batch_views))
        total = total / float(bv)
        mse_v = mse_v / float(bv)
        l1_v = l1_v / float(bv)
        ssim_v = ssim_v / float(bv)
        dice_v = dice_v / float(bv)

        tv = torch.tensor(0.0, device=dev)
        if float(args.tv_weight) > 0:
            pts = (torch.rand(int(args.tv_points), 3, device=dev) * 2.0 - 1.0) * torch.tensor(
                list(half_box), device=dev, dtype=torch.float32
            )
            tv = tv_loss_grad(mlp, pe, pts, half_size_mm=half_box)
            total = total + float(args.tv_weight) * tv

        opt.zero_grad(set_to_none=True)
        total.backward()
        opt.step()

        if step % 50 == 0 or step == int(args.steps) - 1:
            log.append(
                {
                    "step": float(step),
                    "mse": float(mse_v.item()),
                    "l1": float(l1_v.item()),
                    "ssim_loss": float(ssim_v.item()),
                    "dice_loss": float(dice_v.item()),
                    "tv": float(tv.item()),
                    "total": float(total.item()),
                }
            )

    summary = {
        "elapsed_sec": float(time.time() - t0),
        "i0": float(i0),
        "view_mode": str(args.view_mode),
        "view_count": int(sel.indices.size),
        "target_mode": str(args.target_mode),
        "half_mm": float(args.half_mm),
        "n_samples": int(args.n_samples),
        "steps": int(args.steps),
        "lr": float(args.lr),
        "batch_views": int(args.batch_views),
        "patch": int(args.patch),
        "tv_weight": float(args.tv_weight),
        "tv_points": int(args.tv_points),
        "posenc_freqs": int(args.posenc_freqs),
        "mlp_width": int(args.mlp_width),
        "mlp_depth": int(args.mlp_depth),
        "loss_log": log,
    }
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
