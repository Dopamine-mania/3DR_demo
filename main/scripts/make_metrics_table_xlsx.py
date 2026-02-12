from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np

from ct3dr.metrics import compute_all
from ct3dr.xlsx_min import write_xlsx_simple


def norm01(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    lo, hi = np.percentile(x, [1.0, 99.0])
    if hi <= lo:
        hi = lo + 1.0
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def metrics_pair(a_path: Path, b_path: Path, dice_thr: float = 0.5) -> dict[str, float] | None:
    if not a_path.exists() or not b_path.exists():
        return None
    a = np.load(a_path, mmap_mode="r")
    b = np.load(b_path, mmap_mode="r")
    if a.shape != b.shape:
        # Center-crop both to the common minimum shape.
        target = tuple(int(min(sa, sb)) for sa, sb in zip(a.shape, b.shape))

        def crop_center(x: np.ndarray, shape):
            slices = []
            for dim, t in zip(x.shape, shape):
                start = max(0, (dim - t) // 2)
                slices.append(slice(start, start + t))
            return np.asarray(x[tuple(slices)])

        a = crop_center(a, target)
        b = crop_center(b, target)
    a_n = norm01(np.asarray(a))
    b_n = norm01(np.asarray(b))
    res = compute_all(a_n, b_n, data_range=1.0, dice_threshold=float(dice_thr))
    return res.to_dict()


def main() -> int:
    ap = argparse.ArgumentParser(description="Create metrics summary Excel for Stage2/Stage3.")
    ap.add_argument("--out_xlsx", type=Path, default=Path("main/output/metrics_summary.xlsx"))
    ap.add_argument("--dice_thr", type=float, default=0.5)
    ap.add_argument("--ref_full", type=Path, default=Path("main/output/stage2_fdk_full/vol.npy"))
    ap.add_argument("--stage2_fdk_sparse", type=Path, default=Path("main/output/stage2_fdk_sparse90/vol.npy"))
    ap.add_argument("--stage2_naf_sparse", type=Path, default=Path("main/output/stage2_nafcs_sparse90/vol.npy"))
    ap.add_argument("--stage3_fdk_limited", type=Path, default=Path("main/output/stage3_fdk_limited120/vol.npy"))
    ap.add_argument("--stage3_naf_limited", type=Path, default=Path("main/output/stage3_nafattn_limited120/vol.npy"))
    args = ap.parse_args()

    headers = ["Stage", "Case", "MSE", "PSNR", "SSIM", "DICE"]
    rows: list[list[object]] = []

    def add(stage: str, case: str, a: Path, b: Path):
        m = metrics_pair(a, b, dice_thr=float(args.dice_thr))
        if m is None:
            rows.append([stage, case, None, None, None, None])
            return
        rows.append([stage, case, m["mse"], m["psnr"], m["ssim"], m.get("dice")])

    # Stage2 (Sparse 90)
    add("Stage2", "FDK sparse90 vs FDK full", args.stage2_fdk_sparse, args.ref_full)
    add("Stage2", "NAF+CS sparse90 vs FDK full", args.stage2_naf_sparse, args.ref_full)

    # Stage3 (Limited 120)
    add("Stage3", "FDK limited120 vs FDK full", args.stage3_fdk_limited, args.ref_full)
    add("Stage3", "NAF-Attention limited120 vs FDK full", args.stage3_naf_limited, args.ref_full)

    write_xlsx_simple(args.out_xlsx, sheet_name="Metrics", headers=headers, rows=rows)
    print(f"Wrote {args.out_xlsx}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
