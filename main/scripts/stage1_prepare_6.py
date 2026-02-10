from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from ct3dr.geom import load_recon_params, load_scan_params, to_jsonable
from ct3dr.io import dr_shape_from_file, list_dr_files, read_dr_memmap, save_preview_png_u16
from ct3dr.roi import crop_dr_dataset_to_npy, estimate_object_bbox


def main() -> int:
    ap = argparse.ArgumentParser(description="Stage1: parse params + crop 6号数据 DR to a compact .npy")
    ap.add_argument(
        "--data_dir",
        type=Path,
        default=Path("main/data/6-ScanTask-仅4个微粒"),
        help="Folder containing *.DR and ScanTask.xml",
    )
    ap.add_argument("--out_dir", type=Path, default=Path("main/processed/6"), help="Output folder")
    ap.add_argument("--downsample", type=int, default=1, help="Downsample factor after crop")
    ap.add_argument("--thr", type=float, default=0.75, help="Threshold for bbox estimation (0..1)")
    ap.add_argument("--pad", type=int, default=64, help="Padding (pixels) for crop box")
    args = ap.parse_args()

    data_dir: Path = args.data_dir
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    scan_xml = data_dir / "ScanTask.xml"
    recon_txt = data_dir / "Volume" / "ReconPara.txt"
    if not scan_xml.exists():
        raise FileNotFoundError(f"Missing {scan_xml}")
    if not recon_txt.exists():
        raise FileNotFoundError(f"Missing {recon_txt}")

    recon = load_recon_params(recon_txt)
    scan = load_scan_params(scan_xml, recon_txt=recon_txt)

    dr_files = list_dr_files(data_dir)
    if not dr_files:
        raise FileNotFoundError(f"No DR files found under {data_dir}")
    shape_hw = dr_shape_from_file(dr_files[0])

    crop = estimate_object_bbox(
        dr_folder=data_dir,
        shape_hw=shape_hw,
        downsample=8,
        thr=float(args.thr),
        pad_px=int(args.pad),
    )

    # Save a preview image pre-crop.
    mm0 = read_dr_memmap(dr_files[0], shape_hw)
    save_preview_png_u16(mm0, out_dir / "preview_0001_full.png", downsample=8)
    save_preview_png_u16(mm0[crop.r0 : crop.r1, crop.c0 : crop.c1], out_dir / "preview_0001_crop.png", downsample=8)

    out_npy = crop_dr_dataset_to_npy(
        dr_folder=data_dir,
        out_dir=out_dir,
        shape_hw=shape_hw,
        crop=crop,
        downsample=int(args.downsample),
    )

    meta = {"scan": to_jsonable(scan), "recon": to_jsonable(recon)}
    (out_dir / "scan_recon_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"Wrote {out_npy}")
    print(f"Crop box: r[{crop.r0},{crop.r1}) c[{crop.c0},{crop.c1}) downsample={args.downsample}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
