from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np

from ct3dr.particles import extract_particles


def main() -> int:
    ap = argparse.ArgumentParser(description="Extract particle stats from a 3D volume (.npy).")
    ap.add_argument("volume", type=Path, help="3D volume saved as .npy (Z,Y,X)")
    ap.add_argument("--voxel_mm", type=float, default=0.1)
    ap.add_argument("--thr", type=float, default=None, help="Threshold for segmentation; default=Otsu")
    ap.add_argument("--min_vox", type=int, default=200)
    ap.add_argument("--max_vox", type=int, default=200000)
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--out_json", type=Path, default=None)
    ap.add_argument("--out_csv", type=Path, default=None)
    args = ap.parse_args()

    vol = np.load(args.volume)
    parts = extract_particles(
        vol,
        voxel_size_mm=float(args.voxel_mm),
        threshold=args.thr,
        min_voxels=int(args.min_vox),
        max_voxels=int(args.max_vox),
        keep_top_k=int(args.topk) if args.topk is not None else None,
    )
    payload = {"count": len(parts), "particles": [p.to_dict() for p in parts]}
    print(json.dumps(payload, ensure_ascii=False, indent=2))

    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.out_csv is not None:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.out_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(
                f,
                fieldnames=["label", "voxel_count", "center_z", "center_y", "center_x", "diameter_mm"],
            )
            w.writeheader()
            for p in parts:
                w.writerow(
                    {
                        "label": p.label,
                        "voxel_count": p.voxel_count,
                        "center_z": p.center_zyx[0],
                        "center_y": p.center_zyx[1],
                        "center_x": p.center_zyx[2],
                        "diameter_mm": p.diameter_mm,
                    }
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
