from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np

from ct3dr.metrics import compute_all
from ct3dr.io import read_raw_f32


def load_array(path: Path, raw_shape: tuple[int, ...] | None) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        return np.load(path)
    if path.suffix.lower() == ".raw":
        if raw_shape is None:
            raise ValueError("--raw_shape required for .raw input")
        return read_raw_f32(path, shape=raw_shape)
    raise ValueError(f"Unsupported file: {path}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Evaluate metrics between two arrays (2D or 3D).")
    ap.add_argument("a", type=Path)
    ap.add_argument("b", type=Path)
    ap.add_argument("--raw_shape", type=str, default=None, help="For .raw: e.g. 3820,813,813 or 813,813")
    ap.add_argument("--data_range", type=float, default=1.0)
    ap.add_argument("--dice_thr", type=float, default=None)
    ap.add_argument(
        "--normalize",
        choices=["none", "p1p99"],
        default="none",
        help="Optional normalization before metrics: p1p99 maps percentiles [1,99] -> [0,1].",
    )
    ap.add_argument("--out_json", type=Path, default=None)
    args = ap.parse_args()

    raw_shape = None
    if args.raw_shape is not None:
        raw_shape = tuple(int(x) for x in args.raw_shape.split(","))

    a = load_array(args.a, raw_shape)
    b = load_array(args.b, raw_shape)

    if args.normalize == "p1p99":
        def norm01(x):
            x = x.astype(np.float32)
            lo, hi = np.percentile(x, [1.0, 99.0])
            if hi <= lo:
                hi = lo + 1.0
            return np.clip((x - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)

        a = norm01(a)
        b = norm01(b)

    res = compute_all(a, b, data_range=float(args.data_range), dice_threshold=args.dice_thr)
    d = res.to_dict()
    print(json.dumps(d, ensure_ascii=False, indent=2))
    if args.out_json is not None:
        args.out_json.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
