from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _to_uint8(img: np.ndarray, p_lo: float = 1.0, p_hi: float = 99.9, gamma: float = 0.7) -> np.ndarray:
    x = img.astype(np.float32)
    lo, hi = np.percentile(x, [float(p_lo), float(p_hi)])
    if hi <= lo:
        hi = lo + 1.0
    y = np.clip((x - lo) / (hi - lo), 0.0, 1.0)
    y = y ** float(gamma)
    return (y * 255.0).astype(np.uint8)


def mip_xy(vol: np.ndarray, crop: int = 140) -> np.ndarray:
    v = np.clip(vol, 0, None)
    img = np.max(v, axis=0)
    ny, nx = img.shape
    cy, cx = ny // 2, nx // 2
    r = crop // 2
    img = img[cy - r : cy + r, cx - r : cx + r]
    return img


def best_slice_z(vol: np.ndarray, crop: int = 140) -> tuple[int, np.ndarray]:
    nz, ny, nx = vol.shape
    cy, cx = ny // 2, nx // 2
    r = crop // 2
    y0, y1 = cy - r, cy + r
    x0, x1 = cx - r, cx + r
    scores = []
    for z in range(nz):
        s = vol[z, y0:y1, x0:x1]
        scores.append(float(np.percentile(s, 99.9)))
    z = int(np.argmax(scores))
    return z, vol[z, y0:y1, x0:x1]


def _load_vol(path: Path) -> np.ndarray:
    return np.load(path, mmap_mode="r")


def _label(im: Image.Image, text: str) -> Image.Image:
    im = im.convert("RGB")
    draw = ImageDraw.Draw(im)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 18)
    except Exception:
        font = ImageFont.load_default()
    pad = 8
    tw, th = draw.textbbox((0, 0), text, font=font)[2:]
    draw.rectangle([0, 0, tw + 2 * pad, th + 2 * pad], fill=(0, 0, 0))
    draw.text((pad, pad), text, fill=(255, 255, 255), font=font)
    return im


def main() -> int:
    ap = argparse.ArgumentParser(description="Make Stage2 battle figure: FDK(full) vs FDK(sparse) vs NAF+CS(sparse)")
    ap.add_argument("--fdk_full", type=Path, default=Path("main/output/stage2_fdk_full/vol.npy"))
    ap.add_argument("--fdk_sparse", type=Path, default=Path("main/output/stage2_fdk_sparse90/vol.npy"))
    ap.add_argument("--nafcs_sparse", type=Path, default=Path("main/output/stage2_nafcs_sparse90/vol.npy"))
    ap.add_argument("--out_png", type=Path, default=Path("main/output/stage2_battle.png"))
    ap.add_argument("--crop", type=int, default=140)
    ap.add_argument("--mode", choices=["slice", "mip"], default="mip")
    args = ap.parse_args()

    vols = [
        ("FDK 全采样", _load_vol(args.fdk_full)),
        ("FDK 1/10采样", _load_vol(args.fdk_sparse)),
        ("NAF+CS 1/10采样", _load_vol(args.nafcs_sparse)),
    ]

    imgs = []
    for title, v in vols:
        if args.mode == "mip":
            img = mip_xy(v, crop=int(args.crop))
        else:
            z, img = best_slice_z(v, crop=int(args.crop))
            title = f"{title} (z={z})"
        u8 = _to_uint8(img, p_lo=1.0, p_hi=99.9, gamma=0.7)
        im = Image.fromarray(u8)
        im = _label(im, title)
        imgs.append(im)

    w, h = imgs[0].size
    out = Image.new("RGB", (w * 3, h), (0, 0, 0))
    for i, im in enumerate(imgs):
        out.paste(im, (i * w, 0))
    args.out_png.parent.mkdir(parents=True, exist_ok=True)
    out.save(args.out_png)
    print(f"Wrote {args.out_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

