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


def mip_ortho(vol: np.ndarray, crop: int = 140) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    v = np.clip(vol, 0, None).astype(np.float32)
    xy = np.max(v, axis=0)  # (Y,X)
    xz = np.max(v, axis=1)  # (Z,X)
    yz = np.max(v, axis=2)  # (Z,Y)

    def crop_center(img: np.ndarray) -> np.ndarray:
        h, w = img.shape
        cy, cx = h // 2, w // 2
        r = crop // 2
        y0, y1 = max(0, cy - r), min(h, cy + r)
        x0, x1 = max(0, cx - r), min(w, cx + r)
        return img[y0:y1, x0:x1]

    return crop_center(xy), crop_center(xz), crop_center(yz)


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


def _panel(vol: np.ndarray, title: str, crop: int) -> Image.Image:
    xy, xz, yz = mip_ortho(vol, crop=crop)
    tiles = []
    for sub in (xy, xz, yz):
        u8 = _to_uint8(sub, p_lo=1.0, p_hi=99.9, gamma=0.7)
        tiles.append(Image.fromarray(u8))
    w = max(t.size[0] for t in tiles)
    h = max(t.size[1] for t in tiles)
    tiles = [t.resize((w, h), resample=Image.BILINEAR) for t in tiles]
    panel = Image.new("L", (w, h * 3), 0)
    for i, t in enumerate(tiles):
        panel.paste(t, (0, i * h))
    return _label(panel, title)


def main() -> int:
    ap = argparse.ArgumentParser(description="Stage3 battle: FDK full vs FDK limited120 vs NAF-Attention limited120")
    ap.add_argument("--fdk_full", type=Path, default=Path("main/output/stage2_fdk_full/vol.npy"))
    ap.add_argument("--fdk_limited", type=Path, default=Path("main/output/stage3_fdk_limited120/vol.npy"))
    ap.add_argument("--naf_limited", type=Path, default=Path("main/output/stage3_nafattn_limited120/vol.npy"))
    ap.add_argument("--out_png", type=Path, default=Path("main/output/stage3_limited120_battle.png"))
    ap.add_argument("--crop", type=int, default=140)
    args = ap.parse_args()

    vols = [
        ("FDK 全视角", _load_vol(args.fdk_full)),
        ("FDK 120°有限角", _load_vol(args.fdk_limited)),
        ("NAF-Attention 120°有限角", _load_vol(args.naf_limited)),
    ]

    panels = [_panel(v, title, crop=int(args.crop)) for title, v in vols]
    w, h = panels[0].size
    out = Image.new("RGB", (w * 3, h), (0, 0, 0))
    for i, p in enumerate(panels):
        out.paste(p, (i * w, 0))
    args.out_png.parent.mkdir(parents=True, exist_ok=True)
    out.save(args.out_png)
    print(f"Wrote {args.out_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

