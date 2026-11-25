"""Cut out photographed pointing hands for the Kang-style gesture stimuli (option A).

Input photos (one hand per photo): the forearm enters from the LEFT edge of the frame, arm roughly
horizontal, index finger extended and pointing RIGHT, on a plain background that contrasts with
skin and sleeve (white wall / paper). Nothing else touching the hand.

Per photo:
  background colour = median of the top, bottom and right borders (the arm crosses the left one)
  mask              = pixels farther than --thr from that colour, largest connected component,
                      holes filled, 1-px erosion, feathered edge
  tip               = mean y of the mask's rightmost columns, at the rightmost x
  base              = centre of the mask where it meets the left crop edge (the forearm cut)
Writes <out>/<name>.png (RGBA, cropped) and <out>/hands.json {name: {"base": [x, y], "tip": [x, y]}}
plus <out>/contact_sheet.png for a visual check. deixis.stimuli.draw_photo_hand uses base/tip to
rotate the cut-out so it points from the origin at the target with a fixed base->tip length.
"""
import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps
from scipy import ndimage


def cut_out(img: Image.Image, thr: float, max_side: int):
    img = ImageOps.exif_transpose(img).convert("RGB")
    img.thumbnail((max_side, max_side), Image.LANCZOS)
    a = np.asarray(img).astype(np.float32)
    H, W, _ = a.shape
    b = max(2, min(H, W) // 50)
    border = np.concatenate([a[:b].reshape(-1, 3), a[-b:].reshape(-1, 3), a[:, -b:].reshape(-1, 3)])
    bg = np.median(border, 0)
    m = np.linalg.norm(a - bg, axis=2) > thr
    lab, n = ndimage.label(m)
    if n == 0:
        raise ValueError("no foreground found; lower --thr or use a plainer background")
    keep = 1 + int(np.argmax(ndimage.sum(m, lab, range(1, n + 1))))
    m = ndimage.binary_fill_holes(lab == keep)
    m = ndimage.binary_erosion(m, iterations=1)
    ys, xs = np.nonzero(m)
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    if x0 > W * 0.05:
        print(f"  warning: the arm does not reach the left edge (starts at x={x0}); base may be off")
    alpha = Image.fromarray((m * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(1.0))
    rgba = img.convert("RGBA")
    rgba.putalpha(alpha)
    rgba = rgba.crop((x0, y0, x1 + 1, y1 + 1))
    mc = m[y0:y1 + 1, x0:x1 + 1]
    w = mc.shape[1]
    k = max(1, w // 100)  # rightmost ~1% of columns
    ty = float(np.nonzero(mc[:, -k:].any(1))[0].mean())
    by = float(np.nonzero(mc[:, :k].any(1))[0].mean())
    return rgba, [0.0, by], [float(w - 1), ty]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", required=True, help="dir with hand photos (jpg/png/heic->convert first)")
    p.add_argument("--out", default="data/hands")
    p.add_argument("--thr", type=float, default=40.0, help="colour distance from background (0-441)")
    p.add_argument("--max_side", type=int, default=1600)
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    meta = {}
    files = sorted(f for e in ("jpg", "jpeg", "png", "JPG", "JPEG", "PNG") for f in glob.glob(os.path.join(args.src, f"*.{e}")))
    for f in files:
        name = os.path.splitext(os.path.basename(f))[0]
        print(name)
        rgba, base, tip = cut_out(Image.open(f), args.thr, args.max_side)
        rgba.save(os.path.join(args.out, f"{name}.png"))
        meta[name] = {"base": base, "tip": tip, "size": list(rgba.size)}
    with open(os.path.join(args.out, "hands.json"), "w") as fh:
        json.dump(meta, fh, indent=1)
    # contact sheet: each cut-out on black with base (green) and tip (red)
    tiles = []
    for name, d in meta.items():
        im = Image.open(os.path.join(args.out, f"{name}.png"))
        s = 300 / max(im.size)
        im = im.resize((max(1, int(im.width * s)), max(1, int(im.height * s))))
        t = Image.new("RGB", (320, 340), (0, 0, 0))
        t.paste(im, (10, 10), im)
        dr = ImageDraw.Draw(t)
        for (x, y), c in ((d["base"], (0, 255, 0)), (d["tip"], (255, 0, 0))):
            x, y = 10 + x * s, 10 + y * s
            dr.ellipse([x - 5, y - 5, x + 5, y + 5], outline=c, width=2)
        dr.text((10, 318), name, fill="white")
        tiles.append(t)
    if tiles:
        sheet = Image.new("RGB", (320 * min(4, len(tiles)), 340 * ((len(tiles) + 3) // 4)))
        for i, t in enumerate(tiles):
            sheet.paste(t, ((i % 4) * 320, (i // 4) * 340))
        sheet.save(os.path.join(args.out, "contact_sheet.png"))
    print(f"{len(meta)} hands -> {args.out}")


if __name__ == "__main__":
    main()
