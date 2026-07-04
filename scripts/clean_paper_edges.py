#!/usr/bin/env python3
"""clean_paper_edges.py — remove faint paper-edge smudges from a matched
dip-pen pair (outline + color), in place, identically on both files.

Generations sometimes render a soft paper-edge shadow along the image
borders. The corner-sampled whiten step can miss it (the flood only matches
the one tone it sampled), leaving gray bands and speckle hugging the edges
that later replay as stray marks in the animation.

This sweeps the border margin and whites out components that read as edge
junk while protecting real artwork:
- only components contained ENTIRELY within the border margin are touched —
  anything connected to the figure extends inward and is kept
- faintness required (component mean gray > 105 in the outline), so dark
  ink overshooting toward the edge survives; tiny faint specks go
  regardless of size
- low saturation required (mean saturation < 0.06 in the color image), so
  marker color reaching the edge is never eaten

Usage: python3 clean_paper_edges.py OUTLINE.jpg COLOR.jpg [--margin 0.025]

Requires: numpy, Pillow, scipy.
"""
import argparse
import sys

import numpy as np
from PIL import Image

try:
    from scipy import ndimage
except ImportError:
    sys.exit("error: missing dependency scipy. Install with: pip install scipy")


def mean_saturation(rgb):
    a = rgb.astype(np.float32) / 255.0
    mx, mn = a.max(1), a.min(1)
    l = (mx + mn) / 2
    d = mx - mn
    s = np.where(d == 0, 0.0,
                 np.where(l < 0.5, d / np.maximum(mx + mn, 1e-6),
                          d / np.maximum(2 - mx - mn, 1e-6)))
    return float(s.mean())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("outline")
    ap.add_argument("color")
    ap.add_argument("--margin", type=float, default=0.025,
                    help="border band as a fraction of the longest side (default 0.025)")
    args = ap.parse_args()

    o_img = Image.open(args.outline).convert("RGB")
    c_img = Image.open(args.color).convert("RGB")
    if o_img.size != c_img.size:
        sys.exit(f"error: dimension mismatch {o_img.size} vs {c_img.size}")
    outline = np.asarray(o_img).copy()
    color = np.asarray(c_img).copy()
    h, w = outline.shape[:2]
    m = max(8, int(round(max(h, w) * args.margin)))

    in_band = np.zeros((h, w), dtype=bool)
    in_band[:m, :] = in_band[-m:, :] = True
    in_band[:, :m] = in_band[:, -m:] = True

    gray = outline.mean(axis=2)
    nonwhite = (gray < 250) | (color.min(axis=2) < 250)
    labels, n = ndimage.label(nonwhite, structure=np.ones((3, 3)))

    removed_px = removed_comps = 0
    remove_mask = np.zeros((h, w), dtype=bool)
    for sl_idx, sl in enumerate(ndimage.find_objects(labels), start=1):
        if sl is None:
            continue
        comp = labels[sl] == sl_idx
        # entirely inside the border band?
        if not in_band[sl][comp].all():
            continue
        cy, cx = np.nonzero(comp)
        cy, cx = cy + sl[0].start, cx + sl[1].start
        g = float(gray[cy, cx].mean())
        sat = mean_saturation(color[cy, cx])
        if sat < 0.06 and (g > 105 or len(cy) < 150):
            remove_mask[cy, cx] = True
            removed_px += len(cy)
            removed_comps += 1

    # Extreme-edge strip: per-pixel sweep for faint desaturated residue that
    # rides a protected component (e.g. a speck trail connected to marker
    # color that legitimately reaches the border). Only faint (gray > 105)
    # AND desaturated pixels within ~m/6 of the border are cleared, so ink
    # and colored marker touching the edge survive pixel-for-pixel.
    e = max(4, m // 6)
    strip = np.zeros((h, w), dtype=bool)
    strip[:e, :] = strip[-e:, :] = True
    strip[:, :e] = strip[:, -e:] = True
    cand = strip & nonwhite & (gray > 105) & ~remove_mask
    if cand.any():
        cy, cx = np.nonzero(cand)
        a = color[cy, cx].astype(np.float32) / 255.0
        mx, mn = a.max(1), a.min(1)
        l = (mx + mn) / 2
        d = mx - mn
        s = np.where(d == 0, 0.0,
                     np.where(l < 0.5, d / np.maximum(mx + mn, 1e-6),
                              d / np.maximum(2 - mx - mn, 1e-6)))
        faint = s < 0.06
        remove_mask[cy[faint], cx[faint]] = True
        removed_px += int(faint.sum())

    if removed_px:
        outline[remove_mask] = 255
        color[remove_mask] = 255
        Image.fromarray(outline).save(args.outline, quality=95)
        Image.fromarray(color).save(args.color, quality=95)
    print(f"[clean] removed {removed_comps} edge components + strip residue "
          f"({removed_px:,} px total) within a {m}px border band; artwork untouched")


if __name__ == "__main__":
    main()
