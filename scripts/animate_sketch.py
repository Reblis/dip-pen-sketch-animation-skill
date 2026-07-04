#!/usr/bin/env python3
"""animate_sketch.py — turn a matched dip-pen pair (outline + color) into an
MP4 of the drawing process: the ink appears stroke by stroke like a pen
drawing it, then the marker color fills in with diagonal sweeps.

Usage:
  python3 animate_sketch.py OUTLINE.jpg COLOR.jpg OUT.mp4 \
      [--fps 30] [--ink-seconds 8] [--marker-seconds 5] \
      [--pause 0.75] [--end-hold 1.5] [--max-size 1600] [--no-cursor]

The two inputs must be the matched pair the dip-pen-sketch-combo flow
produces: identical linework, identical pure-white background, identical
dimensions. Because frames are built by progressively revealing the actual
pixels of those two images, the animation is deterministic and its final
frame IS the color image (after any --max-size resize) — nothing is redrawn
or re-generated, so the video always matches the stills exactly.

Requires: numpy, Pillow, ffmpeg on PATH.
"""
import argparse
import subprocess
import sys
from collections import deque

import numpy as np
from PIL import Image, ImageFilter

INK_THRESHOLD = 200     # gray level below which an outline pixel counts as ink
MARKER_DIFF = 24        # per-channel |color-outline| above which a pixel is marker
BAND_WIDTH = 48         # diagonal marker stroke width, px (scaled with image)


def load_pair(outline_path, color_path, max_size):
    outline = Image.open(outline_path).convert("RGB")
    color = Image.open(color_path).convert("RGB")
    if outline.size != color.size:
        sys.exit(f"error: dimension mismatch {outline.size} vs {color.size} — "
                 "inputs must be the matched pair from the combo flow")
    w, h = outline.size
    scale = min(1.0, max_size / max(w, h))
    if scale < 1.0:
        w, h = int(w * scale), int(h * scale)
    w -= w % 2  # yuv420p needs even dimensions
    h -= h % 2
    if (w, h) != outline.size:
        outline = outline.resize((w, h), Image.LANCZOS)
        color = color.resize((w, h), Image.LANCZOS)
    return np.asarray(outline).copy(), np.asarray(color).copy()


def order_ink_pixels(outline):
    """Order ink pixels the way a pen would draw them: connected strokes,
    top-to-bottom across the figure, each stroke traversed continuously
    (depth-first, so branches are followed to their end before backtracking).
    """
    gray = outline.mean(axis=2)
    mask = gray < INK_THRESHOLD
    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    components = []
    neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    ys_all, xs_all = np.nonzero(mask)
    # Iterate seeds in scan order so component discovery is top-to-bottom.
    for y0, x0 in zip(ys_all, xs_all):
        if visited[y0, x0]:
            continue
        # Depth-first walk = pen following one branch to its end, then the next.
        stack = [(y0, x0)]
        visited[y0, x0] = True
        path = []
        while stack:
            y, x = stack.pop()
            path.append((y, x))
            for dy, dx in neighbors:
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not visited[ny, nx]:
                    visited[ny, nx] = True
                    stack.append((ny, nx))
        components.append(path)

    ordered = [p for comp in components for p in comp]
    ys = np.fromiter((p[0] for p in ordered), dtype=np.int32, count=len(ordered))
    xs = np.fromiter((p[1] for p in ordered), dtype=np.int32, count=len(ordered))
    return ys, xs


def order_marker_pixels(outline, color, band_width):
    """Order marker pixels like marker strokes: region by region (top-left
    regions first), each region filled in diagonal zigzag bands."""
    diff = np.abs(color.astype(np.int16) - outline.astype(np.int16)).max(axis=2)
    mask_img = Image.fromarray(((diff > MARKER_DIFF) * 255).astype(np.uint8))
    mask = np.asarray(mask_img.filter(ImageFilter.MedianFilter(5))) > 127
    h, w = mask.shape
    labels = np.full((h, w), -1, dtype=np.int32)
    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    region_order = []

    ys_all, xs_all = np.nonzero(mask)
    for y0, x0 in zip(ys_all, xs_all):
        if labels[y0, x0] != -1:
            continue
        rid = len(region_order)
        labels[y0, x0] = rid
        q = deque([(y0, x0)])
        while q:
            y, x = q.popleft()
            for dy, dx in neighbors:
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and labels[ny, nx] == -1:
                    labels[ny, nx] = rid
                    q.append((ny, nx))
        region_order.append(rid)

    ys, xs = np.nonzero(mask)
    lab = labels[ys, xs]
    band = (ys + xs) // band_width
    # Zigzag: alternate the within-band direction so the "marker" sweeps
    # back and forth instead of jumping.
    within = np.where(band % 2 == 0, xs - ys, -(xs - ys))
    order = np.lexsort((within, band, lab))
    return ys[order].astype(np.int32), xs[order].astype(np.int32)


def frame_counts(total_pixels, n_frames):
    """Cumulative pixel counts per frame with a gentle ease-out."""
    if n_frames <= 0 or total_pixels == 0:
        return []
    t = np.linspace(0.0, 1.0, n_frames)
    eased = 1.0 - (1.0 - t) ** 1.6
    counts = np.round(eased * total_pixels).astype(np.int64)
    counts[-1] = total_pixels
    return counts


def draw_cursor(frame, y, x, radius, rgb):
    h, w = frame.shape[:2]
    yy, xx = np.ogrid[max(0, y - radius):min(h, y + radius + 1),
                      max(0, x - radius):min(w, x + radius + 1)]
    circle = (yy - y) ** 2 + (xx - x) ** 2 <= radius ** 2
    frame[max(0, y - radius):min(h, y + radius + 1),
          max(0, x - radius):min(w, x + radius + 1)][circle] = rgb


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("outline")
    ap.add_argument("color")
    ap.add_argument("output")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--ink-seconds", type=float, default=8.0)
    ap.add_argument("--marker-seconds", type=float, default=5.0)
    ap.add_argument("--pause", type=float, default=0.75, help="hold on the finished outline before the marker phase")
    ap.add_argument("--end-hold", type=float, default=1.5, help="hold on the finished color image")
    ap.add_argument("--max-size", type=int, default=1600, help="downscale longest side to this before rendering")
    ap.add_argument("--no-cursor", action="store_true", help="disable the pen/marker cursor dot")
    args = ap.parse_args()

    outline, color = load_pair(args.outline, args.color, args.max_size)
    h, w = outline.shape[:2]
    scale = max(h, w) / 1600.0
    pen_radius = max(3, int(round(4 * scale)))
    marker_radius = max(7, int(round(10 * scale)))
    band = max(24, int(round(BAND_WIDTH * scale)))

    print(f"[animate] {w}x{h} · ordering ink strokes…", flush=True)
    ink_ys, ink_xs = order_ink_pixels(outline)
    print(f"[animate] {len(ink_ys):,} ink px · ordering marker strokes…", flush=True)
    mk_ys, mk_xs = order_marker_pixels(outline, color, band)
    print(f"[animate] {len(mk_ys):,} marker px · rendering…", flush=True)

    n_ink = max(1, int(round(args.fps * args.ink_seconds)))
    n_pause = int(round(args.fps * args.pause))
    n_marker = max(1, int(round(args.fps * args.marker_seconds)))
    n_hold = int(round(args.fps * args.end_hold))

    ff = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{w}x{h}", "-r", str(args.fps), "-i", "-",
         "-c:v", "libx264", "-preset", "medium", "-crf", "18",
         "-pix_fmt", "yuv420p", "-movflags", "+faststart", args.output],
        stdin=subprocess.PIPE,
    )

    def emit(frame):
        ff.stdin.write(frame.tobytes())

    canvas = np.full_like(outline, 255)

    # Phase 1 — pen: reveal ink pixels along the stroke order.
    prev = 0
    for count in frame_counts(len(ink_ys), n_ink):
        ys, xs = ink_ys[prev:count], ink_xs[prev:count]
        canvas[ys, xs] = outline[ys, xs]
        prev = count
        if args.no_cursor or count == 0:
            emit(canvas)
        else:
            fr = canvas.copy()
            cy, cx = int(ink_ys[count - 1]), int(ink_xs[count - 1])
            draw_cursor(fr, cy, cx, pen_radius, np.array([20, 20, 20], dtype=np.uint8))
            emit(fr)

    # Pause on the finished outline.
    canvas = outline.copy()
    for _ in range(n_pause):
        emit(canvas)

    # Phase 2 — marker: reveal color pixels in diagonal strokes over the outline.
    prev = 0
    for count in frame_counts(len(mk_ys), n_marker):
        ys, xs = mk_ys[prev:count], mk_xs[prev:count]
        canvas[ys, xs] = color[ys, xs]
        prev = count
        if args.no_cursor or count == 0:
            emit(canvas)
        else:
            fr = canvas.copy()
            cy, cx = int(mk_ys[count - 1]), int(mk_xs[count - 1])
            draw_cursor(fr, cy, cx, marker_radius, color[cy, cx])
            emit(fr)

    # End: the exact color image, held.
    canvas = color.copy()
    for _ in range(max(1, n_hold)):
        emit(canvas)

    ff.stdin.close()
    if ff.wait() != 0:
        sys.exit("error: ffmpeg failed")
    total = (n_ink + n_pause + n_marker + max(1, n_hold)) / args.fps
    print(f"[animate] wrote {args.output} · {total:.1f}s @ {args.fps}fps")


if __name__ == "__main__":
    main()
