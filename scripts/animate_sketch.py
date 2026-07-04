#!/usr/bin/env python3
"""animate_sketch.py — turn a matched dip-pen pair (outline + color) into an
MP4 of the drawing process, replaying the sketch the way a human would draw
it: a pen tip travels ALONG each ink stroke (skeleton path), revealing the
stroke's width as it moves; then the marker color is laid down as discrete
overlapping diagonal swipes, region by region.

Usage:
  python3 animate_sketch.py OUTLINE.jpg COLOR.jpg OUT.mp4 \
      [--fps 30] [--ink-seconds 8] [--marker-seconds 5] \
      [--pause 0.75] [--end-hold 1.5] [--max-size 1600] [--no-cursor]

The two inputs must be the matched pair the dip-pen-sketch-combo flow
produces: identical linework, identical pure-white background, identical
dimensions. Frames reveal the actual pixels of those two images — nothing is
redrawn — so the animation is deterministic and its final frame IS the color
image (after any --max-size resize).

How the human-drawn look is achieved:
- The ink is skeletonized (medial axis). Skeleton pixels are traced into
  strokes by always continuing in the STRAIGHTEST direction through
  crossings, so two hatching lines that intersect are replayed as two
  straight pen strokes, not one blob.
- Strokes are ordered by greedy nearest-neighbor travel from the top of the
  figure — the pen finishes a stroke and moves to the closest next one, so
  it works around the drawing like a person instead of teleporting.
- The tip reveals a disc matched to the stroke's local thickness (medial
  axis distance + a margin for the anti-aliased fringe), so lines grow
  tip-to-tail at constant speed.
- Phase transitions crossfade (~0.3–0.5s) instead of hard-swapping, so
  sub-threshold pixels never pop in a single frame.

Requires: numpy, Pillow, scipy, scikit-image, and ffmpeg on PATH.
"""
import argparse
import subprocess
import sys

import numpy as np
from PIL import Image, ImageFilter

try:
    from scipy import ndimage
    from skimage.morphology import medial_axis
except ImportError as e:
    sys.exit(f"error: missing dependency ({e.name}). Install with: pip install scipy scikit-image")

INK_CORE = 200        # gray level below which a pixel is confident ink
INK_FRINGE = 248      # gray level below which a pixel belongs to the ink phase (AA fringe)
MARKER_DIFF = 24      # per-channel |color-outline| above which a pixel is marker
BAND_WIDTH = 36       # diagonal marker swipe width, px at 1600 (scaled)

NB8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


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
    w -= w % 2
    h -= h % 2
    if (w, h) != outline.size:
        outline = outline.resize((w, h), Image.LANCZOS)
        color = color.resize((w, h), Image.LANCZOS)
    return np.asarray(outline).copy(), np.asarray(color).copy()


def trace_strokes(core):
    """Skeletonize the ink and trace it into strokes (polylines of skeleton
    points with local radii). At junctions the tracer continues in the
    straightest direction, so crossing lines replay as separate strokes."""
    skel, dist = medial_axis(core, return_distance=True)
    h, w = skel.shape
    sk = skel.copy()

    # Degree map (8-connectivity neighbour count on the skeleton).
    k = np.ones((3, 3), dtype=np.uint8)
    deg = ndimage.convolve(skel.astype(np.uint8), k, mode="constant") - skel.astype(np.uint8)

    visited = np.zeros_like(sk, dtype=bool)

    def neighbors(y, x):
        for dy, dx in NB8:
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and sk[ny, nx] and not visited[ny, nx]:
                yield ny, nx

    def walk(y0, x0):
        path = [(y0, x0)]
        visited[y0, x0] = True
        while True:
            cands = list(neighbors(*path[-1]))
            if not cands:
                break
            if len(path) >= 3:
                (py, px), (cy, cx) = path[-3], path[-1]
                dy, dx = cy - py, cx - px
                n = max(1e-6, (dy * dy + dx * dx) ** 0.5)
                dy, dx = dy / n, dx / n
                # straightest continuation through crossings
                cands.sort(key=lambda p: -((p[0] - cy) * dy + (p[1] - cx) * dx))
            ny, nx = cands[0]
            visited[ny, nx] = True
            path.append((ny, nx))
        return path

    strokes = []
    ys, xs = np.nonzero(sk & (deg == 1))          # endpoints first: natural stroke starts
    for y, x in zip(ys, xs):
        if not visited[y, x]:
            strokes.append(walk(y, x))
    ys, xs = np.nonzero(sk)                        # leftovers: loops / dots
    for y, x in zip(ys, xs):
        if not visited[y, x]:
            strokes.append(walk(y, x))
    return strokes, dist


def order_strokes(strokes):
    """Greedy nearest-neighbor travel starting from the topmost stroke, so
    the pen moves to the closest next stroke instead of teleporting."""
    if not strokes:
        return []
    remaining = list(range(len(strokes)))
    starts = np.array([s[0] for s in strokes], dtype=np.float64)
    ends = np.array([s[-1] for s in strokes], dtype=np.float64)
    order = []
    cur = min(remaining, key=lambda i: (strokes[i][0][0], strokes[i][0][1]))
    pos = np.array(strokes[cur][-1], dtype=np.float64)
    order.append((cur, False))
    remaining.remove(cur)
    rem = np.array(remaining, dtype=np.intp)
    while rem.size:
        d_start = ((starts[rem] - pos) ** 2).sum(1)
        d_end = ((ends[rem] - pos) ** 2).sum(1)
        best = int(np.argmin(np.minimum(d_start, d_end)))
        idx = int(rem[best])
        flip = d_end[best] < d_start[best]
        order.append((idx, bool(flip)))
        pos = np.array(strokes[idx][0] if flip else strokes[idx][-1], dtype=np.float64)
        rem = np.delete(rem, best)
    return [(strokes[i][::-1] if flip else strokes[i]) for i, flip in order]


def disc_offsets(max_r=12):
    table = {}
    for r in range(1, max_r + 1):
        yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
        m = yy * yy + xx * xx <= r * r
        table[r] = (yy[m], xx[m])
    return table


def build_marker_strokes(outline, color, band_width):
    """Decompose the marker layer into chisel-tip STROKES, the way the ink is
    decomposed into pen strokes: regions top-down; within a region, one
    stroke per diagonal band — the tip travels the band's centerline
    (alternating direction, like scrubbing back and forth) and lays the
    color down around itself as it moves. Returns (strokes, mask) where each
    stroke is a list of (y, x) tip positions spaced ~band_width/4 apart."""
    diff = np.abs(color.astype(np.int16) - outline.astype(np.int16)).max(axis=2)
    mask_img = Image.fromarray(((diff > MARKER_DIFF) * 255).astype(np.uint8))
    mask = np.asarray(mask_img.filter(ImageFilter.MedianFilter(5))) > 127
    labels, n = ndimage.label(mask, structure=np.ones((3, 3)))
    strokes = []
    if n == 0:
        return strokes, mask
    tops = ndimage.minimum(np.indices(mask.shape)[0], labels, index=range(1, n + 1))
    step = max(4, band_width // 4)
    for lab in np.argsort(tops) + 1:               # regions top-down
        ys, xs = np.nonzero(labels == lab)
        band = (ys + xs) // band_width
        for b in np.unique(band):
            sel = band == b
            by, bx = ys[sel], xs[sel]
            buckets = (bx - by) // step            # position along the band
            ubuckets = np.unique(buckets)
            if b % 2:
                ubuckets = ubuckets[::-1]          # zigzag: alternate pull direction
            path = []
            for ub in ubuckets:
                m = buckets == ub
                path.append((int(round(by[m].mean())), int(round(bx[m].mean()))))
            strokes.append(path)
    return strokes, mask


def frame_counts(total, n_frames):
    if n_frames <= 0 or total == 0:
        return []
    t = np.linspace(0.0, 1.0, n_frames)
    eased = t + (t * (1 - t)) * 0.2               # nearly linear, slight ease
    counts = np.round(eased / eased[-1] * total).astype(np.int64)
    counts[-1] = total
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
    band = max(20, int(round(BAND_WIDTH * scale)))

    gray = outline.mean(axis=2)
    fringe = gray < INK_FRINGE
    core = gray < INK_CORE

    print(f"[animate] {w}x{h} · skeletonizing + tracing strokes…", flush=True)
    strokes, dist = trace_strokes(core)
    strokes = order_strokes(strokes)
    total_pts = sum(len(s) for s in strokes)
    print(f"[animate] {len(strokes):,} strokes · {total_pts:,} skeleton px · building marker swipes…", flush=True)
    mk_strokes, mk_mask = build_marker_strokes(outline, color, band)
    print(f"[animate] {len(mk_strokes):,} marker swipes · rendering…", flush=True)

    n_ink = max(1, int(round(args.fps * args.ink_seconds)))
    n_pause = int(round(args.fps * args.pause))
    n_marker = max(1, int(round(args.fps * args.marker_seconds)))
    n_hold = max(1, int(round(args.fps * args.end_hold)))
    n_fade1 = min(max(2, int(round(args.fps * 0.3))), max(2, n_pause))
    n_fade2 = max(2, int(round(args.fps * 0.5)))

    ff = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{w}x{h}", "-r", str(args.fps), "-i", "-",
         "-c:v", "libx264", "-preset", "medium", "-crf", "18",
         "-pix_fmt", "yuv420p", "-movflags", "+faststart", args.output],
        stdin=subprocess.PIPE,
    )

    def emit(frame):
        ff.stdin.write(frame.tobytes())

    def crossfade(canvas, target, n):
        for i in range(1, n + 1):
            a = i / n
            emit((canvas.astype(np.float32) * (1 - a) + target.astype(np.float32) * a).astype(np.uint8))
        return target.copy()

    canvas = np.full_like(outline, 255)
    discs = disc_offsets()
    revealed = np.zeros((h, w), dtype=bool)

    # ---- Phase 1: pen replays each stroke tip-to-tail ----
    flat = [(y, x, int(min(12, max(2, round(dist[y, x] + 1.5))))) for s in strokes for (y, x) in s]
    prev = 0
    for count in frame_counts(len(flat), n_ink):
        for (y, x, r) in flat[prev:count]:
            oy, ox = discs[r]
            py, px = np.clip(y + oy, 0, h - 1), np.clip(x + ox, 0, w - 1)
            sel = fringe[py, px] & ~revealed[py, px]
            if sel.any():
                sy, sx = py[sel], px[sel]
                revealed[sy, sx] = True
                canvas[sy, sx] = outline[sy, sx]
        prev = count
        if args.no_cursor or count == 0:
            emit(canvas)
        else:
            fr = canvas.copy()
            cy, cx, _ = flat[count - 1]
            draw_cursor(fr, cy, cx, pen_radius, np.array([20, 20, 20], dtype=np.uint8))
            emit(fr)

    # settle onto the exact outline (covers residue smoothly), then pause
    canvas = crossfade(canvas, outline, n_fade1)
    for _ in range(max(0, n_pause - n_fade1)):
        emit(canvas)

    # ---- Phase 2: the chisel tip pulls each marker swipe, region by region ----
    chisel = max(8, int(round(band * 0.7)))
    if chisel not in discs:
        yy, xx = np.mgrid[-chisel:chisel + 1, -chisel:chisel + 1]
        m = yy * yy + xx * xx <= chisel * chisel
        discs[chisel] = (yy[m], xx[m])
    flat_mk = [(y, x) for s in mk_strokes for (y, x) in s]
    revealed_mk = np.zeros((h, w), dtype=bool)
    prev = 0
    for count in frame_counts(len(flat_mk), n_marker):
        for (y, x) in flat_mk[prev:count]:
            oy, ox = discs[chisel]
            py, px = np.clip(y + oy, 0, h - 1), np.clip(x + ox, 0, w - 1)
            sel = mk_mask[py, px] & ~revealed_mk[py, px]
            if sel.any():
                sy, sx = py[sel], px[sel]
                revealed_mk[sy, sx] = True
                canvas[sy, sx] = color[sy, sx]
        prev = count
        if args.no_cursor or count == 0:
            emit(canvas)
        else:
            fr = canvas.copy()
            cy, cx = flat_mk[count - 1]
            tip = color[cy, cx].astype(np.float32)
            # chisel tip: the laid color with a darker rim so it reads against
            # the color it is laying down
            draw_cursor(fr, cy, cx, marker_radius, (tip * 0.55).astype(np.uint8))
            draw_cursor(fr, cy, cx, max(2, marker_radius - 3), tip.astype(np.uint8))
            emit(fr)

    # settle onto the exact color image, then hold
    canvas = crossfade(canvas, color, n_fade2)
    for _ in range(max(0, n_hold - 1)):
        emit(canvas)

    ff.stdin.close()
    if ff.wait() != 0:
        sys.exit("error: ffmpeg failed")
    total = (n_ink + max(n_pause, n_fade1) + n_marker + n_fade2 + max(0, n_hold - 1)) / args.fps
    print(f"[animate] wrote {args.output} · ~{total:.1f}s @ {args.fps}fps")


if __name__ == "__main__":
    main()
