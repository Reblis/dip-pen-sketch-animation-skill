# dip-pen-sketch-animation — a Claude Code skill by [Reblis.com](https://reblis.com)

Turn any photo into a rough old-school dip-pen sketch **and a video of it being drawn**:

```
dip-pen-sketch-animation /path/to/photo.jpg
```

![pen strokes draw the ink, then marker sweeps the color in](example.gif)

It produces **three files** from one photo: a pure black-ink line sketch, the *same drawing* with loose alcohol-marker color on top, and an **MP4 animation of the drawing process** — the pen strokes appear first, ink line by ink line, then the marker color sweeps in, ending on the finished colored sketch. The whole "Philly sketch" TikTok/CapCut look, as a reveal video.

The stills follow the [dip-pen-sketch-combo](https://github.com/Reblis/dip-pen-sketch-combo-skill) rules: the outline is **derived directly from the finished colored image**, so the linework matches **stroke-for-stroke by construction**, and both are flattened to the same pure-white paper. The animation is then built **programmatically from those two images** — every frame reveals their actual pixels (connected ink strokes first, then diagonal marker bands) — so the video is deterministic, always ends exactly on the delivered stills, and costs nothing beyond the two image generations.

> Want stills only? See the sibling skills **[dip-pen-sketch-combo](https://github.com/Reblis/dip-pen-sketch-combo-skill)** (matched pair), **[dip-pen-sketch-outline](https://github.com/Reblis/dip-pen-sketch-outline-skill)** (ink only), and **[dip-pen-sketch-color](https://github.com/Reblis/dip-pen-sketch-color-skill)** (color only).

## Install

```bash
git clone https://github.com/Reblis/dip-pen-sketch-animation-skill ~/.claude/skills/dip-pen-sketch-animation
```

Restart Claude Code (or start a new session) so the skill registers, then run it on any photo path or image URL.

## Requirements

The two sketch images come from an AI image-editing model — that part is **not** local/free. The animation render **is** local and free:

- **[Claude Code](https://claude.com/claude-code)**
- **A connected [fal.ai](https://fal.ai) MCP server with credits.** The skill runs **`fal-ai/nano-banana-2/edit`** — Google's **Nano Banana 2** — twice (base ink sketch, then color-on-top). **Cost ≈ $0.06–0.08 per run.**
- `curl` and **ImageMagick** (`convert`) for fetching, resizing, and background whitening.
- **`python3` + `numpy`, `Pillow`, `scipy`, `scikit-image`, and `ffmpeg`** for the animation (`pip install numpy pillow scipy scikit-image` · `apt install ffmpeg`).
- *Optional, free:* a connected **nanobanana / Gemini MCP** runs the same model family at no cost — but its free tier is frequently rate-limited, which is why fal.ai is the default path.

## What you get

- **`<name>_anim_outline.jpg`** — rough scratchy dip-pen ink sketch, background removed, blank white paper.
- **`<name>_anim_color.jpg`** — the identical drawing with loose Copic-style marker color.
- **`<name>_dippen_animation.mp4`** — ~15s, 30fps (both tunable): pen strokes draw the ink (~8s), a beat on the finished outline, marker sweeps the color in diagonally (~5s), hold on the finished piece. A pen/marker cursor dot rides the reveal frontier; disable with `--no-cursor`.
- **Likeness preserved** — faces, expressions, hair, and clothing stay recognizable.

## How the animation works

The bundled `scripts/animate_sketch.py` replays the drawing from the image's own marks. The ink is skeletonized and traced into individual pen strokes — continuing in the straightest direction through crossings, so two hatch lines that intersect replay as two separate strokes — ordered by nearest-neighbor pen travel (the tip finishes a stroke and moves to the closest next one, never teleporting) and revealed tip-to-tail at true local width. The marker phase recovers the **actual painted strokes**: marker pixels are clustered into tone layers, and each tone's connected swaths — the streaks as the marker laid them — are traced along their own direction and replayed light-to-dark, so the base coat goes down first and the darker shading passes appear visibly on top, the way real marker work layers. Phase transitions crossfade, and frames stream straight to ffmpeg. No video model, no re-generation: the final frame *is* the color deliverable, bit for bit.

## Inputs

A local image path, a direct image URL, or — for Instagram/Facebook/Pinterest links (which are web pages, not images, and block hotlinking) — the skill downloads the photo first.

## Credit

Built by [Reblis.com](https://reblis.com). Sketch generations use Google's Nano Banana 2 via [fal.ai](https://fal.ai). MIT licensed.
