# PowerPoint Finalisation Instructions

Goal:
Create a self-contained `.pptx` with embedded MP4 animations, plus PDF and MP4 backups.

## Step 1: Open PowerPoint

Use Microsoft PowerPoint desktop if possible.

## Step 2: Build from storyboard

Use:
- `deck/slide_storyboard.md`
- `deck/slide_manifest.yaml`
- `deck/copilot_prompt.md`

## Step 3: Insert static assets

Use:
- `assets/images/slide_plates/`
- `assets/equations/svg/`
- `assets/equations/png/`

Preferred:
- SVG equations if PowerPoint handles them well.
- PNG equations if SVG import is awkward.

## Step 4: Insert videos

For each animation:
- Insert -> Video -> This Device
- Select file from `assets/video/mp4/`
- Set playback to Play on Click
- Place first-frame fallback image behind or on the previous slide.

Major reveal animations:
- moon movement confidence
- star movement confidence if used
- response-map path if available
- invisible background morph moon
- invisible background morph star if used

## Step 5: Make deck self-contained

In PowerPoint:
- ensure videos are embedded, not linked;
- use media compatibility/optimisation tools if available;
- save as `.pptx`.

## Step 6: Export backups

Create:
- `final/xai_demo_final.pptx`
- `final/xai_demo_final.pdf`
- `final/xai_demo_final.mp4`

The `.pptx` is the live deck.
The `.pdf` is the static emergency version.
The `.mp4` is the guaranteed playback version.

## Step 7: Test

Test the `.pptx` on another machine before presenting.

## Optional skeleton status

`python-pptx` is not installed in this environment, so `skeleton_static.pptx` was not generated. This does not block the asset pipeline; use the storyboard, manifest, slide plates, equations, and MP4s in PowerPoint/Copilot.
