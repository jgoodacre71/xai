# XAI Demo Slide Production Pack

This repository contains a production asset pipeline for a professional slide deck based on `xai_demo.ipynb`.

The final deck argues that XAI is not merely post-hoc audit. It is a way to make learned structure visible, testable, and correctable.

## Core thesis

A model can pass the sampled exam while learning the wrong function.

The useful explanation is the factor that controls the model score under intervention.

## Build

Install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run everything:

```bash
make all
```

Outputs:

- `assets/images/`
- `assets/equations/`
- `assets/video/`
- `deck/`
- `powerpoint/`

## No zip files

This repository intentionally stores all assets as normal files and folders.

## Final presentation

Use Microsoft PowerPoint/Copilot with:

- `deck/copilot_prompt.md`
- `deck/slide_storyboard.md`
- `deck/slide_manifest.yaml`
- `assets/`

Final outputs should be:

- `final/xai_demo_final.pptx`
- `final/xai_demo_final.pdf`
- `final/xai_demo_final.mp4`

## Story

1. Supervised learning and empirical risk
2. Many functions can pass the same finite exam
3. Same object, different position
4. Response maps reveal learned geometry
5. Position shortcut data audit
6. CNN appears to work again
7. Same object, invisible background shift
8. Background shortcut data audit
9. Heatmaps are not enough
10. Mitigation and re-test
11. XAI as experimental model science
