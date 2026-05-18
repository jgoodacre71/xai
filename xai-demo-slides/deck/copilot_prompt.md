# Copilot Prompt for PowerPoint

Create a professional scientific presentation from the attached storyboard and assets.

Topic:
Explainable AI as experimental model science, using a controlled moon/star shortcut-learning demo.

Audience:
Technical research audience: machine learning, statistics, XAI, anomaly detection, scientific ML.

Tone:
Professional, scientific, clear, slightly dramatic at reveal moments.

Design:
- off-white background
- charcoal text
- restrained accent colours
- no stock art
- no decorative icons
- no busy template
- one main proof object per slide
- longer explanation in speaker notes
- do not create an image gallery

Use the supplied assets only:
- images from `assets/images/`
- equations from `assets/equations/`
- videos from `assets/video/mp4/`

Do not invent new visuals.
Do not use unrelated AI imagery.
Do not rename technical concepts.

Important:
The story should unfold. Do not title early slides "Act I" or "Act II". Do not reveal the shortcuts before the counterfactual slides.

Slide sequence:
Use `deck/slide_storyboard.md` and `deck/slide_manifest.yaml`.

For animations:
- Use first-frame image as placeholder.
- Insert MP4 from local file.
- Set major reveal animations to play on click.
- Keep reveal videos large and central.

Core thesis:
Accuracy tells us that a model passed the sampled exam.
XAI asks which factor controlled the score.
The useful explanation is not a heatmap alone; it is the factor-control story:
hypothesise -> intervene -> measure -> change process -> re-test.

Create a slide deck that is clean, minimal, and research-grade.
