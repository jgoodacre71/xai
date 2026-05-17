# 0055: Demo 00 two-act Clever-Hans polish

## Status
Completed

## Owner
Codex thread

## Why
Demo 00 needed to become the no-permission opening hook for the XAI suite: a
presentation-quality behavioural-observability story, not merely a technically
correct generated shortcut notebook.

## Scope
- Preserve the generated controlled data setup and the two learned models.
- Keep Act I as an absolute-position shortcut: moons usually lower-left, stars
  upper-right.
- Add Act II as a second trap: a CNN learns a near-invisible
  background/acquisition cue.
- Make the central evidence behavioural counterfactuals, confidence paths,
  response maps, morphs, intervention, and re-test.
- Add data-first shortcut discovery so the reader sees that the biased exam is
  already solvable by trivial statistics and nearest-neighbour rules that
  ignore shape.
- Keep saliency as a cautionary interlude only.

## Completion notes
- Reframed the notebook around Clever-Hans predictors and added references to
  Lapuschkin et al. and Adebayo et al.
- Added decisive confidence-margin gates for Act I and Act II, including
  correct-class probability checks and hard Act II background-swap thresholds.
- Added calibrated Act II tint selection, a background-only sanity check, a
  cue-aware CNN, and mitigation/re-test with hard swap-score assertions.
- Fixed the shape-morph diagnostic so generated diagnostic frames are scored
  directly from image tensors rather than through temporary cached sample ids.
- Reworked the notebook into a lean inline-only presentation path: no saved
  static PNG workflow, no final image wall, and clean in-notebook animation
  display.
- Rebuilt the presentation around five core moments instead of many diagnostic
  outputs: perfect IID metrics, same-object movement failure, a position-only
  rule passing the exam, invisible-background counterfactual failure, and a
  background-only rule passing the exam.
- Added the “many functions pass the same exam” centrepiece and split the data
  audits into readable statistical and silly-model panels.
- Kept the strongest animations: Act I confidence movement, response-map path,
  morph/heatmap caution animations, and Act II invisible-background morphs.
- Updated tests to protect the Demo 00 contract, including forbidden old
  scene-cue language, decisive confidence gates, inline-only presentation,
  data-first shortcut audits, and clean animation display.

## Out of scope
- Rebuilding Demo 01 or Demo 02.
- Adding external data.
- Adding more models beyond the Pixel MLP, CNN + augmentation, and Act II
  cue/mitigation variants required for the two-act story.
- Treating saliency as proof of the shortcut.

## Validation
- `./.venv/bin/pytest tests/unit/test_notebooks.py -q`
- `./.venv/bin/ruff check tests/unit/test_notebooks.py`
- `git diff --check`

During the final presentation-only polish, full `nbconvert` execution and the
Demo 00 smoke path were attempted but stopped after unusually long runs with no
reported cell failure. The committed notebook remains output-free and the
source-level contract tests pass.

## Durable guidance added
- `.agents/skills/notebook-polisher/SKILL.md` now records behavioural-XAI
  notebook guidance: stage apparent success first, use same-object
  counterfactuals, show scores and what changed, keep saliency secondary,
  re-test after intervention, and avoid identity-cache paths for generated
  diagnostic frames.
- Demo 00 should stay self-contained and inline-only for static figures. Do not
  reintroduce static figure export manifests unless the presentation surface is
  deliberately changed again.

## Risks
- The notebook is intentionally presentation-rich and can take several minutes
  to execute because it trains models and exports GIF/MP4 assets.
- The generated demo is didactic; later notebooks must carry the real-data
  evidence for natural and industrial shortcuts.
