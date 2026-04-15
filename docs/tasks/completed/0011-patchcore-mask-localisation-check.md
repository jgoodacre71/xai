# 0011: PatchCore Mask Localisation Check

## Status
Complete

## Owner
Codex thread

## Why
The PatchCore bottle report now shows multiple examples with anomaly overlays,
nearest-normal provenance, and counterfactual previews. To make the hero demo
more rigorous, the report should also verify the explanation against available
MVTec ground-truth masks. This supports the repo rule that explanation images
used to support a claim should have a corresponding counter-test or
verification path.

## Source of truth
- REPO_SPEC.md
- AGENTS.md
- docs/ARCHITECTURE.md
- docs/XAI_CONTRACT.md
- docs/PATCHCORE_NOTES.md

## Scope
Add reusable binary-mask localisation metrics and visual overlays, then use
them in the PatchCore bottle report for anomalous examples with masks.

## Out of scope
- Pixel-level AUROC or benchmark-grade evaluation.
- Training a new model.
- Changing the current random-weight ResNet feature limitation.
- Notebook authoring.

## Deliverables
- Reusable localisation metric code under `src/xai_demo_suite/`.
- Ground-truth mask overlay image generation.
- PatchCore report section showing top-patch/mask overlap metrics.
- Unit tests for mask metrics, overlay writing, and report integration.
- Documentation updates.

## Constraints
- Preserve provenance fields: source image ids and patch coordinates.
- Do not commit raw data, masks, generated reports, or cached artefacts.
- Keep behaviour valid when a manifest record has no mask.
- Use UK English in docs and user-facing copy.

## Proposed file changes
- `src/xai_demo_suite/evaluate/localisation.py` — mask loading and patch/mask
  overlap metrics.
- `src/xai_demo_suite/vis/image_panels.py` — ground-truth mask overlay helper.
- `src/xai_demo_suite/reports/patchcore_bottle.py` — render mask check per
  example when masks are available.
- `tests/unit/test_patchcore_report.py` — cover metrics, overlay, and report
  output.
- `docs/PATCHCORE_NOTES.md` and `README.md` — document the verification panel.

## Validation plan
1. `./.venv/bin/ruff check .`
2. `./.venv/bin/mypy src`
3. `./.venv/bin/pytest -q`
4. `./.venv/bin/xai-demo-report patchcore-bottle --max-train 2 --patch-size 128 --stride 128 --max-examples 3 --no-cache`

## Risks
- A coarse top patch can overlap only part of a fine mask, so the copy must not
  overclaim localisation quality.
- Some future datasets may omit masks, so report rendering must degrade
  gracefully.

## Decision log
### 2026-04-15
- Decision: Add simple top-patch/mask overlap metrics rather than full
  benchmark metrics.
- Reason: The report needs a visible verification path now, while a full
  benchmark evaluation deserves its own task.
- Follow-up: Add pixel-level and image-level metrics once the feature extractor
  is upgraded.

## Progress log
### 2026-04-15
- Completed: Confirmed local MVTec AD bottle manifest contains mask paths for
  anomalous examples.
- Verification: `git status --short` was clean before edits.
- Remaining: Implement, test, regenerate, and commit.

### 2026-04-15
- Completed: Added reusable binary-mask loading and top-patch overlap metrics,
  mask overlay rendering, and a PatchCore report localisation-check section.
- Verification: `./.venv/bin/ruff check .`; `./.venv/bin/mypy src`;
  `./.venv/bin/pytest -q`; `./.venv/bin/xai-demo-report patchcore-bottle --max-train 2 --patch-size 128 --stride 128 --max-examples 3 --no-cache`.
- Remaining: None for this task. The generated report honestly shows that the
  current random-weight feature path does not overlap the masks for the first
  three selected examples, so model-quality improvement is the next task.
