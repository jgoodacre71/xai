# 0026: PatchCore Benchmark Panel

## Status
Complete

## Owner
Codex thread

## Why
Demo 03 now uses a serious pretrained feature-map PatchCore path on real MVTec
AD bottle data, but the report still centres on selected examples. To be fit
for purpose, the hero demo needs dataset-level evidence: image-level separation
between good and defective test images, defect-type score summaries, and
localisation diagnostics over masked anomalies.

## Source of truth
- REPO_SPEC.md
- AGENTS.md
- docs/XAI_CONTRACT.md
- docs/PATCHCORE_NOTES.md
- docs/DEMO_STATUS.md
- docs/tasks/completed/0024-pretrained-feature-map-patchcore.md
- docs/tasks/completed/0025-demo-ready-suite-runner.md

## Scope
- Add reusable report-side benchmark summary records.
- Score MVTec AD bottle test-good and anomalous records with the same memory
  bank/extractor used for selected examples.
- Render image-level ROC AUC, score summaries by defect type, and top-patch mask
  overlap diagnostics.
- Keep runtime bounded with a configurable benchmark cap.
- Update CLI, suite runner, tests, and docs.

## Out of scope
- Official PatchCore benchmark reproduction.
- Pixel-level AUROC over full-resolution dense maps.
- New datasets.

## Deliverables
- Demo 03 benchmark panel.
- CLI/config support for benchmark record limits.
- Tests for AUC and report rendering.
- Docs/status updates.

## Constraints
- Preserve source image and patch-coordinate provenance.
- Benchmark metrics must be labelled as local/report-level diagnostics, not
  official benchmark numbers.
- Generated outputs, data, and caches remain uncommitted.
- Use UK English.

## Validation plan
1. `./.venv/bin/ruff check .`
2. `./.venv/bin/mypy src`
3. `./.venv/bin/pytest -q`
4. `./.venv/bin/xai-demo-report patchcore-bottle --feature-extractor feature_map_resnet18_pretrained --max-train 20 --max-examples 3 --coreset-size 512 --input-size 224`
5. `./.venv/bin/xai-demo-report suite --include-mvtec --mvtec-feature-extractor feature_map_resnet18_pretrained --mvtec-max-train 20 --mvtec-max-examples 3 --mvtec-coreset-size 512 --mvtec-input-size 224`
6. `./.venv/bin/xai-demo-report verify`

## Risks
- Full test-split scoring is slower with deep features, so a cap should be
  available for local demos.
- A coarse patch grid is not a pixel-level benchmark; wording must remain
  explicit.

## Decision log
### 2026-04-16
- Decision: Add image-level ROC AUC and top-patch localisation diagnostics to
  Demo 03 before chasing additional datasets.
- Reason: This closes the largest credibility gap in the real-data hero demo
  while preserving the existing report and provenance design.

## Progress log
### 2026-04-16
- Completed: Re-read repo instructions, PatchCore skill, spec, demo status, and
  current Demo 03 implementation.
- Completed: Added Demo 03 benchmark records, image-level max-patch ROC AUC,
  defect-type score summaries, top-patch mask diagnostics, `--benchmark-limit`,
  suite pass-through, report rendering, tests, and docs.
- Verification:
  `./.venv/bin/ruff check .`;
  `./.venv/bin/mypy src`;
  `./.venv/bin/pytest -q`;
  `./.venv/bin/xai-demo-report patchcore-bottle --feature-extractor feature_map_resnet18_pretrained --max-train 20 --max-examples 3 --coreset-size 512 --input-size 224`;
  `./.venv/bin/xai-demo-report suite --include-mvtec --mvtec-feature-extractor feature_map_resnet18_pretrained --mvtec-max-train 20 --mvtec-max-examples 3 --mvtec-coreset-size 512 --mvtec-input-size 224`;
  `./.venv/bin/xai-demo-report verify`.
- Result: Local MVTec AD bottle test diagnostics scored 83 images and reported
  image-level max-patch ROC AUC of 1.000 for the current pretrained ResNet-18
  feature-map coreset run. This remains a local diagnostic, not an official
  PatchCore benchmark reproduction.
- Remaining: Add official-style pixel metrics or benchmark comparison only if a
  later task needs that level of evaluation.
