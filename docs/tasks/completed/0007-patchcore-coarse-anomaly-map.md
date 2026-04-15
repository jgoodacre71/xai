# 0007-patchcore-coarse-anomaly-map: coarse anomaly-map overlay

## Status
Complete

## Owner
Codex thread

## Why
REPO_SPEC.md section 3.2 lists an anomaly map as a mandatory PatchCore hero
view. The current bottle report shows the top scored patch and nearest-normal
provenance, but not a whole-image score map. This task adds a coarse patch-score
overlay while preserving the explicit note that this is not yet full PatchCore
heatmap interpolation.

## Source of truth
- AGENTS.md
- .agents/PLANS.md
- REPO_SPEC.md section 3.2
- docs/PATCHCORE_NOTES.md
- docs/XAI_CONTRACT.md
- docs/tasks/completed/0006-patchcore-bottle-report-slice.md

## Scope
- Convert patch scores into a normalised coarse anomaly map.
- Save a visible overlay image for the report.
- Add the overlay to the static bottle report.
- Add tests for normalisation and overlay generation.
- Update docs to record the current limitation.

## Out of Scope
- Smooth full-resolution anomaly-map interpolation.
- Pixel-level AUROC evaluation.
- Counterfactual patch replacement.
- Notebook work.

## Deliverables
- visual helper for patch-score overlays
- report update showing the anomaly map
- tests for the helper and report output
- docs update

## Constraints
- Keep generated report assets ignored by git.
- Do not hide that this is a coarse patch-score map.
- Keep source evidence and patch coordinates intact.
- Preserve UK English in docs and report text.

## Proposed File Changes
- `src/xai_demo_suite/vis/image_panels.py`
- `src/xai_demo_suite/reports/patchcore_bottle.py`
- `tests/unit/test_patchcore_report.py`
- `docs/PATCHCORE_NOTES.md`

## Validation Plan
1. `./.venv/bin/ruff check .`
2. `./.venv/bin/mypy src`
3. `./.venv/bin/pytest -q`
4. `./.venv/bin/xai-demo-report patchcore-bottle --max-train 2 --patch-size 128 --stride 128 --no-cache`

## Acceptance Criteria
- The report writes an anomaly-map overlay asset.
- The report HTML displays the overlay and labels it as coarse.
- Tests prove patch-score normalisation is deterministic.
- Tests prove the overlay image is written.
- Full checks and the local report smoke command pass.

## Risks
- Random-weight ResNet features make the map visually demonstrative rather than
  final-quality anomaly detection.
- Overlapping patch scores require an averaging rule that may change when full
  PatchCore interpolation is implemented.

## Decision Log
### 2026-04-15
- Decision: add a coarse map now, not a polished full-resolution anomaly map.
- Reason: the spec requires an anomaly-map view, and a coarse map is enough to
  exercise the report and scoring path before the final PatchCore implementation.
- Follow-up: replace or complement this with interpolated feature-map scores in
  a later task.

## Progress Log
### 2026-04-15
- Completed: task created after repo audit and lint fix; coarse anomaly-map
  overlay helper implemented; report updated; tests and docs updated.
- Verification: `./.venv/bin/ruff check .`, `./.venv/bin/mypy src`,
  `./.venv/bin/pytest -q`, and
  `./.venv/bin/xai-demo-report patchcore-bottle --max-train 2 --patch-size 128 --stride 128 --no-cache`
  passed.
- Remaining: full anomaly-map interpolation and counterfactual patch replacement
  are still follow-up tasks.
