# 0014: PatchCore Report Overview

## Status
Complete

## Owner
Codex thread

## Why
The PatchCore report contains the required evidence panels, provenance views,
counterfactual previews, and mask checks. It should also open with a concise
overview so a viewer can immediately see which examples were selected, whether
the top patch intersected the mask, and how the counterfactual score changed.

## Source of truth
- REPO_SPEC.md
- AGENTS.md
- docs/XAI_CONTRACT.md
- docs/PATCHCORE_NOTES.md

## Scope
Add an at-a-glance summary section to the static PatchCore bottle report.

## Out of scope
- Full benchmark evaluation.
- New model features.
- New notebooks.

## Deliverables
- Report summary table and aggregate bullets.
- Tests covering the summary.
- Regenerated local report.
- Documentation update if behaviour changes.

## Constraints
- Keep detailed per-example panels intact.
- Do not overclaim coarse top-patch mask overlap as benchmark performance.
- Preserve UK English.

## Proposed file changes
- `src/xai_demo_suite/reports/patchcore_bottle.py`
- `tests/unit/test_patchcore_report.py`
- `docs/PATCHCORE_NOTES.md`

## Validation plan
1. `./.venv/bin/ruff check .`
2. `./.venv/bin/mypy src`
3. `./.venv/bin/pytest -q`
4. `./.venv/bin/xai-demo-report patchcore-bottle --max-examples 3 --no-cache`

## Risks
- Summary metrics can be mistaken for benchmark metrics unless labelled
  carefully.

## Decision log
### 2026-04-15
- Decision: Summarise selected-example metrics only.
- Reason: This is useful for the demo while staying honest about scope.
- Follow-up: Add benchmark metrics later after final PatchCore feature-map work.

## Progress log
### 2026-04-15
- Completed: Opened the task after adding notebook release artefact.
- Verification: Previous task checks were clean.
- Remaining: Implement, test, regenerate, and commit.

### 2026-04-15
- Completed: Added selected-example overview table and aggregate diagnostics to
  the PatchCore report.
- Verification: `./.venv/bin/ruff check .`; `./.venv/bin/mypy src`;
  `./.venv/bin/pytest -q`; `./.venv/bin/xai-demo-report patchcore-bottle --max-examples 3 --no-cache`.
- Remaining: None for this task.
