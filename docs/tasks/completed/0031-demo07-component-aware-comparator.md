# 0031: Demo 07 Component-aware Comparator

## Status
Complete

## Owner
Codex thread

## Why
Demo 07 already had a real MVTec LOCO AD `juice_bottle` path, but the last
major first-wave spec gap remained open: the report still showed only what
PatchCore-style patch novelty can do, without a contrasting logic-aware or
component-aware method.

## Source of truth
- REPO_SPEC.md
- AGENTS.md
- docs/TODO.md
- docs/DEMO_STATUS.md
- docs/PATCHCORE_NOTES.md
- docs/tasks/completed/0023-real-mvtec-loco-demo-07.md

## Scope
- Add a narrow component-aware comparator for Demo 07 on the aligned
  `juice_bottle` category.
- Integrate the comparator into the real LOCO report so the report contrasts
  PatchCore patch novelty with a packaging-rule check.
- Add focused tests and update repo memory/docs.

## Out of scope
- A general symbolic reasoning model.
- Broad LOCO multi-category support.
- OCR-heavy or external-model dependency expansion.

## Deliverables
- `src/xai_demo_suite/models/component_rules.py`
- Upgraded real Demo 07 report with comparator diagnostics
- Updated docs and task memory
- Focused unit tests

## Constraints
- Keep the claim honest: the comparator is category-specific, not a universal
  anomaly model.
- Preserve the existing synthetic fallback for fresh clones without LOCO data.
- Use UK English and keep raw datasets uncommitted.

## Validation plan
1. `./.venv/bin/ruff check src/xai_demo_suite/models/component_rules.py src/xai_demo_suite/reports/patchcore_logic.py tests/unit/test_component_rules.py tests/unit/test_patchcore_logic_report.py`
2. `./.venv/bin/pytest tests/unit/test_component_rules.py tests/unit/test_patchcore_logic_report.py -q`
3. `./.venv/bin/xai-demo-report patchcore-logic`
4. `./.venv/bin/mypy src`
5. `./.venv/bin/xai-demo-report verify`

## Risks
- A fixed-layout comparator can easily be overclaimed unless it is labelled
  clearly as a narrow packaging-rule check.
- The aligned ROI must be stable enough across the prepared local category to
  produce a meaningful comparison.

## Decision log
### 2026-04-16
- Decision: Implement a front-label template comparator for `juice_bottle`
  rather than introducing a broader new model family.
- Reason: It closes the spec gap with a serious, legible, data-backed contrast
  while staying consistent with the current repo scope and local datasets.

## Progress log
### 2026-04-16
- Completed: Re-read the spec, TODOs, agents, PatchCore notes, and current Demo
  07 implementation to confirm the remaining first-wave gap.
- Completed: Added a category-specific component-template baseline that learns
  the expected `juice_bottle` front-label region from nominal training images.
- Completed: Upgraded the real Demo 07 report to include component scores,
  comparator visuals, and test-split diagnostics against `good`,
  `logical_anomalies`, and `structural_anomalies`.
- Completed: Added focused unit coverage for the comparator and updated repo
  docs/TODO state.
