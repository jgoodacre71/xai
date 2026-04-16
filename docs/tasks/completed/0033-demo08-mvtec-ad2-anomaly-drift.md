# 0033: Demo 08 MVTec AD 2 Anomaly Drift

## Status
Complete

## Owner
Codex thread

## Why
The repo now had MVTec AD 2 data support, but that work was still invisible in
the demos. The next step was to turn that second-wave dataset into visible demo
value by extending an existing report rather than creating another disconnected
path.

## Source of truth
- REPO_SPEC.md
- AGENTS.md
- docs/TODO.md
- docs/DEMO_STATUS.md
- docs/tasks/completed/0030-real-drift-path-demo-08.md
- docs/tasks/completed/0032-mvtec-ad-2-support.md

## Scope
- Extend Demo 08 so it can render optional anomaly-drift sections for prepared
  MVTec AD 2 scenario manifests.
- Preserve the current bottle anomaly-drift section.
- Add fixture-backed tests and update docs/task memory.

## Out of scope
- Replacing Demo 03 with MVTec AD 2.
- Full benchmark reproduction on MVTec AD 2.
- Adding VisA, MetaShift, or Spawrious in the same task.

## Deliverables
- Extended `src/xai_demo_suite/reports/explanation_drift.py`
- CLI support for AD 2 processed-root override in `src/xai_demo_suite/cli/demo.py`
- Updated report tests
- Updated docs and task memory

## Constraints
- Keep fresh-clone behaviour intact when no local MVTec data exists.
- Keep the report explicit that these are local PatchCore-style diagnostics, not
  official benchmark claims.
- Use UK English and leave raw datasets uncommitted.

## Validation plan
1. `./.venv/bin/ruff check src tests`
2. `./.venv/bin/mypy src`
3. `./.venv/bin/pytest tests/unit/test_explanation_drift_report.py -q`
4. `./.venv/bin/xai-demo-report explanation-drift`
5. `./.venv/bin/xai-demo-report verify`

## Risks
- Adding multiple anomaly sections can bloat the report if the structure is not
  kept tight.
- Scenario-specific caches need to stay separated so local results do not bleed
  across datasets.

## Decision log
### 2026-04-16
- Decision: Extend Demo 08 rather than building a new standalone MVTec AD 2
  report first.
- Reason: That turns the second-wave dataset into visible demo value while
  preserving the four-pillar structure in the spec.

## Progress log
### 2026-04-16
- Completed: Refactored Demo 08 anomaly reporting so it can render multiple
  optional anomaly-drift sections.
- Completed: Preserved the existing MVTec AD bottle section and added optional
  MVTec AD 2 scenario sections discovered from prepared manifests.
- Completed: Added fixture-backed unit coverage for the AD 2 report path and
  updated docs/report descriptions.
