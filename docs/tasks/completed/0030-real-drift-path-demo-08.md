# 0030: Real Drift Path for Demo 08

## Status
Complete

## Owner
Codex thread

## Why
Demo 08 is still the original synthetic drift report built around a
deterministic hybrid classifier and hand-picked evidence boxes. The spec now
calls for a stronger robustness and explanation-drift story with real
corruption or acquisition-shift paths, and ideally one classifier plus one
anomaly detector.

## Source of truth
- REPO_SPEC.md
- AGENTS.md
- docs/TODO.md
- docs/DEMO_STATUS.md
- docs/tasks/completed/0018-explanation-drift-shortcut-report.md
- docs/tasks/completed/0029-neural-industrial-shortcut-demo.md

## Scope
- Replace the old Demo 08 classifier path with the learned industrial shortcut
  models from Demo 02.
- Add real corruption-like perturbations such as blur, contrast, compression,
  and acquisition-style lighting/fading shifts.
- Add an anomaly-detector drift section when local MVTec bottle data is
  prepared.
- Update tests, docs, and checked-in task memory.

## Out of scope
- Full MVTec AD 2 support.
- A logic-aware Demo 07 comparator in the same task.

## Deliverables
- Stronger perturbation helpers
- Upgraded Demo 08 report
- Optional MVTec anomaly-drift section
- Docs and task-memory updates
- Tests

## Constraints
- The report must still run without external datasets.
- Use UK English.
- Raw datasets remain uncommitted.

## Validation plan
1. `./.venv/bin/ruff check src tests`
2. `./.venv/bin/mypy src`
3. `./.venv/bin/pytest tests/unit/test_explanation_drift_report.py -q`
4. `./.venv/bin/xai-demo-report explanation-drift`
5. `./.venv/bin/xai-demo-report verify`

## Risks
- Adding an optional anomaly section can make the report logic more complex.
- The drift story needs to remain legible rather than turning into a large table
  of weak metrics.

## Decision log
### 2026-04-16
- Decision: Make the classifier section always available and gate the anomaly
  section on prepared MVTec bottle data.
- Reason: That preserves portability while still giving prepared machines the
  fuller spec story.

## Progress log
### 2026-04-16
- Completed: Audited the current Demo 08 implementation, spec language, and
  current drift helpers.
- Completed: Replaced the old deterministic box-tracking classifier path with
  the learned industrial shortcut models from Demo 02.
- Completed: Added stronger perturbations including lighting, blur, contrast,
  JPEG compression, and shadow-band acquisition shifts.
- Completed: Added an optional local PatchCore anomaly-drift section using
  prepared MVTec bottle data, image-level AUC, top-patch movement, and
  mask-coverage checks.
- Validation:
  - `./.venv/bin/ruff check src tests`
  - `./.venv/bin/mypy src`
  - `./.venv/bin/pytest tests/unit/test_explanation_drift_report.py -q`
  - `./.venv/bin/xai-demo-report explanation-drift`
  - `./.venv/bin/xai-demo-report verify`
