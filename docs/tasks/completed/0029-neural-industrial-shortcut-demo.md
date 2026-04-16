# 0029: Neural Industrial Shortcut Demo

## Status
Complete

## Owner
Codex thread

## Why
Demo 02 is still the old deterministic shortcut classifier from the first
synthetic pass. The spec and `docs/TODO.md` call for a stronger industrial
shortcut story: either a real dataset path or a neural industrial shortcut
baseline with real explanation methods.

## Source of truth
- REPO_SPEC.md
- AGENTS.md
- docs/TODO.md
- docs/DEMO_STATUS.md
- docs/tasks/completed/0016-industrial-shortcut-synthetic-report.md
- docs/tasks/completed/0028-real-waterbirds-demo01.md

## Scope
- Upgrade the synthetic industrial dataset so it can support learned models.
- Add a neural shortcut baseline for Demo 02.
- Add a shortcut-mitigation intervention model for comparison.
- Add Grad-CAM, Integrated Gradients, and perturbation diagnostics.
- Update the report, CLI options, tests, and checked-in repo memory.

## Out of scope
- Sourcing NEU or GC10-DET in this task.
- Reworking Demo 08 in the same change.

## Deliverables
- Stronger synthetic industrial dataset variants
- Neural Demo 02 model path
- Upgraded Demo 02 report and demo card
- Docs and task-memory updates
- Tests

## Constraints
- Keep the repo runnable on a fresh clone.
- Use UK English.
- Raw external datasets remain uncommitted.

## Validation plan
1. `./.venv/bin/ruff check src tests`
2. `./.venv/bin/mypy src`
3. `./.venv/bin/pytest -q`
4. `./.venv/bin/xai-demo-report shortcut-industrial`
5. `./.venv/bin/xai-demo-report verify`

## Risks
- The synthetic data may still be too simple if variation is not widened enough.
- CPU-only training time needs to stay acceptable for a local demo.

## Decision log
### 2026-04-16
- Decision: Upgrade Demo 02 with a neural synthetic baseline rather than pause
  for an external industrial classification dataset.
- Reason: It closes a major spec gap immediately, keeps the suite runnable, and
  gives Demo 08 a stronger future base model to build on.

## Progress log
### 2026-04-16
- Completed: Audited the spec, TODOs, and existing Demo 02 implementation.
- Completed: Upgraded the synthetic industrial dataset to include a broader
  deterministic set of training and challenge-case variations.
- Completed: Added a learned convolutional Demo 02 baseline and a
  stamp-invariant intervention trained on stamp-randomised and stamp-masked
  augmentations.
- Completed: Replaced the old deterministic report with Grad-CAM, Integrated
  Gradients, and known-region shortcut diagnostics over the stamp and part.
- Validation:
  - `./.venv/bin/ruff check src tests`
  - `./.venv/bin/mypy src`
  - `./.venv/bin/pytest tests/unit/test_synthetic_fixtures.py tests/unit/test_shortcut_classification.py tests/unit/test_shortcut_industrial_report.py tests/unit/test_explanation_drift_report.py -q`
  - `./.venv/bin/xai-demo-report shortcut-industrial`
  - `./.venv/bin/xai-demo-report verify`
