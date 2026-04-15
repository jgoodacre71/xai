# 0018: Explanation Drift Shortcut Report

## Status
Complete

## Owner
Codex thread

## Why
The remaining primary pillar in the spec is robustness and explanation drift.
The suite needs a working demo where prediction movement and explanation
movement are tracked separately under benign perturbations.

## Source of truth
- REPO_SPEC.md
- AGENTS.md
- docs/ARCHITECTURE.md
- docs/XAI_CONTRACT.md
- docs/DEMO_CATALOGUE.md

## Scope
Use the synthetic industrial shortcut setup to create perturbations and compare
score shift with explanation-region shift for the shortcut classifier and the
shape intervention classifier.

## Out of scope
- Deep model robustness benchmarks.
- Real image corruptions beyond simple deterministic perturbations.
- Notebook authoring.

## Deliverables
- Perturbation/drift helpers.
- Static explanation drift report and CLI command.
- Demo card/local index integration.
- Tests and docs.

## Constraints
- Generated perturbation images remain ignored.
- The report must distinguish prediction shift from explanation shift.
- Keep reusable logic in `src/`.
- UK English in docs and report copy.

## Proposed file changes
- `src/xai_demo_suite/explain/drift.py`
- `src/xai_demo_suite/reports/explanation_drift.py`
- `src/xai_demo_suite/cli/demo.py`
- tests and docs

## Validation plan
1. `./.venv/bin/ruff check .`
2. `./.venv/bin/mypy src`
3. `./.venv/bin/pytest -q`
4. `./.venv/bin/xai-demo-report explanation-drift`

## Risks
- Synthetic perturbations are didactic; report copy must not claim production
  robustness coverage.

## Decision log
### 2026-04-15
- Decision: Use deterministic shortcut-demo perturbations first.
- Reason: This gives an immediately runnable explanation-drift demo before
  introducing real robustness datasets.
- Follow-up: Extend to PatchCore and real corruptions later.

## Progress log
### 2026-04-15
- Completed: Opened task after Demo 04 was committed.
- Verification: Previous task checks were clean.
- Remaining: Implement, test, generate, and commit.

### 2026-04-15
- Completed: Added drift measurement helpers, deterministic perturbations,
  hybrid shortcut/shape evidence tracking, `explanation-drift` CLI command,
  static report, demo card/index integration, tests, and docs.
- Verification: `./.venv/bin/ruff check .`; `./.venv/bin/mypy src`;
  `./.venv/bin/pytest -q`; regenerated `patchcore-bottle`, `patchcore-limits`,
  `shortcut-industrial`, `patchcore-wrong-normal`, and `explanation-drift`
  reports.
- Remaining: None for this task. The local index now links Demo 02, Demo 03,
  Demo 04, Demo 05, and Demo 08.
