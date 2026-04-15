# 0016: Industrial Shortcut Synthetic Report

## Status
Complete

## Owner
Codex thread

## Why
The suite now has PatchCore hero and limits demos. The spec also requires a
Shortcut Lab showing a model that appears to work but relies on a nuisance
feature. A synthetic industrial shortcut demo can run locally without external
data and will make the failure legible through perturbation evidence.

## Source of truth
- REPO_SPEC.md
- AGENTS.md
- docs/ARCHITECTURE.md
- docs/XAI_CONTRACT.md
- docs/DEMO_CATALOGUE.md

## Scope
Generate synthetic industrial classification images with a corner stamp
shortcut, compare a stamp-based classifier with a shape-based intervention, and
write a static report with metrics, evidence, and counterfactual stamp swaps.

## Out of scope
- Deep neural network training.
- Waterbirds sourcing.
- Grad-CAM implementation.
- External datasets.

## Deliverables
- Synthetic shortcut generator.
- Lightweight classifier/evidence code.
- Static shortcut report and CLI command.
- Demo card/local index integration.
- Tests and docs.

## Constraints
- Generated images must stay under ignored output paths.
- Be explicit that the classifier is intentionally simple and didactic.
- Keep reusable logic in `src/`.
- UK English in docs and report copy.

## Proposed file changes
- `src/xai_demo_suite/data/synthetic/industrial_shortcuts.py`
- `src/xai_demo_suite/models/classification/shortcut.py`
- `src/xai_demo_suite/reports/shortcut_industrial.py`
- `src/xai_demo_suite/cli/demo.py`
- tests and docs

## Validation plan
1. `./.venv/bin/ruff check .`
2. `./.venv/bin/mypy src`
3. `./.venv/bin/pytest -q`
4. `./.venv/bin/xai-demo-report shortcut-industrial`

## Risks
- A hand-built classifier is less realistic than a neural model, so the report
  must frame it as a didactic shortcut lab.
- The visual evidence should verify the claim by perturbing the stamp region.

## Decision log
### 2026-04-15
- Decision: Start with a deterministic synthetic shortcut demo.
- Reason: It gives a complete working Shortcut Lab without adding external data
  or model-training dependencies.
- Follow-up: Add Waterbirds or a real industrial dataset later.

## Progress log
### 2026-04-15
- Completed: Opened task after completing PatchCore Limits Lab.
- Verification: Git status was clean; no remote is configured.
- Remaining: Implement, test, generate, and commit.

### 2026-04-15
- Completed: Added synthetic industrial shortcut data, deterministic shortcut
  and shape classifiers, counterfactual stamp helpers, static report, CLI
  command, tests, and docs.
- Verification: `./.venv/bin/ruff check .`; `./.venv/bin/mypy src`;
  `./.venv/bin/pytest -q`; `./.venv/bin/xai-demo-report patchcore-bottle --max-examples 3 --no-cache`;
  `./.venv/bin/xai-demo-report patchcore-limits --no-cache`;
  `./.venv/bin/xai-demo-report shortcut-industrial`.
- Remaining: None for this task. The generated local index now links Demo 02,
  Demo 03, and Demo 05.
