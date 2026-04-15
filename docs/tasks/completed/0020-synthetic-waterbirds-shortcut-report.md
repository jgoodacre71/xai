# 0020: Synthetic Waterbirds Shortcut Report

## Status
Complete

## Owner
Codex thread

## Why
The spec names Demo 01 as a Waterbirds or equivalent spurious-correlation
classification story. The repository already has an industrial shortcut report,
but it still lacks the canonical habitat/background shortcut narrative that
anchors the shortcut-learning pillar.

## Source of truth
- REPO_SPEC.md
- AGENTS.md
- docs/ARCHITECTURE.md
- docs/XAI_CONTRACT.md
- docs/DEMO_CATALOGUE.md
- docs/DEMO_STATUS.md

## Scope
Add a deterministic synthetic Waterbirds-style demo with bird foreground classes,
land/water backgrounds, shortcut and shape-based classifiers, worst-group style
metrics, evidence boxes, counterfactual background swaps, a static report, a demo
card, CLI wiring, suite integration, tests, and docs updates.

## Out of scope
- Downloading or committing the real Waterbirds dataset.
- Training a neural ResNet or ViT.
- Adding Grad-CAM or Integrated Gradients.

## Deliverables
- Synthetic habitat shortcut dataset generator.
- Classification helpers for background and bird-shape decision rules.
- Demo 01 static report and demo card.
- CLI and suite integration.
- Unit tests and documentation updates.

## Constraints
- No raw external data is committed.
- The report must run in a fresh clone without network access.
- The narrative must be honest that this is a synthetic Waterbirds-style proxy,
  not the real dataset.
- Use UK English.

## Proposed file changes
- `src/xai_demo_suite/data/synthetic/waterbirds.py`
- `src/xai_demo_suite/models/classification/habitat_shortcut.py`
- `src/xai_demo_suite/reports/waterbirds_shortcut.py`
- `src/xai_demo_suite/cli/demo.py`
- `src/xai_demo_suite/reports/suite.py`
- tests and docs

## Validation plan
1. `./.venv/bin/ruff check .`
2. `./.venv/bin/mypy src`
3. `./.venv/bin/pytest -q`
4. `./.venv/bin/xai-demo-report waterbirds-shortcut`
5. `./.venv/bin/xai-demo-report suite --include-mvtec --no-cache`
6. `./.venv/bin/xai-demo-report verify`

## Risks
- A deterministic proxy can illustrate the failure mode but cannot replace the
  real Waterbirds benchmark.

## Decision log
### 2026-04-15
- Decision: Implement a synthetic Waterbirds-style proxy before real dataset
  sourcing.
- Reason: The suite needs a runnable Demo 01 now, and generated synthetic data
  respects the no-raw-data policy.
- Follow-up: Add a real Waterbirds or equivalent dataset adapter later.

## Progress log
### 2026-04-15
- Completed: Audited spec, agents, catalogue, status, and existing shortcut
  code.
- Verification: Latest committed suite runner verified 5 cards and 36 paths.
- Remaining: Implement, test, generate, verify, and commit.

### 2026-04-15
- Completed: Added synthetic habitat-bird data generation, deterministic
  habitat and bird-shape classifiers, Demo 01 report/card, CLI command, suite
  integration, tests, and docs/status updates.
- Verification: `./.venv/bin/ruff check .`; `./.venv/bin/mypy src`;
  `./.venv/bin/pytest -q`; `./.venv/bin/xai-demo-report
  waterbirds-shortcut`; `./.venv/bin/xai-demo-report suite --include-mvtec
  --no-cache`; `./.venv/bin/xai-demo-report verify`.
- Remaining: Source a real Waterbirds or equivalent dataset later.
