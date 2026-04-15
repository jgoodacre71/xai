# 0021: Dedicated PatchCore Severity and Logic Reports

## Status
Complete

## Owner
Codex thread

## Why
The spec lists Demo 06 and Demo 07 as separate PatchCore limitation demos. The
current suite includes these themes inside Demo 05, but there are no dedicated
reports, cards, or CLI commands for severity calibration and logical anomalies.

## Source of truth
- REPO_SPEC.md
- AGENTS.md
- docs/ARCHITECTURE.md
- docs/XAI_CONTRACT.md
- docs/DEMO_CATALOGUE.md
- docs/DEMO_STATUS.md
- docs/PATCHCORE_NOTES.md

## Scope
Add dedicated Demo 06 and Demo 07 static reports using synthetic slot-board
cases: a controlled scratch severity sweep for novelty-versus-severity mismatch,
and a logical component-swap case showing patch novelty without symbolic rule
understanding. Wire both into the CLI, suite runner, verification, docs, and
tests.

## Out of scope
- Sourcing MVTec LOCO AD.
- Implementing a symbolic reasoning model.
- Replacing the deterministic PatchCore-style feature path with pretrained
  multi-scale PatchCore.

## Deliverables
- Synthetic severity sweep generation.
- Demo 06 static report and demo card.
- Demo 07 static report and demo card.
- CLI and suite integration.
- Unit tests and documentation updates.

## Constraints
- Generated outputs remain uncommitted.
- Synthetic reports must run without external datasets.
- Reports must be explicit about what is metadata, what is PatchCore-style
  output, and what remains outside the model.
- Use UK English.

## Proposed file changes
- `src/xai_demo_suite/data/synthetic/slot_boards.py`
- `src/xai_demo_suite/reports/patchcore_severity.py`
- `src/xai_demo_suite/reports/patchcore_logic.py`
- `src/xai_demo_suite/cli/demo.py`
- `src/xai_demo_suite/reports/suite.py`
- tests and docs

## Validation plan
1. `./.venv/bin/ruff check .`
2. `./.venv/bin/mypy src`
3. `./.venv/bin/pytest -q`
4. `./.venv/bin/xai-demo-report patchcore-severity --no-cache`
5. `./.venv/bin/xai-demo-report patchcore-logic --no-cache`
6. `./.venv/bin/xai-demo-report suite --include-mvtec --no-cache`
7. `./.venv/bin/xai-demo-report verify`

## Risks
- Synthetic logical anomalies illustrate the concept but do not replace MVTec
  LOCO AD structural/logical comparisons.

## Decision log
### 2026-04-15
- Decision: Add synthetic dedicated reports now, then leave MVTec LOCO sourcing
  as the next data task.
- Reason: The suite needs complete demo slots and runnable local artefacts before
  depending on another large external dataset.
- Follow-up: Add LOCO sourcing and compare these synthetic claims with real
  logical anomaly examples.

## Progress log
### 2026-04-15
- Completed: Re-audited the spec, agents, demo status, and PatchCore limits
  implementation.
- Verification: Git working tree was clean before edits.
- Remaining: Implement, test, generate, verify, and commit.

### 2026-04-15
- Completed: Added the synthetic severity sweep, shared synthetic PatchCore
  report helpers, dedicated Demo 06 and Demo 07 reports/cards, CLI commands,
  suite integration, tests, and docs updates.
- Verification: `./.venv/bin/ruff check .`; `./.venv/bin/mypy src`;
  `./.venv/bin/pytest -q`; `./.venv/bin/xai-demo-report
  patchcore-severity --no-cache`; `./.venv/bin/xai-demo-report
  patchcore-logic --no-cache`; `./.venv/bin/xai-demo-report suite
  --include-mvtec --no-cache`; `./.venv/bin/xai-demo-report verify`.
- Remaining: Source MVTec LOCO AD and add a component-aware or logic-aware
  comparator later.
