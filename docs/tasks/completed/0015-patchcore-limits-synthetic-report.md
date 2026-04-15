# 0015: PatchCore Limits Synthetic Report

## Status
Complete

## Owner
Codex thread

## Why
The repository now has a strong PatchCore hero demo, but the spec also requires
a PatchCore Limits Lab that makes count, severity, and semantic/logic limits
explicit. A deterministic synthetic slot-board report can provide a second
working demo without fetching new external data.

## Source of truth
- REPO_SPEC.md
- AGENTS.md
- docs/ARCHITECTURE.md
- docs/XAI_CONTRACT.md
- docs/PATCHCORE_NOTES.md
- docs/DEMO_CATALOGUE.md

## Scope
Generate synthetic slot-board images, run the existing PatchCore-style memory
bank over selected anomalies, and write a static HTML report showing where
PatchCore helps and where it stops being the right abstraction.

## Out of scope
- MVTec LOCO AD sourcing.
- A full logic-aware model.
- Benchmark-grade evaluation.
- Committing generated synthetic images or reports.

## Deliverables
- Synthetic slot-board generator under `src/xai_demo_suite/data/synthetic/`.
- PatchCore limits report under `src/xai_demo_suite/reports/`.
- CLI command for the report.
- Demo card and local index integration.
- Unit tests and docs updates.

## Constraints
- Generated images must remain local artefacts under ignored paths.
- Report copy must not overclaim: counts and semantic labels are dataset
  metadata, not native PatchCore outputs.
- Reuse existing PatchCore scoring and provenance machinery.
- Use UK English.

## Proposed file changes
- `src/xai_demo_suite/data/synthetic/slot_boards.py`
- `src/xai_demo_suite/data/synthetic/__init__.py`
- `src/xai_demo_suite/reports/patchcore_limits.py`
- `src/xai_demo_suite/reports/cards.py`
- `src/xai_demo_suite/reports/__init__.py`
- `src/xai_demo_suite/cli/demo.py`
- tests and docs

## Validation plan
1. `./.venv/bin/ruff check .`
2. `./.venv/bin/mypy src`
3. `./.venv/bin/pytest -q`
4. `./.venv/bin/xai-demo-report patchcore-limits --no-cache`

## Risks
- Synthetic examples can be too toy-like unless the report is explicit about
  what they demonstrate.
- PatchCore scores may partly correlate with generated count/severity metadata;
  the report should focus on what is and is not natively represented.

## Decision log
### 2026-04-15
- Decision: Use deterministic synthetic slot boards before sourcing MVTec LOCO.
- Reason: This creates an immediately runnable demo for the limits pillar and
  keeps momentum without new data dependencies.
- Follow-up: Add MVTec LOCO sourcing and comparison later.

## Progress log
### 2026-04-15
- Completed: Audited spec, agents, git state, Git remote state, and task memory.
- Verification: Working tree was clean; no Git remote is configured.
- Remaining: Implement generator, report, CLI, tests, docs, generate, and commit.

### 2026-04-15
- Completed: Added deterministic slot-board generator, PatchCore limits report,
  `patchcore-limits` CLI command, all-card local index generation, tests, and
  docs.
- Verification: `./.venv/bin/ruff check .`; `./.venv/bin/mypy src`;
  `./.venv/bin/pytest -q`; `./.venv/bin/xai-demo-report patchcore-bottle --max-examples 3 --no-cache`;
  `./.venv/bin/xai-demo-report patchcore-limits --no-cache`.
- Remaining: None for this task. The generated `outputs/index.html` now links
  the PatchCore hero demo and the PatchCore limits demo.
