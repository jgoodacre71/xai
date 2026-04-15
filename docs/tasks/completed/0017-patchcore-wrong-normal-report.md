# 0017: PatchCore Wrong-Normal Report

## Status
Complete

## Owner
Codex thread

## Why
The spec requires Demo 04: PatchCore learns the wrong normal. The suite already
has a PatchCore hero demo, a limits demo, and a shortcut classification demo.
The next missing story is anomaly-detector shortcut learning: contaminate the
nominal set with a nuisance, rebuild the memory bank, and show how exemplar
retrieval and anomaly maps become misleading.

## Source of truth
- REPO_SPEC.md
- AGENTS.md
- docs/ARCHITECTURE.md
- docs/XAI_CONTRACT.md
- docs/PATCHCORE_NOTES.md
- docs/DEMO_CATALOGUE.md

## Scope
Generate deterministic clean and nuisance-contaminated slot boards, build two
PatchCore-style memory banks, and write a static report comparing clean-bank and
contaminated-bank explanations.

## Out of scope
- Real MVTec LOCO AD sourcing.
- Full benchmark evaluation.
- New pretrained deep features.

## Deliverables
- Synthetic nuisance-contamination generator.
- Static wrong-normal PatchCore report and CLI command.
- Demo card/local index integration.
- Tests and docs.

## Constraints
- Generated images and memory banks stay ignored.
- Preserve source image ids and patch coordinates.
- Report copy must be explicit that the nuisance is dataset-generated metadata.
- Use UK English.

## Proposed file changes
- `src/xai_demo_suite/data/synthetic/slot_boards.py`
- `src/xai_demo_suite/reports/patchcore_wrong_normal.py`
- `src/xai_demo_suite/cli/demo.py`
- tests and docs

## Validation plan
1. `./.venv/bin/ruff check .`
2. `./.venv/bin/mypy src`
3. `./.venv/bin/pytest -q`
4. `./.venv/bin/xai-demo-report patchcore-wrong-normal --no-cache`

## Risks
- The synthetic nuisance may look too simple unless the report frames it as a
  controlled failure mode.
- Patch score magnitudes are not calibrated across memory-bank variants.

## Decision log
### 2026-04-15
- Decision: Use a corner-tab nuisance in synthetic normal boards.
- Reason: It is visually clear and aligns with the spec examples of tabs,
  borders, and stamps contaminating nominal data.
- Follow-up: Add real acquisition-shift datasets later.

## Progress log
### 2026-04-15
- Completed: Audited spec, agents, task memory, git history, and current repo
  status.
- Verification: Working tree was clean before edits.
- Remaining: Implement, test, generate, and commit.

### 2026-04-15
- Completed: Added synthetic nuisance-board generation, clean and contaminated
  PatchCore memory-bank comparison, `patchcore-wrong-normal` CLI command, report
  assets, demo card/index integration, tests, and docs.
- Verification: `./.venv/bin/ruff check .`; `./.venv/bin/mypy src`;
  `./.venv/bin/pytest -q`; regenerated `patchcore-bottle`, `patchcore-limits`,
  `shortcut-industrial`, and `patchcore-wrong-normal` reports.
- Remaining: None for this task. The local index now links Demo 02, Demo 03,
  Demo 04, and Demo 05.
