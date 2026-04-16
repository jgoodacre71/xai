# 0036: Demo 01 MetaShift Extension

## Status
Complete

## Owner
Codex thread

## Why
The repo had MetaShift dataset support, but Pillar A still presented Waterbirds
as the only real natural-context shortcut path. The next quality upgrade was to
extend Demo 01 so it could show the same shortcut contract on a second natural
dataset when prepared.

## Source of truth
- REPO_SPEC.md
- AGENTS.md
- docs/TODO.md
- docs/DEMO_STATUS.md
- docs/DEMO_CATALOGUE.md
- docs/tasks/completed/0035-visa-drift-and-metashift-support.md

## Scope
- Extend Demo 01 so it can render an optional MetaShift section from the
  prepared cat-vs-dog indoor/outdoor manifest.
- Generalise the frozen-backbone binary probe so it no longer hard-codes
  Waterbirds label names.
- Add CLI support, focused tests, and docs/task-memory updates.

## Out of scope
- A standalone MetaShift report.
- ProtoPNet integration in the same pass.
- New external datasets beyond MetaShift.

## Deliverables
- Updated `src/xai_demo_suite/models/classification/waterbirds.py`
- Updated `src/xai_demo_suite/reports/waterbirds_shortcut.py`
- Updated `src/xai_demo_suite/cli/demo.py`
- Focused report tests
- Updated docs and task memory

## Constraints
- Keep the current Waterbirds path intact.
- Keep fresh-clone fallback behaviour intact when no prepared manifests exist.
- Use the same evaluation and explanation contract across both natural-context
  datasets.

## Validation plan
1. `./.venv/bin/ruff check src tests`
2. `./.venv/bin/mypy src`
3. `./.venv/bin/pytest tests/unit/test_waterbirds_shortcut_report.py -q`
4. `./.venv/bin/xai-demo-report waterbirds-shortcut --no-real-data`

## Risks
- A dataset-specific code path could have duplicated most of the Waterbirds
  report logic. The implementation should instead generalise the binary probe
  and the real-data report renderer.

## Decision log
### 2026-04-16
- Decision: Extend the existing Demo 01 report rather than create a separate
  MetaShift demo entry point.
- Reason: The spec frames Waterbirds and optional natural-context datasets as
  one coherent shortcut-learning pillar, not separate pillars.

## Progress log
### 2026-04-16
- Completed: Generalised the frozen ResNet-18 linear probe so it infers binary
  label names from the prepared manifest instead of hard-coding Waterbirds
  labels.
- Completed: Extended Demo 01 so prepared MetaShift manifests add a
  natural-context section with the same ERM-versus-group-balanced metrics,
  explanation maps, and perturbation diagnostics.
- Completed: Added CLI support, fixture-backed tests, and doc updates.
