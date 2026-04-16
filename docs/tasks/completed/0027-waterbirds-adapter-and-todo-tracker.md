# 0027: Waterbirds Adapter and Todo Tracker

## Status
Complete

## Owner
Codex thread

## Why
The spec now has a durable summary of what remains missing, but the highest
priority open gap is still real Demo 01 data. The repo needs a checked-in todo
tracker plus a real Waterbirds dataset adapter/workflow so the synthetic proxy
is no longer the only dataset path for the shortcut pillar.

## Source of truth
- REPO_SPEC.md
- AGENTS.md
- docs/DATASETS.md
- docs/DEMO_STATUS.md
- docs/tasks/completed/0025-demo-ready-suite-runner.md
- docs/tasks/completed/0026-patchcore-benchmark-panel.md

## Scope
- Add a checked-in TODO tracker for remaining spec gaps.
- Add a Waterbirds dataset entry to the local data registry.
- Implement `xai-demo-data` support for Waterbirds fetch/prepare.
- Build a canonical processed manifest for Waterbirds.
- Add tests and docs.

## Out of scope
- Training the final real Demo 01 classifier.
- Grad-CAM / Integrated Gradients implementation.
- MetaShift, Spawrious, VisA, or MVTec AD 2 support in this task.

## Deliverables
- `docs/TODO.md`
- `data/data_registry.yaml`
- Waterbirds downloader and manifest builder
- CLI integration and tests
- Docs updates

## Constraints
- Raw datasets must remain uncommitted.
- The Waterbirds tarball source and usage notes must be documented conservatively.
- Use UK English.

## Validation plan
1. `./.venv/bin/ruff check .`
2. `./.venv/bin/mypy src`
3. `./.venv/bin/pytest -q`
4. `./.venv/bin/xai-demo-data list`
5. `./.venv/bin/xai-demo-data fetch waterbirds --category waterbird_complete95_forest2water2 --dry-run`

## Risks
- Waterbirds distribution metadata is less clean than the MVTec family.
- The initial adapter may need to support multiple metadata column variants.

## Decision log
### 2026-04-16
- Decision: Implement the real Waterbirds dataset path before adding more
  synthetic or optional benchmark extras.
- Reason: It is the largest remaining first-wave spec gap outside the PatchCore
  pillar.

## Progress log
### 2026-04-16
- Completed: Audited the spec, current shortcut demos, data CLI patterns, and
  dataset docs.
- Completed: Added `docs/TODO.md`, `data/data_registry.yaml`, Waterbirds
  fetch/prepare support, manifest generation, CLI integration, tests, and docs.
- Validation:
  - `./.venv/bin/ruff check src tests`
  - `./.venv/bin/mypy src`
  - `./.venv/bin/pytest tests/unit/test_waterbirds_downloader.py tests/unit/test_waterbirds_manifest.py tests/unit/test_waterbirds_shortcut_report.py -q`
  - `./.venv/bin/xai-demo-data list`
  - `./.venv/bin/xai-demo-data fetch waterbirds --category waterbird_complete95_forest2water2 --dry-run`
