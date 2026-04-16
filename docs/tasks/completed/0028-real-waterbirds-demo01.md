# 0028: Real Waterbirds Demo 01

## Status
Complete

## Owner
Codex thread

## Why
The Waterbirds adapter and manifest workflow were in place, but Demo 01 still
stopped at a synthetic proxy. The spec called for a real shortcut-learning
story with classifier behaviour, group metrics, and explanation evidence in one
report.

## Source of truth
- REPO_SPEC.md
- AGENTS.md
- docs/TODO.md
- docs/DATASETS.md
- docs/DEMO_STATUS.md
- docs/tasks/completed/0027-waterbirds-adapter-and-todo-tracker.md

## Scope
- Replace Demo 01's synthetic-only path with a manifest-backed real-data mode.
- Train a serious local classifier baseline on prepared Waterbirds data.
- Add explanation outputs and perturbation checks to the report.
- Keep the synthetic fallback for fresh clones without prepared data.
- Update docs, status memory, and tests.

## Out of scope
- End-to-end fine-tuning of large backbones.
- Waterbirds benchmark reproduction claims.
- Real industrial shortcut data for Demo 02.

## Deliverables
- Waterbirds manifest loader
- Frozen ResNet-18 linear probe path for Demo 01
- Real report mode with ERM versus group-balanced comparison
- Grad-CAM and Integrated Gradients overlays
- Context-masking perturbation diagnostics
- CLI/report/docs/test updates

## Constraints
- Raw datasets must remain uncommitted.
- Waterbirds usage notes must stay conservative.
- Use UK English.

## Validation plan
1. `./.venv/bin/ruff check src tests`
2. `./.venv/bin/mypy src`
3. `./.venv/bin/pytest tests/unit/test_waterbirds_downloader.py tests/unit/test_waterbirds_manifest.py tests/unit/test_waterbirds_shortcut_report.py -q`

## Risks
- Waterbirds foreground localisation is not available in the prepared manifest,
  so spatial explanation summaries must be framed as centre-versus-background
  proxies rather than true bird-mask metrics.
- A frozen-feature linear probe is strong enough for a serious local demo, but
  it should not be presented as a full benchmark reproduction.

## Decision log
### 2026-04-16
- Decision: Use frozen pretrained ResNet-18 features with a trainable linear
  head for the real Demo 01 baseline.
- Reason: This is materially stronger than a toy rule while still remaining
  local, deterministic enough for a demo repo, and fast enough to run without
  remote services.
- Decision: Compare plain ERM with inverse-group-frequency weighting.
- Reason: It keeps the intervention simple and legible, and ties directly to the
  shortcut-learning story in the spec.
- Decision: Use Grad-CAM, Integrated Gradients, and centre/background masking
  probes together.
- Reason: No single explanation view is enough on its own; the combination makes
  the shortcut evidence path clearer.

## Progress log
### 2026-04-16
- Completed: Added `src/xai_demo_suite/data/waterbirds_manifest.py` and tests.
- Completed: Added `src/xai_demo_suite/models/classification/waterbirds.py`
  with frozen ResNet-18 probes, ERM and group-balanced training modes, Grad-CAM,
  and Integrated Gradients.
- Completed: Upgraded `src/xai_demo_suite/reports/waterbirds_shortcut.py` to
  use the real manifest-backed path when available and keep the synthetic
  fallback otherwise.
- Completed: Added CLI options for Waterbirds report control and updated
  README/status/dataset docs/todo memory.
- Validation:
  - `./.venv/bin/ruff check src tests`
  - `./.venv/bin/mypy src`
  - `./.venv/bin/pytest tests/unit/test_waterbirds_downloader.py tests/unit/test_waterbirds_manifest.py tests/unit/test_waterbirds_shortcut_report.py -q`
