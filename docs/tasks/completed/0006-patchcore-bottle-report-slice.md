# 0006-patchcore-bottle-report-slice: PatchCore bottle report slice

## Status
Complete

## Owner
Codex thread

## Why
The spec says the PatchCore hero demo must be visual and must show real
nearest-normal source evidence, not just feature-space scores. We now have local
MVTec AD bottle data, patch provenance, a ResNet feature path, and memory-bank
caching. The next step is a runnable report slice that a user can inspect.

## Source of truth
- AGENTS.md
- .agents/PLANS.md
- REPO_SPEC.md section 3.2, PatchCore Lab
- docs/ARCHITECTURE.md
- docs/XAI_CONTRACT.md
- docs/PATCHCORE_NOTES.md
- docs/DATASETS.md
- docs/tasks/completed/0003-patchcore-provenance-foundation.md
- docs/tasks/completed/0004-patchcore-feature-extractor-interface.md
- docs/tasks/completed/0005-patchcore-resnet-extractor-and-cache.md

## Scope
- Add reusable visual/report code under `src/`.
- Generate a static PatchCore bottle report from the prepared local MVTec AD
  manifest.
- Include the mandatory early views:
  - input image;
  - top anomalous patch;
  - top-k nearest normal patches;
  - full source image references via source paths and boxes;
  - per-patch distance summary.
- Add a CLI entry point for running the report.
- Add tests using synthetic images so CI does not require local MVTec data.

## Out of scope
- Full anomaly-map heatmap interpolation.
- Notebook polishing.
- Counterfactual patch replacement.
- Coreset selection.
- Pretrained-weight download.

## Deliverables
- `src/xai_demo_suite/vis/`
- `src/xai_demo_suite/reports/`
- CLI command for the bottle report
- tests for panel generation and report output
- docs update explaining how to run the local report

## Constraints
- Keep all generated reports under ignored output/artefact paths.
- Do not commit raw data, processed manifests, generated images, or reports.
- Keep source evidence real: patch crops must come from actual source image
  paths and recorded coordinates.
- Use UK English in report copy and docs.
- Keep notebooks thin; this report must be package/CLI-driven.

## Proposed File Changes
- `src/xai_demo_suite/vis/image_panels.py`
- `src/xai_demo_suite/reports/patchcore_bottle.py`
- `src/xai_demo_suite/cli/demo.py`
- `pyproject.toml`
- `docs/PATCHCORE_NOTES.md`
- `README.md`
- `tests/unit/test_patchcore_report.py`

## Validation Plan
1. `./.venv/bin/ruff check .`
2. `./.venv/bin/mypy src`
3. `./.venv/bin/pytest -q`
4. `./.venv/bin/xai-demo-report patchcore-bottle --max-train 2 --patch-size 128 --stride 128`

## Acceptance Criteria
- The report command creates an HTML file and image assets locally.
- The HTML includes the selected input image, top anomalous patch, nearest
  nominal patches, and a distance table.
- Tests prove source patch crops use recorded coordinates.
- The report can run on the local MVTec AD bottle manifest.
- Git remains clean apart from committed source/docs/test changes.

## Risks
- The current ResNet path uses random weights, so the report demonstrates
  provenance mechanics rather than final anomaly quality.
- Generated visual design is intentionally minimal until the pipeline is stable.

## Decision Log
### 2026-04-15
- Decision: create a static HTML report before notebooks.
- Reason: AGENTS.md and ARCHITECTURE.md require reusable package logic first.
- Follow-up: notebooks can later call this report/data path rather than
  reimplementing it.

## Progress Log
### 2026-04-15
- Completed: static report builder, visual crop/box helpers, report CLI, tests,
  docs, and local MVTec AD bottle report generation.
- Verification: `./.venv/bin/ruff check .`, `./.venv/bin/mypy src`,
  `./.venv/bin/pytest -q`, and
  `./.venv/bin/xai-demo-report patchcore-bottle --max-train 2 --patch-size 128 --stride 128`
  passed.
- Remaining: add anomaly-map interpolation and counterfactual patch replacement.
