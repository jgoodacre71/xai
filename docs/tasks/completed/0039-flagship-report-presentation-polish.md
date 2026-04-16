# 0039: Flagship Report Presentation Polish

## Status
Completed

## Owner
Codex thread

## Why
The demo suite is functionally strong, but the flagship reports still read like
separate technical artefacts rather than one polished presentation product.
They need a consistent report header, presenter-facing summary, and clearer
cross-links between the four main pillars.

## Source of truth
- REPO_SPEC.md
- AGENTS.md
- docs/DEMO_CATALOGUE.md
- docs/DEMO_STATUS.md
- docs/tasks/completed/0038-local-demo-hub-polish.md

## Scope
- Add shared static report chrome for the flagship report pages.
- Improve executive summaries, live-demo framing, and related-demo links.
- Regenerate the local suite and verify the outputs.

## Out of scope
- New modelling logic.
- New datasets or training pipelines.
- Remote deployment.

## Deliverables
- Shared presentation helpers under `src/xai_demo_suite/reports/`
- Updated flagship report HTML generation
- Focused tests for the new presenter-facing copy and links
- Regenerated local outputs
- Completed task memory

## Validation plan
1. `./.venv/bin/ruff check src tests`
2. `./.venv/bin/mypy src`
3. `./.venv/bin/pytest tests/unit/test_waterbirds_shortcut_report.py tests/unit/test_patchcore_logic_report.py tests/unit/test_explanation_drift_report.py tests/unit/test_suite_reports.py -q`
4. `./.venv/bin/xai-demo-report suite --include-mvtec --mvtec-feature-extractor feature_map_resnet18_pretrained --mvtec-max-train 20 --mvtec-max-examples 3 --mvtec-coreset-size 512 --mvtec-input-size 224`
5. `./.venv/bin/xai-demo-report verify`

## Risks
- Presentation helpers can drift into too much abstraction if they start
  constraining report-specific content.
- The full suite rebuild remains the slowest step because of Demo 03.

## Decision log
### 2026-04-16
- Decision: Add one shared report chrome helper instead of hand-diverging each
  flagship page.
- Reason: The suite needs coherence across reports, not four separate styles.

## Progress log
### 2026-04-16
- Completed: Added shared static report chrome for the flagship report pages.
- Completed: Wired presenter-facing headers, demo briefs, and related-demo
  links into Demos 01, 03, 07, and 08.
- Completed: Regenerated the strongest local suite and re-verified the outputs.
