# 0038: Local Demo Hub Polish

## Status
Completed

## Owner
Codex thread

## Why
The spec-level feature work is complete, so the next pass is product polish.
The local `outputs/index.html` is the first thing a user sees when exploring the
generated suite, so it should make the prepared-data state and demo ordering
obvious instead of behaving like a plain card dump.

## Source of truth
- REPO_SPEC.md
- AGENTS.md
- docs/DEMO_STATUS.md
- docs/DEMO_CATALOGUE.md
- docs/TODO.md

## Scope
- Improve the generated local demo hub with prepared-data summary badges and
  story-order demo sorting.
- Regenerate the strongest available local suite on this machine.
- Verify the regenerated outputs and update task memory/docs as needed.

## Out of scope
- New demo logic.
- New datasets.
- Remote deployment or GitHub configuration.

## Deliverables
- Updated `src/xai_demo_suite/reports/cards.py`
- Focused tests for the local index
- Regenerated local outputs
- Updated docs and completed task memory

## Constraints
- Keep the index static and file-based.
- Reflect only datasets that are actually prepared locally.
- Preserve the current demo-card/report workflow.

## Validation plan
1. `./.venv/bin/ruff check src tests`
2. `./.venv/bin/mypy src`
3. `./.venv/bin/pytest tests/unit/test_demo_cards.py tests/unit/test_suite_reports.py -q`
4. `./.venv/bin/xai-demo-report suite --include-mvtec --mvtec-feature-extractor feature_map_resnet18_pretrained --mvtec-max-train 20 --mvtec-max-examples 3 --mvtec-coreset-size 512 --mvtec-input-size 224`
5. `./.venv/bin/xai-demo-report verify`

## Risks
- The strongest local suite path can take materially longer than the fast smoke
  checks because it rebuilds the pretrained PatchCore bottle report.

## Decision log
### 2026-04-16
- Decision: Polish the local hub rather than add more features.
- Reason: The suite is feature-complete against the checked-in TODOs, so the
  highest-value work is now improving the demo product users actually open.

## Progress log
### 2026-04-16
- Completed: Added prepared-data summary badges and story-order sorting to the
  generated local demo hub.
- Completed: Made suite builds deterministic for temp/test output roots so
  focused tests do not silently pick up prepared local datasets from the
  working tree.
- Completed: Rebuilt the default local suite and re-verified the generated
  outputs.
