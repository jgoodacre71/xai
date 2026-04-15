# 0025: Demo-Ready Suite Runner

## Status
Complete

## Owner
Codex thread

## Why
The repo now has a serious pretrained PatchCore path, but the suite-level
command still defaults the MVTec hero demo to deterministic local features.
For a fit-for-purpose demo suite, there should be a single clear command that
builds the full local presentation set with the stronger model path, and the
local index should be good enough to use as the demo entry point.

## Source of truth
- REPO_SPEC.md
- AGENTS.md
- docs/ARCHITECTURE.md
- docs/XAI_CONTRACT.md
- docs/DEMO_STATUS.md
- docs/PATCHCORE_NOTES.md
- docs/tasks/completed/0024-pretrained-feature-map-patchcore.md

## Scope
- Add suite-level MVTec model configuration flags.
- Make the suite builder pass those settings into Demo 03.
- Improve the generated local demo index presentation while keeping it static
  and dependency-free.
- Update docs and tests.

## Out of scope
- New datasets.
- Official benchmark metrics.
- Remote/GitHub setup.
- Replacing the synthetic shortcut demos with real datasets.

## Deliverables
- CLI support for full-suite pretrained PatchCore runs.
- Presenter-friendly `outputs/index.html` generation.
- Tests for suite configuration and index content.
- Docs/status updates.

## Constraints
- Fresh-clone synthetic suite must still work without Torch or external data.
- Pretrained weights must remain explicit.
- Generated outputs and data remain uncommitted.
- Use UK English.

## Proposed file changes
- `src/xai_demo_suite/reports/suite.py`
- `src/xai_demo_suite/cli/demo.py`
- `src/xai_demo_suite/reports/cards.py`
- tests
- README/docs

## Validation plan
1. `./.venv/bin/ruff check .`
2. `./.venv/bin/mypy src`
3. `./.venv/bin/pytest -q`
4. `./.venv/bin/xai-demo-report suite --include-mvtec --mvtec-feature-extractor feature_map_resnet18_pretrained --mvtec-max-train 20 --mvtec-max-examples 3 --mvtec-coreset-size 512`
5. `./.venv/bin/xai-demo-report verify`

## Risks
- Adding suite flags can make the CLI feel noisy if the defaults are not clear.
- The local index should improve presentation without becoming a separate web
  app.

## Decision log
### 2026-04-15
- Decision: Keep the synthetic suite default lightweight, and add explicit
  MVTec suite flags for the local full demo.
- Reason: Fresh clones must remain runnable while this machine can still produce
  the strongest local presentation artefacts.

## Progress log
### 2026-04-15
- Completed: Rechecked repo status, spec, demo status, suite runner, local index
  writer, and suite tests after commit `c33b322`.
- Completed: Added suite-level MVTec model flags, passed them through to Demo
  03, upgraded the local index into a static presentation page, and added tests
  for suite model configuration and index content.
- Verification:
  `./.venv/bin/ruff check .`;
  `./.venv/bin/mypy src`;
  `./.venv/bin/pytest -q`;
  `./.venv/bin/xai-demo-report suite --include-mvtec --mvtec-feature-extractor feature_map_resnet18_pretrained --mvtec-max-train 20 --mvtec-max-examples 3 --mvtec-coreset-size 512 --mvtec-input-size 224`;
  `./.venv/bin/xai-demo-report verify`.
- Remaining: Add richer benchmark metrics and replace synthetic shortcut proxies
  with real dataset adapters in later tasks.
