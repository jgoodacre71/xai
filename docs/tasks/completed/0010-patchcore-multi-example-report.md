# 0010: PatchCore Multi-Example Report

## Status
Complete

## Owner
Codex thread

## Why
The current PatchCore bottle report is demoable but only shows one anomalous
example. The repository spec asks for selected examples with the mandatory
PatchCore views, so the next increment should make the generated static report
show several anomalous examples from the prepared MVTec AD bottle manifest.

## Source of truth
- REPO_SPEC.md
- docs/ARCHITECTURE.md
- docs/XAI_CONTRACT.md
- docs/PATCHCORE_NOTES.md
- AGENTS.md

## Scope
Add a bounded multi-example path to the PatchCore bottle report so a local
static demo can show several anomalous test images, each with anomaly-map
overlay, top patch, nearest-normal provenance, distance summary, and
counterfactual preview.

## Out of scope
- Pretrained PatchCore feature maps.
- Coreset selection.
- Notebook authoring.
- New datasets beyond the prepared MVTec AD bottle category.

## Deliverables
- Report configuration and CLI option for number of examples.
- HTML rendering for multiple selected examples.
- Per-example generated assets with non-colliding filenames.
- Tests for multi-example report generation.
- README and PatchCore notes update.

## Constraints
- Preserve source image ids and patch coordinates.
- Keep reusable logic in `src/xai_demo_suite/`.
- Do not commit raw datasets, generated reports, caches, or data artefacts.
- Keep the single-example use case valid for smoke tests.

## Proposed file changes
- `src/xai_demo_suite/reports/patchcore_bottle.py` — multi-example scoring,
  asset naming, rendering, and demo card source figures.
- `src/xai_demo_suite/cli/demo.py` — expose `--max-examples`.
- `tests/unit/test_patchcore_report.py` — cover the multi-example path.
- `README.md` — document the new report command.
- `docs/PATCHCORE_NOTES.md` — record current report behaviour.

## Validation plan
1. `./.venv/bin/ruff check .`
2. `./.venv/bin/mypy src`
3. `./.venv/bin/pytest -q`
4. `./.venv/bin/xai-demo-report patchcore-bottle --max-train 2 --patch-size 128 --stride 128 --max-examples 3 --no-cache`

## Risks
- The report can become slow if too many examples are requested.
- The generated HTML can become too long if all per-example detail is expanded.
- Cached memory banks must still match the extractor and patch settings.

## Decision log
### 2026-04-15
- Decision: Add a conservative `max_examples` option defaulting to three.
- Reason: Three examples makes the demo visibly richer while keeping local
  generation and review manageable.
- Follow-up: Replace random-weight ResNet features with a proper pretrained
  PatchCore path in a later task.

## Progress log
### 2026-04-15
- Completed: Read the spec, agents, architecture, XAI contract, current report,
  tests, and PatchCore notes.
- Verification: Existing status was clean before edits.
- Remaining: Implement, regenerate, test, and commit.

### 2026-04-15
- Completed: Added `--max-examples`, repeated the mandatory PatchCore report
  views per selected example, wrote per-example assets, and updated docs/tests.
- Verification: `./.venv/bin/ruff check .`; `./.venv/bin/mypy src`;
  `./.venv/bin/pytest -q`; `./.venv/bin/xai-demo-report patchcore-bottle --max-train 2 --patch-size 128 --stride 128 --max-examples 3 --no-cache`.
- Remaining: None for this task.
