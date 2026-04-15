# 0009-demo-card-and-local-index: demo card and local index

## Status
Complete

## Owner
Codex thread

## Why
REPO_SPEC.md section 17 says each major demo release should ship a short demo
card summarising task, model, explanation methods, key lesson, failure mode,
intervention, and caveats. The PatchCore bottle report now has the core visual
views, but the summary still lives in docs and chat rather than in generated
demo artefacts.

## Source of Truth
- AGENTS.md
- .agents/PLANS.md
- REPO_SPEC.md section 17
- docs/DEMO_CATALOGUE.md
- docs/PATCHCORE_NOTES.md
- docs/tasks/completed/0008-patchcore-counterfactual-preview.md

## Scope
- Add a reusable demo card data structure and HTML/JSON writer.
- Generate a PatchCore bottle demo card next to the report.
- Generate a local demo index linking to the report and card.
- Add tests with temporary output directories.
- Update docs with the generated artefact locations.

## Out of Scope
- Public website design.
- Notebook generation.
- GitHub Pages deployment.
- Multiple demo cards beyond the current PatchCore bottle slice.

## Deliverables
- `src/xai_demo_suite/reports/cards.py`
- report integration
- tests
- docs update

## Constraints
- Generated cards and index files remain under ignored output paths.
- Keep report/card copy honest about current limitations.
- Keep generated artefacts reproducible from package code.

## Proposed File Changes
- `src/xai_demo_suite/reports/cards.py`
- `src/xai_demo_suite/reports/patchcore_bottle.py`
- `tests/unit/test_demo_cards.py`
- `tests/unit/test_patchcore_report.py`
- `docs/PATCHCORE_NOTES.md`
- `README.md`

## Validation Plan
1. `./.venv/bin/ruff check .`
2. `./.venv/bin/mypy src`
3. `./.venv/bin/pytest -q`
4. `./.venv/bin/xai-demo-report patchcore-bottle --max-train 2 --patch-size 128 --stride 128 --no-cache`

## Acceptance Criteria
- Report generation writes `demo_card.json` and `demo_card.html`.
- Report generation writes a local `outputs/index.html`.
- Demo card includes every section required by REPO_SPEC.md section 17.
- Tests verify card JSON/HTML and index generation.
- Full checks and local report smoke command pass.

## Risks
- The card might read like a final release although the model is still a
  random-weight ResNet patch-crop baseline. The caveats must make this explicit.

## Decision Log
### 2026-04-15
- Decision: generate cards from package code, not hand-written output files.
- Reason: generated artefacts should be reproducible and not committed.
- Follow-up: future demos can reuse the same card/index writer.

## Progress Log
### 2026-04-15
- Completed: reusable demo card/index writer, PatchCore bottle report
  integration, docs, tests, and local smoke generation.
- Verification: `./.venv/bin/ruff check .`, `./.venv/bin/mypy src`,
  `./.venv/bin/pytest -q`, and
  `./.venv/bin/xai-demo-report patchcore-bottle --max-train 2 --patch-size 128 --stride 128 --no-cache`
  passed.
- Remaining: future demo releases should reuse `DemoCard`.
