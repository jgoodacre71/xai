# 0013: PatchCore Narrative Notebook

## Status
Complete

## Owner
Codex thread

## Why
The PatchCore hero demo now has a generated static report, demo card, selected
figures, provenance views, counterfactual preview, and mask verification. The
release artefact list in the repository spec also asks for a notebook. The
notebook should tell the story while keeping logic in importable package code.

## Source of truth
- REPO_SPEC.md
- AGENTS.md
- docs/ARCHITECTURE.md
- docs/XAI_CONTRACT.md
- docs/decisions/ADR-0002-notebook-policy.md
- docs/runbooks/add_demo.md

## Scope
Create an output-free narrative notebook for Demo 03 that calls the report
builder from `src/`, documents the four XAI questions, and points readers to
the generated static artefacts.

## Out of scope
- Moving report logic into the notebook.
- Executing the notebook in CI with Jupyter.
- Adding new model training.
- Committing generated report outputs.

## Deliverables
- `notebooks/03_patchcore_mvtec_bottle.ipynb`
- Notebook smoke test using the standard library JSON parser.
- Demo catalogue and docs update if needed.

## Constraints
- Notebook must be a presentation artefact only.
- Notebook must have no saved outputs.
- Code cells must call package APIs rather than reimplementing demo logic.
- UK English in markdown and user-facing text.

## Proposed file changes
- `notebooks/03_patchcore_mvtec_bottle.ipynb` — narrative notebook.
- `tests/unit/test_notebooks.py` — output-free and package-code smoke checks.
- `docs/DEMO_CATALOGUE.md` — record the notebook/report artefacts.
- `README.md` — mention the notebook.

## Validation plan
1. `./.venv/bin/ruff check .`
2. `./.venv/bin/mypy src`
3. `./.venv/bin/pytest -q`

## Risks
- Without nbconvert in the local venv, the smoke test can validate structure
  but not execute cells.
- If the notebook grows too much, it could drift into business logic.

## Decision log
### 2026-04-15
- Decision: Add an output-free `.ipynb` and a standard-library smoke test.
- Reason: This satisfies the checked-in narrative artefact requirement without
  adding a Jupyter dependency to the base environment.
- Follow-up: Add executed HTML notebook export once the notebook dependency
  group is installed in the project environment.

## Progress log
### 2026-04-15
- Completed: Confirmed there is no existing notebook and local
  `./.venv/bin/jupyter-nbconvert` is unavailable.
- Verification: `git status --short` was clean before edits.
- Remaining: Create notebook, add smoke test, update docs, and commit.

### 2026-04-15
- Completed: Added the output-free PatchCore narrative notebook, notebook smoke
  tests, README entry, and demo catalogue entry.
- Verification: `./.venv/bin/ruff check .`; `./.venv/bin/mypy src`;
  `./.venv/bin/pytest -q`.
- Remaining: None for this task. Notebook execution/export is deferred until
  the notebook dependency group is installed in the local environment.
