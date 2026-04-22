# 0051: Notebook surface simplification

## Status
Complete

## Owner
Codex thread

## Why

After the storyline refactor, the repo still had transition-only notebook
artefacts that were no longer part of the active workflow. The goal of this
task was to simplify the project so the notebook tree is the single active demo
surface for now.

## Source of truth

- `AGENTS.md`
- `README.md`
- `docs/DEMO_CATALOGUE.md`
- `docs/references/shap_notebook_review.md`
- `notebooks/`

## Scope

- remove temporary notebook artefacts we are no longer using;
- remove notebook-local cache folders;
- update active docs to point only at the notebook-first surface;
- re-verify the notebook tree.

## Out of scope

- deeper notebook-by-notebook narrative polishing;
- major report-builder changes;
- removal of historical completed task files.

## Deliverables

- simplified active notebook tree;
- SHAP review updated to point at the active notebook copy;
- passing notebook structure and smoke tests.

## Validation plan

1. list active notebook and reference files;
2. run targeted notebook tests;
3. run `git diff --check`.

## Progress log

### 2026-04-22 00:00
- Completed: removed duplicate SHAP reference notebooks, removed the legacy
  duplicate PatchCore notebook, and deleted notebook `__pycache__` trees.
- Verification: active notebook tree inspected with `find`.
- Remaining: update docs and rerun notebook validation.

### 2026-04-22 00:00
- Completed: updated the active docs to point at the notebook-first surface and
  re-ran the notebook checks.
- Verification:
  `./.venv/bin/pytest tests/unit/test_notebooks.py tests/unit/test_notebook_smoke.py -q`;
  `git diff --check`
- Remaining: none.
