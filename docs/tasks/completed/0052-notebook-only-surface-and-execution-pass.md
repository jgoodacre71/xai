# 0052: Notebook-only surface and execution pass

## Status
Complete

## Owner
Codex thread

## Why

After moving to a notebook-first workflow, the repo still had duplicate
notebook-source files and stale generated-output directories. The goal of this
task was to simplify the active surface to runnable `.ipynb` notebooks and then
execute the active notebooks to verify the stories still run cleanly.

## Scope

- remove duplicate notebook `.py` files;
- remove stale generated output folders not part of the active workflow;
- update tests and active docs to the notebook-only model;
- execute each active non-SHAP notebook and verify the generated outputs.

## Deliverables

- notebook-only active notebook tree under `notebooks/`;
- simplified `outputs/` tree containing only the active smoke-built artefacts;
- direct notebook smoke execution tests;
- repo docs aligned to the notebook-only workflow.

## Validation

- executed active notebooks directly from `.ipynb` files in smoke mode
- `./.venv/bin/ruff check tests/unit/test_notebooks.py tests/unit/test_notebook_smoke.py`
- `./.venv/bin/pytest tests/unit/test_notebooks.py tests/unit/test_notebook_smoke.py -q`
- `git diff --check`

## Progress log

### 2026-04-22 00:00
- Completed: removed duplicate notebook `.py` files and stale output folders
  `outputs_diag/` and `outputs_smoke/`.
- Verification: active notebook tree and output tree re-listed after cleanup.
- Remaining: update tests and docs, then execute the notebooks.

### 2026-04-22 00:00
- Completed: switched notebook smoke tests to execute `.ipynb` files directly
  and updated active docs away from paired-script language.
- Verification: notebook tests and Ruff checks passed.
- Remaining: run the active notebooks and inspect the resulting outputs.

### 2026-04-22 00:00
- Completed: executed the active non-SHAP notebooks in smoke mode and rebuilt
  the active `outputs/` tree.
- Verification: all notebook runs completed successfully, every active story
  notebook retained the required narrative sections, and the rebuilt report HTML
  files contained figures.
- Remaining: none.
