# 0050: Storyline notebook refactor

## Status
Complete

## Owner
Codex thread

## Why

The repository currently has good package code and strong HTML report outputs,
but the notebook layer is still too thin for the current goal. For now, the
notebooks themselves are the demo surface. That means they need to be cleaned
up, organised by storyline, and able to run and show the relevant graphics and
outputs directly in the notebook rather than mainly redirecting the reader to
HTML pages.

## Source of truth

- `AGENTS.md`
- `REPO_SPEC.md`
- `docs/ARCHITECTURE.md`
- `docs/XAI_CONTRACT.md`
- `docs/DEMO_CATALOGUE.md`
- `docs/TESTING.md`
- `docs/references/global_vs_local_explainability_shap.ipynb`

## Scope

- reorganise notebooks into separate storyline directories;
- add shared notebook helpers for inline demo display;
- upgrade existing demo notebooks so they read as primary demo artefacts;
- promote the cleaned SHAP notebook into the notebook surface;
- update tests and docs for the new notebook structure.

## Out of scope

- full slide or presentation generation;
- replacing the package report builders;
- large changes to report semantics unrelated to notebook use;
- a full notebook-to-`src` extraction pass for the SHAP work.

## Deliverables

- storyline-based notebook layout;
- cleaner notebook scripts and paired notebooks;
- inline report rendering in notebooks;
- SHAP notebook included in the notebook surface;
- updated tests and documentation.

## Constraints

- keep reusable logic in `src/xai_demo_suite/` where practical;
- preserve the existing report builders as the main computation path;
- use UK English in notebook markdown and docs;
- keep the notebooks runnable in smoke mode.

## Proposed file changes

- `src/xai_demo_suite/notebooks.py`
  - shared notebook runtime helpers
- `notebooks/`
  - reorganise into storyline directories
- `tests/unit/test_notebooks.py`
  - recurse through the new notebook layout
- `tests/unit/test_notebook_smoke.py`
  - point smoke checks at the new scripts
- docs that reference notebook paths
  - update to the new storyline structure

## Validation plan

1. notebook JSON output-free checks;
2. targeted notebook smoke execution;
3. targeted notebook structure tests;
4. `git diff --check`.

## Risks

- the notebook layer can become too heavy if too much logic moves out of `src`;
- recursive notebook-path changes can break tests and docs if done partially;
- the SHAP notebook may still be more prototype than production demo even after
  promotion into the notebook surface.

## Decision log

### 2026-04-22 00:00
- Decision: treat notebooks as the primary demo surface for now.
- Reason: this matches the current user goal better than keeping notebooks as
  thin wrappers around external HTML.
- Follow-up: preserve package-code reuse while making notebooks much more
  presentable and runnable.

## Progress log

### 2026-04-22 00:00
- Completed: mapped the current notebook layer, report builders, and output
  structure.
- Verification: inspected notebooks, tests, and report modules.
- Remaining: implement the storyline reorganisation and cleaner notebook
  runtime.

### 2026-04-22 00:00
- Completed: reorganised the notebooks into storyline directories, added a
  shared notebook runtime helper, moved the legacy duplicate PatchCore notebook
  into references, and promoted the cleaned SHAP notebook into the notebook
  surface.
- Verification: paired `.py` and `.ipynb` files regenerated across the new
  notebook tree.
- Remaining: update tests and docs for the new layout.

### 2026-04-22 00:00
- Completed: updated the notebook tests, README, demo catalogue, demo status,
  and Codex docs to reflect the storyline layout and notebook-first workflow.
- Verification:
  `./.venv/bin/pytest tests/unit/test_notebooks.py tests/unit/test_notebook_smoke.py -q`;
  `./.venv/bin/ruff check src/xai_demo_suite/notebooks.py tests/unit/test_notebooks.py tests/unit/test_notebook_smoke.py`;
  `./.venv/bin/python - <<'PY' ... py_compile ... PY`;
  `git diff --check`
- Remaining: none for this structural refactor; future work can polish the
  notebooks one by one.
