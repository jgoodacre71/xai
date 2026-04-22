# 0048: Import and review external SHAP notebook

## Status
Complete

## Owner
Codex thread

## Why

Import the external `shap.ipynb` notebook into this repository and assess
whether it belongs in the curated `xai` project surface.

## Source of truth

- `AGENTS.md`
- `REPO_SPEC.md`
- `docs/ARCHITECTURE.md`
- `docs/XAI_CONTRACT.md`
- `docs/DEMO_CATALOGUE.md`
- `docs/references/imported_shap.ipynb`

## Scope

- locate the notebook on disk
- import a copy into this repository
- review its content against the repo architecture and scope

## Out of scope

- cleaning or rewriting the notebook
- integrating SHAP into the active demo suite
- adding new dependencies to the project

## Deliverables

- imported notebook copy
- written review of its relevance and risks

## Constraints

- keep the curated notebook layer clean
- avoid promoting out-of-scope material into the main demo set
- preserve the imported notebook as reference material

## Proposed file changes

- `docs/references/imported_shap.ipynb`
  - imported external notebook copy
- `docs/references/shap_notebook_review.md`
  - repo-local analysis

## Validation plan

1. confirm the imported notebook exists in-repo;
2. confirm the imported copy parses as notebook JSON and has no stored outputs;
3. run `git diff --check`.

## Risks

- the notebook mixes relevant and irrelevant material
- the notebook uses a broader dependency surface than the current base repo
- internal SHAP hooks may be fragile if adopted directly

## Progress log

### 2026-04-22 00:00
- Completed: located `/Users/johngoodacre/code/playing-with-time-series/shap.ipynb`.
- Verification: notebook inspected for structure, headings, imports, and cell content.
- Remaining: import and write the review.

### 2026-04-22 00:00
- Completed: copied the notebook into `docs/references/imported_shap.ipynb` and
  added `docs/references/shap_notebook_review.md`.
- Verification: imported notebook parses and has no stored outputs.
- Remaining: none.
