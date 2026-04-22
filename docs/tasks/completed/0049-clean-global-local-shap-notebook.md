# 0049: Clean global versus local SHAP notebook

## Status
Complete

## Owner
Codex thread

## Why

An imported external notebook contains a potentially useful XAI strand around
synthetic X-ray imagery, linear and CNN models, and SHAP-based explanations, but
it is mixed with unrelated time-series and betting material. The notebook needs
to be cleaned into a repo-appropriate artefact focused on global versus local
explainability.

## Source of truth

- `AGENTS.md`
- `REPO_SPEC.md`
- `docs/ARCHITECTURE.md`
- `docs/XAI_CONTRACT.md`
- `docs/DEMO_CATALOGUE.md`
- `docs/references/imported_shap.ipynb`
- `docs/references/shap_notebook_review.md`
- `docs/references/global_vs_local_explainability_shap.ipynb`

## Scope

- identify the XAI-relevant cells in the imported notebook
- remove unrelated non-vision material
- tighten the notebook framing around global versus local explainability
- place the cleaned notebook in a suitable repo location
- assess notebook quality and integration options for this project

## Out of scope

- full productionisation into `src/`
- adding new package dependencies
- adding tests for notebook-only helper code
- creating a full new demo end-to-end

## Deliverables

- cleaned notebook copy
- updated notebook review and integration assessment

## Constraints

- keep the cleaned notebook aligned with the repo's vision-XAI scope
- use UK English in markdown
- avoid promoting unrelated scratchpad material into the curated notebook set

## Proposed file changes

- `docs/references/imported_shap.ipynb`
  - inspect source notebook structure
- `docs/references/global_vs_local_explainability_shap.ipynb`
  - cleaned notebook copy focused on global versus local explainability
- `docs/references/shap_notebook_review.md`
  - refine the quality and integration assessment after cleanup

## Validation plan

1. parse the cleaned notebook as JSON;
2. inspect the resulting headings and cell counts;
3. run `git diff --check`.

## Risks

- the notebook may still be too monolithic even after trimming
- the current code may depend on optional packages not installed in the base repo
- a cleaned notebook can still fall short of repo standards if logic remains notebook-only

## Decision log

### 2026-04-22 00:00
- Decision: start by cleaning the imported notebook into a focused local artefact before deciding whether it deserves full integration.
- Reason: the imported notebook mixes relevant XAI content with clearly unrelated material.
- Follow-up: assess whether the cleaned result belongs in `notebooks/` or should remain a reference notebook.

## Progress log

### 2026-04-22 00:00
- Completed: imported and reviewed the external notebook.
- Verification: notebook structure, headings, and dependency surface inspected.
- Remaining: produce the cleaned XAI-focused notebook and integration assessment.

### 2026-04-22 00:00
- Completed: created a cleaned derivative notebook containing only the vision/XAI
  strand and added clearer framing around global versus local explainability.
- Verification: cleaned notebook parses as JSON, retains no stored outputs, and
  has coherent markdown headings from start to finish.
- Remaining: update the written review and complete final validation.

### 2026-04-22 00:00
- Completed: updated the written assessment to reflect the cleaned copy and its
  integration path.
- Verification: `git diff --check`.
- Remaining: none.
