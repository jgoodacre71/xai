# Notebook Storylines

The notebooks are currently the primary demo surface for this repository.

They are organised by storyline rather than kept in one flat directory:

- `overview/`
  - repo-level orientation and walkthrough order
- `shortcut_lab/`
  - shortcut learning and spurious-correlation demos
- `patchcore_explainability/`
  - provenance-rich PatchCore explanations and normal-set pitfalls
- `patchcore_limits/`
  - count, severity, and logic limits
- `robustness_drift/`
  - explanation drift and robustness
- `global_local_explainability/`
  - focused concept notebooks such as SHAP-based global versus local evidence

The `.ipynb` notebooks are the active demo artefacts and remain output-free in
git.

For demos `01` to `08`, the notebook itself is expected to carry the runnable
demo logic, markdown story, and visible graphics. Those notebooks should not be
thin wrappers around generated HTML reports.
