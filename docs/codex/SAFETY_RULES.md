# Safety Rules

## Repository-facing rules

- Use UK English in repo-facing docs, comments, notebooks, and user-facing text.
- Keep shared logic in `src/xai_demo_suite/`, not only in notebooks.
- Preserve the shared explanation contract:
  - evidence
  - provenance
  - counterfactual change
  - stability

## Data and artefact safety

- Do not commit raw datasets.
- Never overwrite files in `data/raw/`.
- Treat prepared datasets, generated outputs, and cached artefacts as local
  runtime products unless a task explicitly says otherwise.
- Prefer reproducible regeneration over editing generated artefacts by hand.

## Explanation and PatchCore safety

- PatchCore explanation work must preserve source image ids and patch
  coordinates.
- Do not replace provenance with untraceable approximations.
- Any explanation image used to support a claim should have a verification path,
  counter-test, or explicit caveat.
- Do not present local report output as a benchmark claim unless the task
  explicitly establishes that standard.

## Change scope

- Ask before destructive or broad structural changes.
- Preserve public CLI behaviour unless the task requires change.
- Keep notebook edits aligned with package code and tests.

## Documentation discipline

- Update docs when workflow, commands, or demo behaviour changes materially.
- Update the Codex docs when the active demo surface or working assumptions
  change.
- State uncertainty explicitly when behaviour depends on optional data, optional
  ML dependencies, or unverified local state.
