# 0054: Data replication, Demo 00, and storyline alignment

## Status
Completed

## Owner
Codex thread

## Why
The next phase reframes the repository around XAI as model-behaviour
observability: what the model learned, what evidence it used, what would change
the decision, and what risk remains after mitigation.

## Scope
- Add the no-permission Moons/Stars Clever-Hans opener.
- Add data requirements, permission tracking, and work-replication docs.
- Add an IEEE DataPort scouting register.
- Align README, notebook index, demo catalogue, and demo status around the
  conceptual demo order.
- Keep Demo 01 and Demo 02 modelling intact.

## Completion notes
- Added Demo 00 as a generated controlled absolute-position Clever-Hans audit:
  moons usually lower-left, stars usually upper-right, Pixel MLP learns
  location, and a translation-aware CNN with position augmentation is more
  shape-stable.
- Demo 00 now includes movement counterfactuals, position-response maps,
  decision surfaces, shape morphs, saliency caveats, area-normalised relevance,
  average relevance maps, representation neighbours, representation probes,
  minimal evidence removal, a summary panel, and a final evidence ledger.
- Added data requirements, permission matrix, replication workflow, IEEE
  scouting docs/register, data inventory script, and notebook/data-status tests.
- Updated README, notebook index, demo catalogue, demo status, and data docs
  around the conceptual order: controlled shortcut -> industrial shortcut ->
  natural benchmark -> anomaly provenance -> limits -> drift -> dataset
  scouting.
- Demo 01 and Demo 02 modelling were left intact apart from positioning and
  documentation alignment.

## Out of scope
- Rebuilding Demo 01 or Demo 02.
- Reworking PatchCore model logic.
- Selecting or downloading a specific IEEE dataset.
- Committing raw data, processed manifests, or generated notebook outputs.

## Validation plan
- `/Users/johngoodacre/miniforge3/envs/qst/bin/jupyter-nbconvert --execute notebooks/shortcut_lab/00_moons_stars_clever_hans.ipynb --to notebook --inplace`
- `/Users/johngoodacre/miniforge3/envs/qst/bin/jupyter-nbconvert --execute 00_moons_stars_clever_hans.ipynb --to notebook --inplace` from `notebooks/shortcut_lab`
- `./.venv/bin/pytest tests/unit/test_notebooks.py -q`
- `./.venv/bin/pytest tests/unit/test_notebook_smoke.py -q`
- `./.venv/bin/ruff check tests/unit/test_notebooks.py`
- `git diff --check`

## Risks
- Work-use permissions may block some real datasets.
- MVTec-family non-commercial terms may require a VisA or IEEE alternative at
  work.
- Generated Demo 00 is didactic and should not be overclaimed as real-world
  evidence.
- Existing user edits in notebooks `02` to `08` must not be overwritten.
