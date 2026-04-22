# Workflow

## Default approach

For non-trivial tasks:

1. read `AGENTS.md`, the Codex docs, and the relevant task/spec files
2. inspect the active code path before editing
3. state a short plan
4. make focused changes
5. run the smallest relevant validation set
6. summarise outcomes, risks, and the next step

## Inspect-before-edit discipline

Before editing, prefer reading:

- the relevant notebook first if the task is demo-facing
- the dataset adapter or model/helper path actually in use
- the relevant CLI entry point when the task is package- or command-facing
- the closest unit or integration test
- the relevant task file under `docs/tasks/`

Do not assume a report path is active just because a builder still exists.
Trace the notebook or package path that the current demo surface actually uses.

## Planning discipline

- use a task file for multi-step work
- keep the task file concrete enough that another thread can resume it
- update the task file when scope or validation changes materially

## Validation expectations

Prefer the narrowest useful validation slice:

- targeted `uv run pytest ...`
- `uv run ruff check .` when Python or imports changed
- `uv run mypy src` when types or interfaces changed
- `uv run xai-demo-report verify` when report semantics or generated structure changed
- notebook smoke for notebook-facing changes

If a task needs optional ML dependencies or prepared local data, say so
explicitly in the task and in the final validation summary.

For notebook-facing real-data demos such as Demo 01 and Demo 02, the preferred
validation pattern is:

- targeted notebook smoke through `tests/unit/test_notebook_smoke.py`
- direct top-to-bottom execution from the repo root
- direct top-to-bottom execution from the notebook's own folder when robust
  repo-root discovery is part of the contract
- strip stored outputs again before committing because the repo still enforces
  output-free notebooks in git

## Definition of done

A task is done when:

- the requested change is implemented cleanly
- reusable logic lives in package code when that helps maintainability and does
  not conflict with an explicit self-contained-notebook requirement
- relevant tests or smoke checks exist and have been run or clearly scoped
- docs or notebook narrative are updated if behaviour changed
- remaining risks are stated plainly

## When to update the Codex layer

Update `docs/codex/` when any of these change materially:

- active demo surface or recommended read-first path
- main CLI commands or validation commands
- data workflow or local dataset assumptions
- central architectural relationships
- whether the repo is notebook-first or report-first in practice
- important safety constraints for explanation claims or PatchCore provenance
