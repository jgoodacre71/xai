# 0001-foundation-tooling-and-contracts: foundation tooling and contracts

## Status
Complete

## Owner
Codex thread

## Why
The starter pack describes a durable XAI demo repository, but it does not yet
contain runnable package code, a Python toolchain, or tests. This task creates
the smallest useful foundation before adding datasets, PatchCore, or notebooks.

## Source of truth
- REPO_SPEC.md
- docs/ARCHITECTURE.md
- docs/XAI_CONTRACT.md
- docs/TESTING.md
- docs/decisions/ADR-0002-notebook-policy.md
- docs/decisions/ADR-0003-explanation-contract.md

## Scope
- Add project packaging and quality-tool configuration.
- Add the shared explanation artefact contracts.
- Add deterministic utility helpers.
- Add a tiny synthetic image fixture for tests and future smoke checks.
- Add unit tests for the initial contracts and fixture.
- Add a data registry placeholder for first-wave datasets.

## Out of scope
- Real dataset downloaders.
- PatchCore implementation.
- Model training.
- Notebooks and rendered reports.
- CI provider configuration.

## Deliverables
- `pyproject.toml`
- `.gitignore`
- `data_registry.yaml`
- `src/xai_demo_suite/`
- `tests/unit/`

## Constraints
- Keep notebooks thin; reusable logic belongs in `src/`.
- Do not commit raw datasets.
- Use UK English in docs, comments, and public text.
- Keep first checks fast enough for every task.

## Affected files
- `pyproject.toml` for build, lint, type, and test configuration.
- `.gitignore` for local data and generated artefacts.
- `src/xai_demo_suite/explain/contracts.py` for the shared XAI contract.
- `src/xai_demo_suite/utils/` for seed, IO, and logging helpers.
- `src/xai_demo_suite/data/synthetic/fixtures.py` for test fixtures.
- `tests/unit/` for initial regression coverage.

## Validation plan
1. `uv run ruff check .`
2. `uv run mypy src`
3. `uv run pytest -q`

## Acceptance criteria
- The package imports successfully.
- Explanation contract invariants are covered by tests.
- The synthetic fixture writes a deterministic image.
- Ruff, MyPy, and Pytest pass locally.
- No raw data or generated artefacts are tracked by default.

## Risks
- The initial contracts may need revision once PatchCore metadata becomes more concrete.
- The dependency set should remain small until real demos require heavier libraries.

## Progress log
### 2026-04-15
- Completed: initial toolchain, contracts, utilities, fixture, and tests.
- Verification: `./.venv/bin/ruff check .`, `./.venv/bin/mypy src`,
  and `./.venv/bin/pytest -q` all passed in the staged repo.
- Remaining: move the repo into `/Users/johngoodacre/work/xai` and create a
  fresh local environment there if needed.
